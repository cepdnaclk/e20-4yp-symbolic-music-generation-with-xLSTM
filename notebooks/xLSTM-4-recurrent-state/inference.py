import os
import glob
import time
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

import sys
# We need to add the repos folder to the sys.path so we can import the exact models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../repos')))
from helibrunna.source.utilities import model_from_config
from helibrunna.source.languagemodel import LanguageModel as HelibrunnaLanguageModel # For comparison later

class xLSTMGenerator:
    """
    A fast, O(N) autoregressive generator for xLSTM models.
    Bypasses helibrunna's parallel formulation generation loop and uses
    the core xLSTM recurrent `step(state)` method as intended by the authors.
    """
    def __init__(self, model_path_or_repo, config_overrides=None, device="auto", checkpoint_name=None):
        if config_overrides is None:
            config_overrides = {}
            
        # 1. Resolve Device
        self.device = self._resolve_device(device)
        print(f"Loading xLSTM Generator on device: {self.device}")

        # 2. Resolve Model Paths (Using helibrunna's standard logic)
        model_path, tokenizer_path = self._resolve_paths(model_path_or_repo, checkpoint_name)

        # 3. Load Config
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}")
        self.config = OmegaConf.load(config_path)
        if config_overrides:
            self.config = OmegaConf.merge(self.config, config_overrides)

        # 4. Initialize Core xLSTM Model
        self.model = model_from_config(self.config, device=self.device)
        self.model.to(self.device)
        
        # 5. Load Weights
        weights_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights not found at {weights_path}")
        
        print("Loading safetensors weights into model...")
        state_dict = load_file(weights_path)
        
        # Safety for older helibrunna weight saves without cuda
        if not torch.cuda.is_available() and self.config.get("type", "xLSTMLMModel") == "xLSTMLMModel":
            endings = ["xlstm.slstm_cell._recurrent_kernel_"]
            for key, values in state_dict.items():
                for ending in endings:
                    if key.endswith(ending):
                        values = values.permute(0, 2, 1)
                        state_dict[key] = values
                        break

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 6. Load Tokenizer
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_json_path}")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_json_path)
        print("Model and Tokenizer loaded successfully.")

    def _resolve_device(self, device):
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_paths(self, repo, checkpoint_name):
        model_path = None
        tokenizer_path = repo

        if checkpoint_name is not None:
             model_path = os.path.join(repo, checkpoint_name)
        else:
             checkpoint_folders = glob.glob(os.path.join(repo, "checkpoint-*"))
             for checkpoint_folder in checkpoint_folders:
                 if checkpoint_folder.endswith("-last"):
                     model_path = checkpoint_folder
                     break
        
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("No valid model checkpoint found.")
            
        if os.path.exists(os.path.join(repo, "tokenizer.json")):
             tokenizer_path = repo
             
        return model_path, tokenizer_path

    @torch.no_grad()
    def prefill(self, inputs: torch.Tensor):
        """
        Sequentially runs through the input prompt tokens using model.step()
        to build the initial recurrent state dictionary.
        
        Args:
            inputs: Tensor of shape (1, Sequence_Length)
        Returns:
            state: The Recurrent State dictionary up to the last token of the prompt.
            last_logits: The logits for the *final* token in the prompt sequence.
        """
        state = {} # Initialize empty recurrent state
        seq_length = inputs.shape[1]
        
        # Process every token in the prompt one by one
        for i in range(seq_length):
            # Extract just this single token shape (1, 1)
            token = inputs[:, i:i+1]
            
            # Forward step updates the internal state dynamically
            logits, state = self.model.step(token, state=state)
            
        # Return the built-up state and the very last prediction logits
        return state, logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = 100,
        end_tokens: list[str] = [],
        forbidden_tokens: list[str] = [],
        return_structured_output: bool = False
    ):
        """
        Fast recurrent generation using single token steps and propagating
        recurrent state.
        """
        # 1. Setup Data
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert inputs.shape[0] == 1, "Only batch size 1 is supported."
        
        end_token_ids = []
        for end_token in end_tokens:
            if end_token in self.tokenizer.vocab:
                end_token_ids.append(self.tokenizer(end_token).input_ids[0])
                
        # Determine the forbidden tokens ids
        ids_to_mask = []
        for forbidden_token in forbidden_tokens:
            if forbidden_token in self.tokenizer.vocab:
                ids_to_mask.extend(self.tokenizer(forbidden_token).input_ids)

        start_time = time.time()
        
        # 2. Sequential Prefill Step
        # This builds the state for all the prompt tokens except we don't need to predict for them
        state, logits = self.prefill(inputs)
        
        tokens_count = 0
        sequence_length = inputs.shape[1]
        
        # We start our generation with the logits returned by the LAST token of the prefill
        
        outputs_list = [inputs] # Keep track of the full generated sequence tensor
        
        # 3. Recurrent Generation Loop
        while sequence_length < max_length:
            
            # Apply Temperature and Softmax to the extracted logits
            # xLSTM block step returns raw logits.
            
            # Mask out special ids:
            logits[:, :, self.tokenizer.all_special_ids] = float("-inf")
            # Mask out explicitly forbidden tokens:
            if ids_to_mask:
                logits[:, :, ids_to_mask] = float("-inf")
            
            scaled_logits = logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs[0, -1], num_samples=1)
            next_token = next_token.unsqueeze(0) # Shape (1, 1)
            
            # Append generated token to inputs
            outputs_list.append(next_token)
            tokens_count += 1
            sequence_length += 1
            
            # Check for End Tokens
            if next_token[0, 0].item() in end_token_ids:
                break
                
            # Stop condition early if max context len reached (though xlstm is theoretically infinite)
            if sequence_length >= self.config.context_length:
                print("Warning: The maximum trained context length has been reached.")
                # We do not strictly break here because xLSTM can extrapolate linearly
                
            # If not done, perform the recurrent step for the NEXT token!
            logits, state = self.model.step(next_token, state=state)

        # Reconstruct full tensor
        final_sequence = torch.cat(outputs_list, dim=1)
        
        elapsed_time = time.time() - start_time
        tokens_per_second = tokens_count / elapsed_time if elapsed_time > 0 else 0

        output_str = self.tokenizer.decode(final_sequence[0].tolist())

        if return_structured_output:
            return {
                "output": output_str,
                "elapsed_time": elapsed_time,
                "tokens_per_second": tokens_per_second
            }
        return output_str

