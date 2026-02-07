import os
import math
import torch
import argparse
from tqdm import tqdm
import sys
import os

# Ensure museformer is on PYTHONPATH
sys.path.append(os.path.abspath("."))

import sys, os
sys.path.insert(0, os.path.abspath("."))   # ensures repo root is on PYTHONPATH

# 🔑 this must run BEFORE loading checkpoint (registers the task)
import museformer.museformer_lm_task



from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.dictionary import Dictionary

#############################################
# CONFIG
#############################################

CHECKPOINT_PATH = (
    "/scratch1/e20-fyp-xlstm-music-generation/"
    "e20fyptemp1/fyp-musicgen/repos/muzic/museformer/"
    "checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
)

TEST_MIDI_DIR = (
    "/scratch1/e20-fyp-xlstm-music-generation/"
    "e20fyptemp1/fyp-musicgen/repos/muzic/museformer/"
    "data/example_test"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = "<pad>"

#############################################
# STEP 1: Load Museformer model via fairseq
#############################################

def load_museformer_model(ckpt_path):
    print("Loading Museformer checkpoint...")

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path],
        strict=True
    )

    model = models[0]
    model.eval()
    model.to(DEVICE)

    dictionary = task.source_dictionary

    print("Model loaded successfully")
    return model, task, dictionary


#############################################
# STEP 2: Encode MIDI → tokens
#############################################

def encode_midi_to_tokens(task, midi_path):
    """
    Uses Museformer task preprocessing to convert MIDI to token IDs.
    """
    dataset = task.dataset("test")

    # Museformer expects pre-tokenized data normally,
    # but this utility forces on-the-fly encoding
    sample = task.load_dataset_from_file(
        midi_path,
        split="test"
    )

    tokens = sample["source"]
    return tokens


#############################################
# STEP 3: Compute PPL manually
#############################################

def compute_ppl(model, dictionary, token_sequences):
    total_nll = 0.0
    total_tokens = 0

    pad_idx = dictionary.index(PAD_TOKEN)

    with torch.no_grad():
        for tokens in tqdm(token_sequences, desc="Evaluating"):
            tokens = tokens.to(DEVICE)

            # Autoregressive shift
            inputs = tokens[:-1]
            targets = tokens[1:]

            logits = model(
                src_tokens=inputs.unsqueeze(0),
                src_lengths=torch.tensor([len(inputs)]).to(DEVICE)
            )[0].squeeze(0)

            log_probs = torch.log_softmax(logits, dim=-1)

            mask = targets != pad_idx
            nll = -log_probs[torch.arange(len(targets)), targets]

            total_nll += nll[mask].sum().item()
            total_tokens += mask.sum().item()

    ppl = math.exp(total_nll / total_tokens)
    return ppl


#############################################
# MAIN
#############################################

def main():
    model, task, dictionary = load_museformer_model(CHECKPOINT_PATH)

    midi_files = [
        os.path.join(TEST_MIDI_DIR, f)
        for f in os.listdir(TEST_MIDI_DIR)
        if f.endswith(".mid") or f.endswith(".midi")
    ]

    if len(midi_files) == 0:
        raise RuntimeError("No MIDI files found in test directory")

    print(f"Found {len(midi_files)} test MIDI files")

    token_sequences = []

    for midi_path in midi_files:
        tokens = encode_midi_to_tokens(task, midi_path)
        token_sequences.append(tokens)

    ppl = compute_ppl(model, dictionary, token_sequences)

    print("\n====================================")
    print(f"Museformer Test Perplexity: {ppl:.4f}")
    print("====================================")


if __name__ == "__main__":
    main()
