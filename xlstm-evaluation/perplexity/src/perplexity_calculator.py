"""
Perplexity Calculator Module

Computes perplexity over token sequences using teacher forcing.
"""

import math
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Any


def compute_perplexity(
    model,
    sequences: List[torch.Tensor],
    max_context: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Compute perplexity over token sequences using teacher forcing.
    
    Following Museformer's approach: calculate PPL on the first N tokens
    of each sequence.
    
    Args:
        model: Helibrunna LanguageModel instance
        sequences: List of tokenized sequences (each is a 1D tensor)
        max_context: Truncate to first N tokens (None = full sequence)
        show_progress: Show tqdm progress bar
        
    Returns:
        Dictionary with perplexity results and statistics
    """
    total_nll = 0.0
    total_tokens = 0
    skipped_sequences = 0
    sequence_results = []
    
    pad_idx = model.tokenizer.pad_token_id
    
    # Ensure model is in eval mode
    model.model.eval()
    
    desc = f"PPL@{max_context}" if max_context else "PPL@full"
    iterator = tqdm(sequences, desc=desc) if show_progress else sequences
    
    with torch.no_grad():
        for seq_idx, tokens in enumerate(iterator):
            original_length = len(tokens)
            
            # Truncate if specified
            if max_context and len(tokens) > max_context:
                tokens = tokens[:max_context]
            
            # Skip very short sequences (need at least 2 tokens for teacher forcing)
            if len(tokens) < 2:
                skipped_sequences += 1
                continue
            
            tokens = tokens.to(model.device)
            
            # Teacher forcing shift: 
            # - inputs: all tokens except last
            # - targets: all tokens except first
            inputs = tokens[:-1].unsqueeze(0)   # [1, seq_len-1]
            targets = tokens[1:]                 # [seq_len-1]
            
            try:
                # Get model predictions (logits)
                logits = model.predict(inputs).squeeze(0)  # [seq_len-1, vocab_size]
                
                # Compute log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get NLL for each position
                # nll[i] = -log P(targets[i] | inputs[:i+1])
                nll = -log_probs[torch.arange(len(targets), device=model.device), targets]
                
                # Mask padding tokens if present
                if pad_idx is not None:
                    mask = targets != pad_idx
                    valid_nll = nll[mask]
                else:
                    valid_nll = nll
                
                seq_nll = valid_nll.sum().item()
                seq_tokens = len(valid_nll)
                
                total_nll += seq_nll
                total_tokens += seq_tokens
                
                # Store per-sequence results
                sequence_results.append({
                    "original_length": original_length,
                    "evaluated_length": len(tokens),
                    "tokens_scored": seq_tokens,
                    "avg_nll": seq_nll / seq_tokens if seq_tokens > 0 else float('inf'),
                    "ppl": math.exp(seq_nll / seq_tokens) if seq_tokens > 0 else float('inf')
                })
                
            except Exception as e:
                if show_progress:
                    print(f"\nError on sequence {seq_idx}: {str(e)[:100]}")
                skipped_sequences += 1
                continue
    
    # Compute overall statistics
    if total_tokens == 0:
        return {
            "perplexity": float('inf'),
            "avg_nll": float('inf'),
            "total_tokens": 0,
            "error": "No tokens evaluated"
        }
    
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll) if avg_nll < 100 else float('inf')  # Avoid overflow
    
    return {
        "perplexity": ppl,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "num_sequences": len(sequences),
        "num_evaluated": len(sequence_results),
        "skipped_sequences": skipped_sequences,
        "max_context": max_context,
        "sequence_results": sequence_results  # Per-sequence breakdown
    }


def compute_perplexity_multi_context(
    model,
    sequences: List[torch.Tensor],
    context_lengths: List[int] = [1024, 5120, 10240, 16384],
    show_progress: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Compute perplexity at multiple context lengths.
    
    Args:
        model: Helibrunna LanguageModel instance
        sequences: List of tokenized sequences
        context_lengths: List of context lengths to evaluate at
        show_progress: Show progress
        
    Returns:
        Dictionary mapping context_length -> results
    """
    results = {}
    
    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Computing PPL at context length: {ctx_len}")
        print(f"{'='*60}")
        
        # Filter sequences that have at least ctx_len tokens
        valid_sequences = [s for s in sequences if len(s) >= ctx_len]
        print(f"Sequences with >= {ctx_len} tokens: {len(valid_sequences)}/{len(sequences)}")
        
        if len(valid_sequences) == 0:
            results[ctx_len] = {
                "perplexity": float('nan'),
                "error": f"No sequences have >= {ctx_len} tokens"
            }
            continue
        
        results[ctx_len] = compute_perplexity(
            model, 
            valid_sequences, 
            max_context=ctx_len,
            show_progress=show_progress
        )
        
        print(f"PPL@{ctx_len}: {results[ctx_len]['perplexity']:.4f}")
        print(f"Tokens evaluated: {results[ctx_len]['total_tokens']:,}")
    
    return results


def print_results_summary(results: Dict[str, Any]) -> None:
    """Pretty print perplexity results."""
    print("\n" + "=" * 60)
    print("PERPLEXITY RESULTS")
    print("=" * 60)
    
    print(f"\nPerplexity:          {results['perplexity']:.4f}")
    print(f"Average NLL:         {results['avg_nll']:.4f}")
    print(f"Total tokens:        {results['total_tokens']:,}")
    print(f"Sequences evaluated: {results.get('num_evaluated', 'N/A')}")
    print(f"Sequences skipped:   {results.get('skipped_sequences', 0)}")
    print(f"Max context:         {results.get('max_context', 'full')}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the perplexity calculator
    import sys
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perplexity_dir = os.path.dirname(script_dir)
    sys.path.insert(0, perplexity_dir)
    sys.path.insert(0, "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna")
    
    # Set CUDA env vars
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
    os.environ['MAX_JOBS'] = '4'
    
    from src.config import MODEL_PATH, TEST_DATA_PATH
    from src.data_loader import load_tokenized_sequences
    from source.languagemodel import LanguageModel
    
    print("Loading model...")
    model = LanguageModel(MODEL_PATH, device="cuda")
    print(f"Model loaded! Vocab size: {len(model.tokenizer)}")
    
    print("\nLoading 5 test sequences...")
    sequences = load_tokenized_sequences(TEST_DATA_PATH, model.tokenizer, max_samples=5)
    print(f"Loaded {len(sequences)} sequences")
    
    print("\nComputing perplexity at context=1024...")
    results = compute_perplexity(model, sequences, max_context=1024)
    print_results_summary(results)
