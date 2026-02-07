"""
Data Loader Module for Perplexity Evaluation

Handles loading and processing of tokenized music sequences from text files.
Each line in the input file represents one song, with tokens separated by spaces.
"""

import os
from typing import List, Optional, Dict, Any
import torch
from tqdm import tqdm


def load_tokenized_sequences(
    file_path: str,
    tokenizer,
    max_samples: Optional[int] = None,
    show_progress: bool = True
) -> List[torch.Tensor]:
    """
    Load tokenized sequences from a text file.
    
    Args:
        file_path: Path to the text file (e.g., test.txt)
        tokenizer: Tokenizer instance (from Helibrunna LanguageModel)
        max_samples: Limit number of sequences to load (None = all)
        show_progress: Show tqdm progress bar
        
    Returns:
        List of PyTorch tensors, each containing token IDs for one song
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    sequences = []
    
    # Count lines first for progress bar
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_samples is not None:
        lines = lines[:max_samples]
    
    iterator = tqdm(lines, desc="Loading sequences") if show_progress else lines
    
    for line in iterator:
        line = line.strip()
        if not line:
            continue
        
        # Tokenize the space-separated token string
        try:
            token_ids = tokenizer.encode(line, return_tensors="pt")
            # squeeze to remove batch dimension if present
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze(0)
            sequences.append(token_ids)
        except Exception as e:
            if show_progress:
                print(f"\nWarning: Failed to tokenize line: {str(e)[:50]}...")
            continue
    
    return sequences


def get_sequence_stats(sequences: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Get statistics about sequence lengths.
    
    Args:
        sequences: List of token tensors
        
    Returns:
        Dictionary with length statistics
    """
    if not sequences:
        return {"error": "No sequences provided"}
    
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        "num_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
        "total_tokens": sum(lengths),
    }
    
    # Count sequences at or above each context length threshold
    thresholds = [1024, 2048, 5120, 10240, 16384]
    for threshold in thresholds:
        count = sum(1 for l in lengths if l >= threshold)
        stats[f"sequences_at_or_above_{threshold}"] = count
        stats[f"percentage_at_or_above_{threshold}"] = (count / len(sequences)) * 100
    
    return stats


def print_sequence_stats(stats: Dict[str, Any]) -> None:
    """
    Pretty print sequence statistics.
    
    Args:
        stats: Dictionary from get_sequence_stats()
    """
    print("\n" + "=" * 60)
    print("SEQUENCE STATISTICS")
    print("=" * 60)
    
    print(f"\nBasic Stats:")
    print(f"  Total sequences:     {stats['num_sequences']:,}")
    print(f"  Total tokens:        {stats['total_tokens']:,}")
    print(f"  Min length:          {stats['min_length']:,}")
    print(f"  Max length:          {stats['max_length']:,}")
    print(f"  Mean length:         {stats['mean_length']:,.1f}")
    print(f"  Median length:       {stats['median_length']:,}")
    
    print(f"\nContext Length Coverage:")
    for threshold in [1024, 2048, 5120, 10240, 16384]:
        count = stats.get(f"sequences_at_or_above_{threshold}", 0)
        pct = stats.get(f"percentage_at_or_above_{threshold}", 0)
        print(f"  >= {threshold:>6} tokens:   {count:>5} sequences ({pct:>5.1f}%)")
    
    print("=" * 60)


def filter_by_length(
    sequences: List[torch.Tensor],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Filter sequences by length.
    
    Args:
        sequences: List of token tensors
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        
    Returns:
        Filtered list of sequences
    """
    filtered = sequences
    
    if min_length is not None:
        filtered = [s for s in filtered if len(s) >= min_length]
    
    if max_length is not None:
        filtered = [s for s in filtered if len(s) <= max_length]
    
    return filtered


if __name__ == "__main__":
    # Test the data loader
    import sys
    import os
    
    # Add parent directories to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perplexity_dir = os.path.dirname(script_dir)
    sys.path.insert(0, perplexity_dir)
    sys.path.insert(0, "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna")
    
    from src.config import TEST_DATA_PATH, MODEL_PATH
    from source.languagemodel import LanguageModel
    
    print(f"Loading model from: {MODEL_PATH}")
    model = LanguageModel(MODEL_PATH)
    
    print(f"\nLoading test data (first 10 samples): {TEST_DATA_PATH}")
    sequences = load_tokenized_sequences(TEST_DATA_PATH, model.tokenizer, max_samples=10)
    
    stats = get_sequence_stats(sequences)
    print_sequence_stats(stats)
    
    print(f"\nFirst sequence length: {len(sequences[0])}")
    print(f"First 20 tokens: {sequences[0][:20].tolist()}")
