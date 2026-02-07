"""
Utility Functions for Perplexity Evaluation

Logging, file I/O, and results formatting.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional file to write logs to
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("perplexity_eval")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_results(
    results: Dict[str, Any],
    output_path: str,
    model_path: str = None,
    data_file: str = None
) -> str:
    """
    Save perplexity results to JSON file.
    
    Args:
        results: Results dictionary from compute_perplexity()
        output_path: Path to save JSON file
        model_path: Model path for metadata
        data_file: Data file path for metadata
        
    Returns:
        Path to saved file
    """
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "data_file": data_file,
        **results
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove per-sequence results to keep file smaller (optional)
    if "sequence_results" in output:
        output["sequence_results_count"] = len(output["sequence_results"])
        # Keep only summary, not per-sequence details for main results
        del output["sequence_results"]
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output_path


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_ppl_table(results: Dict[int, Dict[str, Any]]) -> str:
    """
    Format multi-context results as a markdown table.
    
    Args:
        results: Dictionary mapping context_length -> results
        
    Returns:
        Markdown table string
    """
    lines = [
        "| Context Length | Perplexity | Avg NLL | Total Tokens | Sequences |",
        "|----------------|------------|---------|--------------|-----------|"
    ]
    
    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        ppl = r.get('perplexity', float('nan'))
        nll = r.get('avg_nll', float('nan'))
        tokens = r.get('total_tokens', 0)
        seqs = r.get('num_evaluated', r.get('num_sequences', 0))
        
        if isinstance(ppl, float) and ppl != float('inf'):
            ppl_str = f"{ppl:.4f}"
        else:
            ppl_str = str(ppl)
            
        if isinstance(nll, float) and nll != float('inf'):
            nll_str = f"{nll:.4f}"
        else:
            nll_str = str(nll)
        
        lines.append(f"| {ctx_len:>14,} | {ppl_str:>10} | {nll_str:>7} | {tokens:>12,} | {seqs:>9} |")
    
    return "\n".join(lines)


def print_banner():
    """Print evaluation banner."""
    print("\n" + "=" * 70)
    print("   xLSTM Music Generation - Perplexity Evaluation")
    print("=" * 70)
