#!/usr/bin/env python3
"""
xLSTM Music Generation - Perplexity Evaluation

Main evaluation script that computes perplexity at multiple context lengths.
Following Museformer's methodology for direct comparison.

Usage:
    python evaluate_perplexity.py --help
    python evaluate_perplexity.py --max-samples 10  # Quick test
    python evaluate_perplexity.py                    # Full evaluation
"""

import os
import sys
import argparse
from datetime import datetime

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../../repos/helibrunna"))

# Set CUDA environment variables before importing torch
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0;8.6;8.9')
os.environ.setdefault('MAX_JOBS', '4')

from src.config import MODEL_PATH, TEST_DATA_PATH, VALID_DATA_PATH, CONTEXT_LENGTHS, RESULTS_DIR
from src.data_loader import load_tokenized_sequences, get_sequence_stats, print_sequence_stats
from src.perplexity_calculator import compute_perplexity, compute_perplexity_multi_context, print_results_summary
from src.utils import save_results, format_ppl_table, print_banner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute perplexity for xLSTM music generation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (10 samples, one context):
    python evaluate_perplexity.py --max-samples 10 --context 1024

  Full evaluation at all context lengths:
    python evaluate_perplexity.py --all-contexts

  Single context length on test set:
    python evaluate_perplexity.py --context 5120

  Evaluate a specific checkpoint:
    python evaluate_perplexity.py --checkpoint checkpoint-80000 --context 1024
        """
    )
    
    parser.add_argument(
        "--model-path", 
        default=MODEL_PATH,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data-file", 
        default=TEST_DATA_PATH,
        help="Path to data file (test.txt or valid.txt)"
    )
    parser.add_argument(
        "--context", 
        type=int, 
        default=None,
        help="Single context length to evaluate (default: 1024)"
    )
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help="Evaluate at all context lengths: 1024, 5120, 10240, 16384"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="Limit number of sequences (for testing)"
    )
    parser.add_argument(
        "--output-dir", 
        default=RESULTS_DIR,
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint folder name (e.g., 'checkpoint-80000'). Default: latest (-last)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.quiet:
        print_banner()
    
    # Determine context lengths FIRST (needed for model loading)
    if args.all_contexts:
        context_lengths = CONTEXT_LENGTHS  # [1024, 5120, 10240, 16384]
    elif args.context:
        context_lengths = [args.context]
    else:
        context_lengths = [1024]  # Default: just 1024
    
    max_context = max(context_lengths)
    
    # Load model with context override if needed (training context was 2048)
    checkpoint_info = f" (checkpoint: {args.checkpoint})" if args.checkpoint else ""
    print(f"\n📦 Loading model from: {args.model_path}{checkpoint_info}")
    
    from source.languagemodel import LanguageModel
    
    # Override context length if evaluating beyond training context (2048)
    if max_context > 2048:
        print(f"   ⚠ Overriding context length to {max_context} (training was 2048)")
        config_overrides = {"context_length": max_context}
    else:
        config_overrides = {}
    
    model = LanguageModel(
        args.model_path, 
        device="cuda", 
        checkpoint_name=args.checkpoint,
        config_overrides=config_overrides
    )
    print(f"   ✓ Model loaded (vocab size: {len(model.tokenizer)})")
    
    # Load data
    print(f"\n📄 Loading data from: {os.path.basename(args.data_file)}")
    sequences = load_tokenized_sequences(
        args.data_file, 
        model.tokenizer, 
        max_samples=args.max_samples,
        show_progress=not args.quiet
    )
    
    if not args.quiet:
        stats = get_sequence_stats(sequences)
        print_sequence_stats(stats)
    else:
        print(f"   ✓ Loaded {len(sequences)} sequences")
    
    # Compute perplexity
    print(f"\n🔢 Computing perplexity at context lengths: {context_lengths}")
    
    if len(context_lengths) == 1:
        results = compute_perplexity(
            model, 
            sequences, 
            max_context=context_lengths[0],
            show_progress=not args.quiet
        )
        if not args.quiet:
            print_results_summary(results)
        else:
            print(f"\n   PPL@{context_lengths[0]}: {results['perplexity']:.4f}")
    else:
        results = compute_perplexity_multi_context(
            model,
            sequences,
            context_lengths=context_lengths,
            show_progress=not args.quiet
        )
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(format_ppl_table(results))
    
    # Save results
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(context_lengths) == 1:
            filename = f"ppl_ctx{context_lengths[0]}_{timestamp}.json"
            save_path = os.path.join(args.output_dir, filename)
            save_results(results, save_path, args.model_path, args.data_file)
        else:
            # Save each context length separately
            for ctx_len, ctx_results in results.items():
                filename = f"ppl_ctx{ctx_len}_{timestamp}.json"
                save_path = os.path.join(args.output_dir, filename)
                save_results(ctx_results, save_path, args.model_path, args.data_file)
        
        print(f"\n💾 Results saved to: {args.output_dir}/")
    
    print("\n✅ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    main()
