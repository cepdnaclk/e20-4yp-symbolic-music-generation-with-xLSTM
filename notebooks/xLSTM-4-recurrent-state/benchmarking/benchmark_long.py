import os
import sys
import torch
import time
import gc

# Add parent dir to path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import xLSTMGenerator

def run_long_benchmark(model_repo_path, prompt, test_lengths=[2000, 4000, 8000, 12000]):
    print("===========================================")
    print("  xLSTM Long Sequence Generation Benchmark ")
    print("===========================================")
    print(f"Model Checkpoint: {model_repo_path}")
    print(f"Testing Lengths: {test_lengths}\n")

    print(">>> Loading xLSTMGenerator (Recurrent Generator) <<<")
    fast_gen = xLSTMGenerator(
        model_path_or_repo=model_repo_path, 
        checkpoint_name="checkpoint-66000-last",
        config_overrides={"context_length": 16384},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    results_fast = []

    for length in test_lengths:
        print(f"\n[xLSTMGenerator] Generating {length} tokens...")
        torch.manual_seed(42)
        start_time = time.time()
        
        fast_gen.generate(
            prompt=prompt,
            max_length=length,
            temperature=1.0,
            return_structured_output=False
        )
        
        elapsed = time.time() - start_time
        tps = length / elapsed
        print(f"   -> Time: {elapsed:.2f}s | Speed: {tps:.2f} tokens/sec")
        results_fast.append(elapsed)

    print("\n===========================================")
    print("             FINAL REPORT                  ")
    print("===========================================")
    print(f"{'Length':>10} | {'xLSTMGen (s)':>15} | {'Tokens/sec':>15}")
    print("-" * 46)
    for i, length in enumerate(test_lengths):
        fast_time = results_fast[i]
        tps = length / fast_time
        print(f"{length:>10} | {fast_time:>15.2f} | {tps:>15.2f}")
    print("===========================================\n")

if __name__ == "__main__":
    model_path = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/xlstm_lmd_512d_4096ctx_12b/run_20260207-1908"
    test_prompt = "s-9 o-0 t-38"
    
    # Helibrunna would take ~30+ minutes for 12000 tokens and likely run out of memory. 
    # We only test the fast generator here.
    run_long_benchmark(model_path, test_prompt, test_lengths=[2000, 4000, 8000, 12000])
