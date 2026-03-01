import os
import sys
import torch
import time
import gc

# We need to add the repos folder to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../repos')))
from helibrunna.source.languagemodel import LanguageModel as HelibrunnaLanguageModel
# Import xLSTMGenerator from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import xLSTMGenerator

def run_benchmark(model_repo_path, prompt, test_lengths=[100, 500, 1000]):
    print("===========================================")
    print("      xLSTM Generation Benchmark Tool      ")
    print("===========================================")
    print(f"Model Checkpoint: {model_repo_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Testing Lengths: {test_lengths}\n")

    results_heli = []
    
    # 1. Benchmark Helibrunna (Slow Parallel)
    print(">>> Loading Helibrunna (Parallel Generator) <<<")
    heli_gen = HelibrunnaLanguageModel(
        model_path_or_repo=model_repo_path,
        checkpoint_name="checkpoint-66000-last",
        config_overrides={"context_length": 16384},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    for length in test_lengths:
        print(f"\n[Helibrunna] Generating {length} tokens...")
        torch.manual_seed(42)
        start_time = time.time()
        
        heli_gen.generate(
            prompt=prompt,
            max_length=length,
            temperature=1.0,
            return_structured_output=False
        )
        
        elapsed = time.time() - start_time
        tps = length / elapsed
        print(f"   -> Time: {elapsed:.2f}s | Speed: {tps:.2f} tokens/sec")
        results_heli.append(elapsed)

    print("\nFreeing Helibrunna from GPU memory...")
    del heli_gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # 2. Benchmark xLSTMGenerator (Fast Recurrent)
    results_fast = []
    
    print("\n>>> Loading xLSTMGenerator (Recurrent Generator) <<<")
    fast_gen = xLSTMGenerator(
        model_path_or_repo=model_repo_path, 
        checkpoint_name="checkpoint-66000-last",
        config_overrides={"context_length": 16384},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

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


    # 3. Final Report
    print("\n===========================================")
    print("             FINAL REPORT                  ")
    print("===========================================")
    print(f"{'Length':>10} | {'Helibrunna (s)':>15} | {'xLSTMGen (s)':>15} | {'Speedup':>10}")
    print("-" * 59)
    for i, length in enumerate(test_lengths):
        heli_time = results_heli[i]
        fast_time = results_fast[i]
        speedup = heli_time / fast_time
        print(f"{length:>10} | {heli_time:>15.2f} | {fast_time:>15.2f} | {speedup:>9.2f}x")
        
    print("===========================================\n")


if __name__ == "__main__":
    # Feel free to change these parameters when running!
    model_path = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/xlstm_lmd_512d_4096ctx_12b/run_20260207-1908"
    test_prompt = "s-9 o-0 t-38"
    
    run_benchmark(model_path, test_prompt, test_lengths=[100, 500, 1000, 2000])
