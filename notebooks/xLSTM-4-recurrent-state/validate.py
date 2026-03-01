import os
import sys
import torch
import time

# We need to add the repos folder to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../repos')))
from helibrunna.source.languagemodel import LanguageModel as HelibrunnaLanguageModel
from inference import xLSTMGenerator

def test_identical_generation(model_repo_path, prompt, length=100, seed=42):
    print(f"--- Starting Validation Setup ---")
    print(f"Using model from: {model_repo_path}")
    print(f"Generation length: {length} tokens")
    
    # Initialize both generators
    print("Loading Helibrunna (Slow Parallel Generator)...")
    heli_gen = HelibrunnaLanguageModel(
        model_path_or_repo=model_repo_path,
        checkpoint_name="checkpoint-66000-last",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n--- Running Generation 1: Helibrunna ---")
    torch.manual_seed(seed)
    
    start_heli = time.time()
    result_heli = heli_gen.generate(
        prompt=prompt,
        max_length=length,
        temperature=1.0,
        return_structured_output=True
    )
    heli_time = time.time() - start_heli
    heli_text = result_heli['output']
    
    print(f"Helibrunna finished in {heli_time:.2f} seconds.")
    
    print("Freeing Helibrunna from GPU memory to avoid OOM...")
    del heli_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nLoading xLSTMGenerator (Fast Recurrent Generator)...")
    fast_gen = xLSTMGenerator(
        model_path_or_repo=model_repo_path, 
        checkpoint_name="checkpoint-66000-last",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n--- Running Generation 2: xLSTMGenerator ---")
    # Reset seed to guarantee same exact multinomial samples
    torch.manual_seed(seed)
    
    start_fast = time.time()
    result_fast = fast_gen.generate(
        prompt=prompt,
        max_length=length,
        temperature=1.0,
        return_structured_output=True
    )
    fast_time = time.time() - start_fast
    fast_text = result_fast['output']
    
    print(f"xLSTMGenerator finished in {fast_time:.2f} seconds.")
    
    print("\n--- Results ---")
    print(f"Speedup: {heli_time / fast_time:.2f}x faster")
    
    # Compare
    if heli_text == fast_text:
        print("\n✅ SUCCESS: Both methods produced exactly the same output text!")
        return True
    else:
        print("\n❌ FAILED: The outputs diverged.")
        print("\nHelibrunna Output:")
        print(repr(heli_text[:500]) + "...")
        print("\nxLSTMGenerator Output:")
        print(repr(fast_text[:500]) + "...")
        return False

if __name__ == "__main__":
    # Feel free to change these parameters when running!
    model_path = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/xlstm_lmd_512d_4096ctx_12b/run_20260207-1908"
        
    test_prompt = "s-9 o-0 t-38"
    
    test_identical_generation(model_path, test_prompt, length=100)
