"""
Diagnostic script: run single_shot with full traceback output.
Usage: conda run -n xlstm python3 -u notebooks/xLSTM-3/diag_single_shot.py
"""
import sys, traceback, time
sys.path.insert(0, 'notebooks/xLSTM-3')
sys.path.insert(0, 'notebooks/xLSTM-3/generate')
sys.path.insert(0, 'repos/helibrunna')
sys.path.insert(0, 'repos/MidiProcessor')

import config as cfg

print("Loading model...", flush=True)
from generator import MusicGenerator
gen = MusicGenerator(
    cfg.MODEL_PATH,
    context_length=cfg.INFERENCE_CONTEXT_LENGTH,
    device="auto",
    helibrunna_path=cfg.HELIBRUNNA_PATH,
)
print("Model loaded.", flush=True)

# Also try to get partial tokens if calling _raw_generate directly
print("\n--- Test 1: single_shot target=2048, seed=43 ---", flush=True)
try:
    result = gen.single_shot(
        target_tokens=2048,
        seed=43,
        prompt=cfg.PROMPT,
        temperature=cfg.TEMPERATURE,
    )
    print(f"SUCCESS: {result['actual_tokens']} tokens, {result['generation_time_s']:.1f}s", flush=True)
    print("First 20 tokens:", result['tokens'][:20], flush=True)
except Exception as e:
    print(f"\n=== EXCEPTION: {type(e).__name__} ===", flush=True)
    print(f"Message: {e}", flush=True)
    traceback.print_exc()

# Now try raw generate to see what tokenizer returns
print("\n--- Test 2: _raw_generate target=2048, seed=43 ---", flush=True)
import torch
torch.manual_seed(43)
try:
    raw = gen._raw_generate(cfg.PROMPT, max_length=2048, temperature=cfg.TEMPERATURE)
    toks = raw.split()
    print(f"Raw generate SUCCESS: {len(toks)} tokens", flush=True)
    print("Last 30 tokens:", toks[-30:], flush=True)
except Exception as e:
    print(f"\n=== RAW GENERATE EXCEPTION: {type(e).__name__} ===", flush=True)
    print(f"Message: {e}", flush=True)
    traceback.print_exc()

print("\nDone.", flush=True)
