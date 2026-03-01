"""
Smoke test for single-shot generation pipeline.
Tests 1 piece at 1024 tokens to verify everything works.

This creates a temporary config and runs the generation pipeline.
"""
import sys
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Create a temporary smoke test config
_SMOKE_CONFIG_PATH = _SCRIPT_DIR / "configs" / "smoke_test_temp.py"

# Write temporary config
_SMOKE_CONFIG_CONTENT = '''"""
Temporary smoke test config - auto-generated, do not edit.
Tests 1 piece at 1024 tokens.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SINGLE_SHOT_DIR = os.path.dirname(_HERE)
_XLSTM4_DIR = os.path.dirname(_SINGLE_SHOT_DIR)
_NOTEBOOKS_DIR = os.path.dirname(_XLSTM4_DIR)
_REPO_ROOT = os.path.dirname(_NOTEBOOKS_DIR)

MIDIPROCESSOR_PATH = os.path.join(_REPO_ROOT, "repos", "MidiProcessor")

MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "repos", "helibrunna", "output",
    "xlstm_lmd_512d_4096ctx_12b", "run_20260207-1908"
)

CHECKPOINT_NAME = "checkpoint-66000-last"
INFERENCE_CONTEXT_LENGTH = 16_384

PROMPT = "s-9 o-0 t-38"
TEMPERATURE = 0.8
TARGET_TOKENS = [1024]  # Only test 1024 tokens
PIECES_PER_COND = 1      # Only 1 piece
SEED = 42

RUN_NAME = "smoke_test"
RESULTS_DIR = os.path.join(_SINGLE_SHOT_DIR, "results", RUN_NAME)
MIDI_DIR = os.path.join(RESULTS_DIR, "midi")
TOKEN_DIR = os.path.join(RESULTS_DIR, "tokens")
LOG_CSV = os.path.join(RESULTS_DIR, "generation_metrics.csv")
'''

# Import and run
from run_generation import run

if __name__ == "__main__":
    # Write temporary config
    _SMOKE_CONFIG_PATH.write_text(_SMOKE_CONFIG_CONTENT)

    print("=" * 70)
    print("SMOKE TEST: Generating 1 piece at 1024 tokens")
    print("=" * 70)

    try:
        run(config_path=str(_SMOKE_CONFIG_PATH))
        print("=" * 70)
        print("Smoke test complete! Check results/smoke_test/")
        print("=" * 70)
    finally:
        # Clean up temporary config
        if _SMOKE_CONFIG_PATH.exists():
            _SMOKE_CONFIG_PATH.unlink()
