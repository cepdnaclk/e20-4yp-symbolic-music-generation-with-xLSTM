"""
xLSTM-4 Quick Test Configuration
=================================
Quick test configuration: 5 pieces per target length (25 total pieces).
Use this for rapid validation before running full experiments.

Estimated time: ~15 minutes
- 1024 tokens: 5 pieces × 7s ≈ 35 seconds
- 2048 tokens: 5 pieces × 13s ≈ 65 seconds
- 4096 tokens: 5 pieces × 25s ≈ 125 seconds
- 8192 tokens: 5 pieces × 51s ≈ 255 seconds
- 12288 tokens: 5 pieces × 77s ≈ 385 seconds
Total: ~14 minutes
"""

import os

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so scripts work from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))  # configs/
_SINGLE_SHOT_DIR = os.path.dirname(_HERE)  # single-shot-generation/
_XLSTM4_DIR = os.path.dirname(_SINGLE_SHOT_DIR)  # xLSTM-4-recurrent-state/
_NOTEBOOKS_DIR = os.path.dirname(_XLSTM4_DIR)  # notebooks/
_REPO_ROOT = os.path.dirname(_NOTEBOOKS_DIR)  # fyp-musicgen/

# Path to the MidiProcessor source (for MIDI conversion)
MIDIPROCESSOR_PATH = os.path.join(_REPO_ROOT, "repos", "MidiProcessor")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Model run directory — xLSTMGenerator auto-discovers checkpoint-*-last
MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "repos", "helibrunna", "output",
    "xlstm_lmd_512d_4096ctx_12b", "run_20260207-1908"
)

# Checkpoint name (best PPL checkpoint)
CHECKPOINT_NAME = "checkpoint-66000-last"

# Override context_length at inference time
# Training context: 4096, inference context: 16384 (allows extrapolation testing)
INFERENCE_CONTEXT_LENGTH = 16_384

# ---------------------------------------------------------------------------
# Generation Hyperparameters (Quick Test Settings)
# ---------------------------------------------------------------------------
# Fixed prompt for all runs (ensures fair comparison)
PROMPT = "s-9 o-0 t-38"

# Sampling temperature (higher = more creative, lower = more conservative)
TEMPERATURE = 0.8

# Target token lengths to generate
# Training context = 4096, so these test within-context and extrapolation
TARGET_TOKENS = [1024, 2048, 4096, 8192, 12288]

# Number of pieces to generate per target length (REDUCED FOR QUICK TEST)
PIECES_PER_COND = 5

# Base random seed (actual seed per piece = SEED + piece_id * 1000 + target_idx)
SEED = 42

# ---------------------------------------------------------------------------
# Output Paths (automatically inferred from config filename)
# ---------------------------------------------------------------------------
# Auto-infer run name from this config file's name (e.g., "quick_test" from "quick_test.py")
_CONFIG_FILENAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = _CONFIG_FILENAME if _CONFIG_FILENAME else "xlstm_recurrent_single_shot"

RESULTS_DIR = os.path.join(_SINGLE_SHOT_DIR, "results", RUN_NAME)
MIDI_DIR    = os.path.join(RESULTS_DIR, "midi")
TOKEN_DIR   = os.path.join(RESULTS_DIR, "tokens")
LOG_CSV     = os.path.join(RESULTS_DIR, "generation_metrics.csv")
