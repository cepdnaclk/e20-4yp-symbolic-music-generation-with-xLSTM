"""
xLSTM-3 Generation Pipeline — Configuration
============================================
This is the ONLY file you need to edit to change model or run parameters.
"""

import os

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so scripts work from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Path to the Helibrunna source (for import)
HELIBRUNNA_PATH = os.path.join(_REPO_ROOT, "repos", "helibrunna")

# Path to the MidiProcessor source (for import)
MIDIPROCESSOR_PATH = os.path.join(_REPO_ROOT, "repos", "MidiProcessor")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Pass the RUN directory — Helibrunna auto-discovers the checkpoint ending in
# "-last". Currently checkpoint-66000-last is the best (lowest PPL = 1.64).
MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "repos", "helibrunna", "output",
    "xlstm_lmd_512d_4096ctx_12b", "run_20260207-1908"
)

# Override context_length at inference time — keeps the Helibrunna context
# wall (inputs.shape[1] >= context_length) far beyond any generated sequence.
# The model was trained with 4096; we allow up to 16 384 here.
INFERENCE_CONTEXT_LENGTH = 16_384

# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
RUN_NAME = "xlstm_512d_4096ctx_ck66k"

# ---------------------------------------------------------------------------
# Generation hyperparameters
# ---------------------------------------------------------------------------
PROMPT = "s-9 o-0 t-38"         # Fixed prompt for all runs (fair comparison)
TEMPERATURE = 0.8                # Sampling temperature
TARGET_TOKENS = [1024, 2048, 4096, 8192, 12288]
STRATEGIES = ["single_shot", "chunked"]
PIECES_PER_COND = 50
SEED = 42                        # Base seed; actual seed per piece = SEED + piece_id

# ---------------------------------------------------------------------------
# Chunked generation parameters
# ---------------------------------------------------------------------------
CONTEXT_TOKENS = 1500   # Target context window (actual may vary by ±CONTEXT_BUFFER)
CONTEXT_BUFFER = 300    # Look-back buffer to find b-1 for left-edge alignment
NEW_TOKENS = 400        # Tokens requested per chunk (~2–3 bars at 158 tok/bar)
MAX_RETRIES = 3         # Max retries per chunk before giving up

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(_HERE, "results", RUN_NAME)
MIDI_DIR    = os.path.join(RESULTS_DIR, "midi")
TOKEN_DIR   = os.path.join(RESULTS_DIR, "tokens")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LOG_CSV     = os.path.join(RESULTS_DIR, "generation_log.csv")
