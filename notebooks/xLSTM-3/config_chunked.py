"""
xLSTM-3 Generation Pipeline — Chunked Strategy Evaluation
==========================================================
20 pieces per condition (1k, 2k, 4k, 8k, 12k)
Total: 100 pieces
"""

import os

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

HELIBRUNNA_PATH = os.path.join(_REPO_ROOT, "repos", "helibrunna")
MIDIPROCESSOR_PATH = os.path.join(_REPO_ROOT, "repos", "MidiProcessor")

# Model (Checkpoint 66k)
MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "repos", "helibrunna", "output",
    "xlstm_lmd_512d_4096ctx_12b", "run_20260207-1908"
)

INFERENCE_CONTEXT_LENGTH = 16_384

# Run identity
RUN_NAME = "xlstm_chunked_eval"

# Generation hyperparameters
PROMPT = "s-9 o-0 t-38"
TEMPERATURE = 0.8
TARGET_TOKENS = [1024, 2048, 4096, 8192, 12288]
STRATEGIES = ["chunked"]
PIECES_PER_COND = 20
SEED = 42

# Chunked-specific tuning
CONTEXT_TOKENS = 1500
CONTEXT_BUFFER = 300
NEW_TOKENS = 400
MAX_RETRIES = 5  # Higher retries for the production run

# Output paths
RESULTS_DIR = os.path.join(_HERE, "results", RUN_NAME)
MIDI_DIR    = os.path.join(RESULTS_DIR, "midi")
TOKEN_DIR   = os.path.join(RESULTS_DIR, "tokens")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LOG_CSV     = os.path.join(RESULTS_DIR, "generation_log.csv")
