"""
Configuration for Perplexity Evaluation

Contains all paths and constants used across the evaluation scripts.
"""

import os

# =============================================================================
# Base Paths
# =============================================================================

PROJECT_ROOT = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen"

# Helibrunna paths
HELIBRUNNA_PATH = os.path.join(PROJECT_ROOT, "repos/helibrunna")

# Trained model path - use run directory (Helibrunna finds latest checkpoint automatically)
MODEL_OUTPUT_DIR = os.path.join(
    HELIBRUNNA_PATH, 
    "output/xlstm_lmd_512d_2048ctx_12b/run_20260126-0516"
)

# Model path for loading (run directory, not checkpoint subdirectory)
MODEL_PATH = MODEL_OUTPUT_DIR

# CUDA environment variables for kernel compilation (from working generation code)
CUDA_ENV = {
    "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",
    "MAX_JOBS": "4"
}

# =============================================================================
# Data Paths
# =============================================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data/lmd_preprocessed/splits")

TEST_DATA_PATH = os.path.join(DATA_DIR, "test.txt")
VALID_DATA_PATH = os.path.join(DATA_DIR, "valid.txt")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.txt")

# =============================================================================
# Evaluation Configuration
# =============================================================================

# Context lengths for perplexity testing (following Museformer methodology)
CONTEXT_LENGTHS = [1024, 5120, 10240, 16384]

# Training context length (for reference)
TRAINING_CONTEXT_LENGTH = 2048

# =============================================================================
# Output Paths
# =============================================================================

PERPLEXITY_DIR = os.path.join(PROJECT_ROOT, "xlstm-evaluation/perplexity")
RESULTS_DIR = os.path.join(PERPLEXITY_DIR, "results")

# =============================================================================
# Model Configuration (from training)
# =============================================================================

MODEL_CONFIG = {
    "embedding_dim": 512,
    "num_blocks": 12,
    "context_length": 2048,
    "slstm_at": [3, 6, 9],
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_results_path(context_length: int = None, suffix: str = "") -> str:
    """Get path for saving results for a specific context length."""
    if context_length:
        filename = f"ppl_context_{context_length}{suffix}.json"
    else:
        filename = f"ppl_full_sequence{suffix}.json"
    return os.path.join(RESULTS_DIR, filename)


def verify_paths():
    """Verify that all required paths exist."""
    paths_to_check = [
        ("Helibrunna", HELIBRUNNA_PATH),
        ("Model output directory", MODEL_OUTPUT_DIR),
        ("Test data", TEST_DATA_PATH),
        ("Validation data", VALID_DATA_PATH),
    ]
    
    all_exist = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("Verifying paths...")
    if verify_paths():
        print("\nAll paths verified successfully!")
    else:
        print("\nSome paths are missing!")
