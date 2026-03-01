"""
single-shot-generation/run_generation.py
=========================================
CLI entry point: generates all pieces using xLSTMGenerator (O(N) recurrent inference)
and writes generation_metrics.csv with comprehensive metrics.

Usage:
    # Use default config (configs/default.py)
    python run_generation.py

    # Use quick test config (5 pieces per length)
    python run_generation.py --config configs/quick_test.py

    # Use custom config
    python run_generation.py --config configs/my_experiment.py

Features:
- Resumable: skips pieces that already have MIDI files
- Incremental CSV logging (safe for interruptions)
- Uses xLSTMGenerator for fast O(N) linear-time generation
- Single-shot only (no chunking needed with recurrent inference)
- Multi-config support: each config auto-generates its own results folder
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import sys
import time
import torch
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Setup paths for imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # single-shot-generation/
_XLSTM4_DIR = _SCRIPT_DIR.parent                       # xLSTM-4-recurrent-state/
_NOTEBOOKS_DIR = _XLSTM4_DIR.parent                    # notebooks/
_REPO_ROOT = _NOTEBOOKS_DIR.parent                     # fyp-musicgen/

# Add to path for imports
sys.path.insert(0, str(_SCRIPT_DIR))      # For token_analysis, converter
sys.path.insert(0, str(_XLSTM4_DIR))      # For inference

# ---------------------------------------------------------------------------
# Import modules (config loaded dynamically based on --config argument)
# ---------------------------------------------------------------------------
from token_analysis import analyse_tokens
from converter import MIDIConverter
from inference import xLSTMGenerator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_generation")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None):
    """
    Load config from a file path or default to configs/default.py.

    Args:
        config_path: Path to config file (e.g., "configs/quick_test.py")
                    If None, uses configs/default.py

    Returns:
        Loaded config module with all settings
    """
    if config_path is None:
        # Default config
        config_path = str(_SCRIPT_DIR / "configs" / "default.py")

    # Convert to absolute path
    abs_path = os.path.abspath(config_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    # Load config as module
    spec = importlib.util.spec_from_file_location("config", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {abs_path}")

    cfg = importlib.util.module_from_spec(spec)
    sys.modules["config"] = cfg  # Inject into sys.modules
    spec.loader.exec_module(cfg)

    logger.info("Loaded config from: %s", abs_path)
    logger.info("  RUN_NAME: %s", cfg.RUN_NAME)
    logger.info("  TARGET_TOKENS: %s", cfg.TARGET_TOKENS)
    logger.info("  PIECES_PER_COND: %s", cfg.PIECES_PER_COND)
    logger.info("  TEMPERATURE: %s", cfg.TEMPERATURE)

    return cfg


# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "strategy", "target_tokens", "piece_id", "filename", "actual_tokens",
    "target_reached", "num_bars", "num_notes", "num_instruments",
    "grammar_error_rate", "incomplete_triplets", "orphan_tokens",
    "tokens_per_bar_mean", "tokens_per_bar_std", "ends_with_bar",
    "repair_needed", "success", "generation_time_s", "tokens_per_second",
    "num_chunks", "seed",
]


def _make_filename(target_tokens: int, piece_id: int) -> str:
    """Canonical file naming: single_shot_{tokens:05d}tok_{piece_id:03d}.mid"""
    return f"single_shot_{target_tokens:05d}tok_{piece_id:03d}.mid"


def _count_error_types(errors) -> tuple[int, int]:
    """Return (incomplete_triplets, orphan_tokens) counts from grammar errors."""
    incomplete = sum(1 for e in errors if "not followed" in e.message)
    orphans = sum(1 for e in errors if "orphan" in e.message)
    return incomplete, orphans


def run(config_path: str | None = None) -> None:
    """
    Main generation loop.

    Args:
        config_path: Path to config file (e.g., "configs/quick_test.py")
                    If None, uses configs/default.py
    """
    # ------------------------------------------------------------------ Load config
    cfg = load_config(config_path)

    # ------------------------------------------------------------------ Setup
    # Add MidiProcessor to path
    if cfg.MIDIPROCESSOR_PATH not in sys.path:
        sys.path.insert(0, cfg.MIDIPROCESSOR_PATH)

    # Create output directories
    for d in [cfg.MIDI_DIR, cfg.TOKEN_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Open CSV log file (append mode for resumability)
    log_path = Path(cfg.LOG_CSV)
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        log_file.flush()

    # ------------------------------------------------------------------ Load model
    logger.info("Loading xLSTMGenerator (O(N) recurrent inference)...")
    logger.info("  Model: %s", cfg.MODEL_PATH)
    logger.info("  Checkpoint: %s", cfg.CHECKPOINT_NAME)

    generator = xLSTMGenerator(
        model_path_or_repo=cfg.MODEL_PATH,
        checkpoint_name=cfg.CHECKPOINT_NAME,
        config_overrides={"context_length": cfg.INFERENCE_CONTEXT_LENGTH},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    converter = MIDIConverter(midiprocessor_path=cfg.MIDIPROCESSOR_PATH)

    logger.info("Model loaded successfully.")
    logger.info("Generation settings:")
    logger.info("  Temperature: %s", cfg.TEMPERATURE)
    logger.info("  Target tokens: %s", cfg.TARGET_TOKENS)
    logger.info("  Pieces per condition: %s", cfg.PIECES_PER_COND)
    logger.info("  Prompt: '%s'", cfg.PROMPT)

    # ------------------------------------------------------------------ Generation loop
    total_pieces = len(cfg.TARGET_TOKENS) * cfg.PIECES_PER_COND
    done = 0

    for target_idx, target_tokens in enumerate(cfg.TARGET_TOKENS):
        for piece_id in range(1, cfg.PIECES_PER_COND + 1):
            done += 1
            filename = _make_filename(target_tokens, piece_id)
            midi_path = os.path.join(cfg.MIDI_DIR, filename)
            tok_path = os.path.join(cfg.TOKEN_DIR, filename.replace(".mid", ".txt"))

            logger.info("[%d/%d] %s", done, total_pieces, filename)

            # Unique seed per piece (varies by piece_id and target_tokens)
            seed = cfg.SEED + piece_id * 1000 + target_idx

            # ------ Skip if already done (resumable)
            if os.path.exists(midi_path):
                logger.info("  Already exists — skipping.")
                # If token file exists, reconstruct log row for CSV completeness
                if os.path.exists(tok_path):
                    try:
                        saved_tokens = Path(tok_path).read_text(encoding="utf-8").split()
                        analysis = analyse_tokens(saved_tokens)
                        incomplete, orphans = _count_error_types(analysis.grammar_errors)
                        writer.writerow({
                            "strategy": "single_shot",
                            "target_tokens": target_tokens,
                            "piece_id": piece_id,
                            "filename": filename,
                            "actual_tokens": len(saved_tokens),
                            "target_reached": len(saved_tokens) >= target_tokens,
                            "num_bars": analysis.num_bars,
                            "num_notes": analysis.num_notes,
                            "num_instruments": analysis.num_instruments,
                            "grammar_error_rate": round(analysis.grammar_error_rate, 6),
                            "incomplete_triplets": incomplete,
                            "orphan_tokens": orphans,
                            "tokens_per_bar_mean": round(analysis.tokens_per_bar_mean, 2),
                            "tokens_per_bar_std": round(analysis.tokens_per_bar_std, 2),
                            "ends_with_bar": analysis.ends_with_bar,
                            "repair_needed": analysis.repair_needed,
                            "success": True,
                            "generation_time_s": "",
                            "tokens_per_second": "",
                            "num_chunks": 1,  # Always 1 for single-shot
                            "seed": seed,
                        })
                        log_file.flush()
                        logger.info("  Resume: wrote log row from saved token file.")
                    except Exception as exc:
                        logger.warning("  Resume: could not reconstruct log row: %s", exc)
                continue

            # ------ Generate
            t0 = time.time()
            try:
                # Set random seed for reproducibility
                torch.manual_seed(seed)

                # Generate using xLSTMGenerator (O(N) recurrent inference)
                result = generator.generate(
                    prompt=cfg.PROMPT,
                    temperature=cfg.TEMPERATURE,
                    max_length=target_tokens,
                    return_structured_output=True
                )

                generation_time_s = result["elapsed_time"]
                tokens_per_second = result["tokens_per_second"]
                output_text = result["output"]

                # Parse output text to token list
                tokens: List[str] = output_text.split()
                actual_tokens = len(tokens)
                target_reached = actual_tokens >= target_tokens

            except Exception as exc:
                logger.error("  Generation failed: %s", exc)
                # Write failure row
                writer.writerow({f: "" for f in CSV_FIELDS} | {
                    "strategy": "single_shot",
                    "target_tokens": target_tokens,
                    "piece_id": piece_id,
                    "filename": filename,
                    "success": False,
                    "generation_time_s": round(time.time() - t0, 2),
                    "seed": seed,
                    "num_chunks": 1,
                })
                log_file.flush()
                continue

            # ------ Analyze grammar (on raw tokens, before MIDI conversion)
            analysis = analyse_tokens(tokens)
            incomplete, orphans = _count_error_types(analysis.grammar_errors)

            # ------ Save raw token file
            Path(tok_path).write_text(" ".join(tokens), encoding="utf-8")

            # ------ Convert to MIDI
            success = converter.tokens_to_midi(tokens, midi_path, use_clean_fallback=True)

            logger.info(
                "  tokens=%d/%d | bars=%d | grammar_errors=%.4f | success=%s | time=%.2fs | speed=%.1f tok/s",
                actual_tokens, target_tokens, analysis.num_bars,
                analysis.grammar_error_rate, success, generation_time_s, tokens_per_second
            )

            # ------ Write log row
            writer.writerow({
                "strategy": "single_shot",
                "target_tokens": target_tokens,
                "piece_id": piece_id,
                "filename": filename,
                "actual_tokens": actual_tokens,
                "target_reached": target_reached,
                "num_bars": analysis.num_bars,
                "num_notes": analysis.num_notes,
                "num_instruments": analysis.num_instruments,
                "grammar_error_rate": round(analysis.grammar_error_rate, 6),
                "incomplete_triplets": incomplete,
                "orphan_tokens": orphans,
                "tokens_per_bar_mean": round(analysis.tokens_per_bar_mean, 2),
                "tokens_per_bar_std": round(analysis.tokens_per_bar_std, 2),
                "ends_with_bar": analysis.ends_with_bar,
                "repair_needed": analysis.repair_needed,
                "success": success,
                "generation_time_s": round(generation_time_s, 2),
                "tokens_per_second": round(tokens_per_second, 2),
                "num_chunks": 1,  # Always 1 for single-shot
                "seed": seed,
            })
            log_file.flush()

    # ------------------------------------------------------------------ Cleanup
    log_file.close()
    logger.info("=" * 70)
    logger.info("Generation complete!")
    logger.info("  Total pieces: %d", total_pieces)
    logger.info("  Metrics CSV: %s", cfg.LOG_CSV)
    logger.info("  MIDI files: %s", cfg.MIDI_DIR)
    logger.info("  Token files: %s", cfg.TOKEN_DIR)
    logger.info("=" * 70)

    # ------------------------------------------------------------------ Generate plots
    logger.info("")
    logger.info("Generating plots...")
    try:
        from plot_results import generate_plots
        plots_dir = generate_plots(csv_path=cfg.LOG_CSV, output_dir=cfg.RESULTS_DIR)
        logger.info("=" * 70)
        logger.info("Plots saved to: %s", plots_dir)
        logger.info("=" * 70)
    except ImportError as e:
        logger.warning("Could not import plot_results.py - skipping plots")
        logger.warning("  Install matplotlib and seaborn to enable plotting")
    except Exception as e:
        logger.warning("Failed to generate plots: %s", e)
        logger.warning("  You can manually generate plots with:")
        logger.warning("  python plot_results.py --csv %s", cfg.LOG_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="xLSTM-4 Single-Shot Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config (configs/default.py - 50 pieces per length)
  python run_generation.py

  # Use quick test config (5 pieces per length)
  python run_generation.py --config configs/quick_test.py

  # Use custom config
  python run_generation.py --config configs/my_experiment.py
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/default.py)"
    )
    args = parser.parse_args()

    run(config_path=args.config)
