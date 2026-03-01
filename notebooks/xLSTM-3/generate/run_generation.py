"""
generate/run_generation.py
==========================
CLI entry point: generates all 500 conditions and writes generation_log.csv.

Usage:
    python notebooks/xLSTM-3/generate/run_generation.py

Resumable: if a MIDI file already exists for a condition, that piece is skipped.
Every piece is logged to generation_log.csv incrementally (safe for interruptions).
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import argparse
import importlib.util

# ---------------------------------------------------------------------------
# Make sibling packages importable regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../xLSTM-3/generate/
_XLSTM3_DIR = _SCRIPT_DIR.parent                       # .../xLSTM-3/
_REPO_ROOT = _XLSTM3_DIR.parent.parent                 # .../fyp-musicgen/

sys.path.insert(0, str(_XLSTM3_DIR))
sys.path.insert(0, str(_XLSTM3_DIR / "generate"))

def load_config(config_path: str | None = None):
    """Load config from a file path or default to config.py."""
    if config_path is None:
        import config as cfg
        return cfg
    
    # Absolute path for clarity
    abs_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("config", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {abs_path}")
    
    cfg = importlib.util.module_from_spec(spec)
    # Inject into sys.modules to satisfy internal imports in generator/converter
    sys.modules["config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

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
# Log row schema — one row per generated piece
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "strategy", "target_tokens", "piece_id", "filename", "actual_tokens",
    "target_reached", "num_bars", "num_notes", "num_instruments",
    "grammar_error_rate", "incomplete_triplets", "orphan_tokens",
    "tokens_per_bar_mean", "tokens_per_bar_std", "ends_with_bar",
    "repair_needed", "success", "generation_time_s", "tokens_per_second",
    "num_chunks", "seed",
]


def _make_filename(strategy: str, target_tokens: int, piece_id: int) -> str:
    """Canonical file naming."""
    return f"{strategy}_{target_tokens:05d}tok_{piece_id:03d}.mid"


def _count_error_types(errors) -> tuple[int, int]:
    """Return (incomplete_triplets, orphan_tokens) counts."""
    incomplete = sum(1 for e in errors if "not followed" in e.message)
    orphans = sum(1 for e in errors if "orphan" in e.message)
    return incomplete, orphans


def run(config_path: str | None = None) -> None:
    cfg = load_config(config_path)

    # Add external repos to path before importing them
    if cfg.HELIBRUNNA_PATH not in sys.path:
        sys.path.insert(0, cfg.HELIBRUNNA_PATH)
    if cfg.MIDIPROCESSOR_PATH not in sys.path:
        sys.path.insert(0, cfg.MIDIPROCESSOR_PATH)

    from generator import MusicGenerator          # noqa: E402
    from converter import MIDIConverter           # noqa: E402
    from token_analysis import analyse_tokens     # noqa: E402

    # ------------------------------------------------------------------ dirs
    for d in [cfg.MIDI_DIR, cfg.TOKEN_DIR, cfg.METRICS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ log
    log_path = Path(cfg.LOG_CSV)
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")  # noqa: WPS515
    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        log_file.flush()

    # ------------------------------------------------------------------ model
    logger.info("Loading model …")
    gen = MusicGenerator(
        model_path=cfg.MODEL_PATH,
        context_length=cfg.INFERENCE_CONTEXT_LENGTH,
        device="auto",
        helibrunna_path=cfg.HELIBRUNNA_PATH,
    )
    conv = MIDIConverter(midiprocessor_path=cfg.MIDIPROCESSOR_PATH)

    # ------------------------------------------------------------------ loop
    total = len(cfg.STRATEGIES) * len(cfg.TARGET_TOKENS) * cfg.PIECES_PER_COND
    done = 0

    for strategy in cfg.STRATEGIES:
        for target_idx, target_tokens in enumerate(cfg.TARGET_TOKENS):
            for piece_id in range(1, cfg.PIECES_PER_COND + 1):
                done += 1
                filename = _make_filename(strategy, target_tokens, piece_id)
                midi_path = os.path.join(cfg.MIDI_DIR, filename)
                tok_path  = os.path.join(cfg.TOKEN_DIR, filename.replace(".mid", ".txt"))

                logger.info("[%d/%d] %s", done, total, filename)

                # ------ Bug B fix: unique seed per (piece_id, target_tokens) condition
                # Old: seed = cfg.SEED + piece_id  ← same seed for all target lengths!
                # New: varies by both piece_id and target_tokens index so every condition
                #      gets a different random trajectory.
                seed = cfg.SEED + piece_id * 1000 + target_idx

                # ------ Skip if already done (resumable)
                if os.path.exists(midi_path):
                    logger.info("  Already exists — skipping.")
                    # Bug C fix: if token file exists, reconstruct and log the row
                    # so the CSV is complete even after a resumed run.
                    if os.path.exists(tok_path):
                        try:
                            saved_tokens = Path(tok_path).read_text(encoding="utf-8").split()
                            analysis = analyse_tokens(saved_tokens)
                            incomplete, orphans = _count_error_types(analysis.grammar_errors)
                            writer.writerow({
                                "strategy":           strategy,
                                "target_tokens":      target_tokens,
                                "piece_id":           piece_id,
                                "filename":           filename,
                                "actual_tokens":      len(saved_tokens),
                                "target_reached":     len(saved_tokens) >= target_tokens,
                                "num_bars":           analysis.num_bars,
                                "num_notes":          analysis.num_notes,
                                "num_instruments":    analysis.num_instruments,
                                "grammar_error_rate": round(analysis.grammar_error_rate, 6),
                                "incomplete_triplets":incomplete,
                                "orphan_tokens":      orphans,
                                "tokens_per_bar_mean":round(analysis.tokens_per_bar_mean, 2),
                                "tokens_per_bar_std": round(analysis.tokens_per_bar_std, 2),
                                "ends_with_bar":      analysis.ends_with_bar,
                                "repair_needed":      analysis.repair_needed,
                                "success":            True,
                                "generation_time_s":  "",
                                "tokens_per_second":  "",
                                "num_chunks":         "",
                                "seed":               seed,
                            })
                            log_file.flush()
                            logger.info("  Resume: wrote log row from saved token file.")
                        except Exception as exc:
                            logger.warning("  Resume: could not reconstruct log row: %s", exc)
                    continue

                # ------ Generate
                t0 = time.time()
                try:
                    if strategy == "single_shot":
                        result = gen.single_shot(
                            target_tokens=target_tokens,
                            prompt=cfg.PROMPT,
                            temperature=cfg.TEMPERATURE,
                            seed=seed,
                        )
                    elif strategy == "chunked":
                        result = gen.chunked(
                            target_tokens=target_tokens,
                            prompt=cfg.PROMPT,
                            temperature=cfg.TEMPERATURE,
                            context_tokens=cfg.CONTEXT_TOKENS,
                            context_buffer=cfg.CONTEXT_BUFFER,
                            new_tokens=cfg.NEW_TOKENS,
                            max_retries=cfg.MAX_RETRIES,
                            seed=seed,
                        )
                    else:
                        raise ValueError(f"Unknown strategy: {strategy!r}")
                except Exception as exc:  # noqa: BLE001
                    logger.error("  Generation failed: %s", exc)
                    # Write a failure row so missing pieces are visible in the log
                    writer.writerow({f: "" for f in CSV_FIELDS} | {
                        "strategy": strategy,
                        "target_tokens": target_tokens,
                        "piece_id": piece_id,
                        "filename": filename,
                        "success": False,
                        "generation_time_s": round(time.time() - t0, 2),
                        "seed": seed,
                    })
                    log_file.flush()
                    continue

                tokens: List[str] = result["tokens"]

                # ------ Analyse grammar (on raw tokens, before MIDI conversion)
                analysis = analyse_tokens(tokens)
                incomplete, orphans = _count_error_types(analysis.grammar_errors)

                # ------ Save raw token file
                Path(tok_path).write_text(" ".join(tokens), encoding="utf-8")

                # ------ Convert to MIDI
                success = conv.tokens_to_midi(tokens, midi_path, use_clean_fallback=True)

                logger.info(
                    "  %s | tokens=%d | bars=%d | errors=%.4f | success=%s",
                    strategy, result["actual_tokens"], analysis.num_bars,
                    analysis.grammar_error_rate, success,
                )

                # ------ Write log row
                writer.writerow({
                    "strategy":             strategy,
                    "target_tokens":        target_tokens,
                    "piece_id":             piece_id,
                    "filename":             filename,
                    "actual_tokens":        result["actual_tokens"],
                    "target_reached":       result["target_reached"],
                    "num_bars":             analysis.num_bars,
                    "num_notes":            analysis.num_notes,
                    "num_instruments":      analysis.num_instruments,
                    "grammar_error_rate":   round(analysis.grammar_error_rate, 6),
                    "incomplete_triplets":  incomplete,
                    "orphan_tokens":        orphans,
                    "tokens_per_bar_mean":  round(analysis.tokens_per_bar_mean, 2),
                    "tokens_per_bar_std":   round(analysis.tokens_per_bar_std, 2),
                    "ends_with_bar":        analysis.ends_with_bar,
                    "repair_needed":        analysis.repair_needed,
                    "success":              success,
                    "generation_time_s":    round(result["generation_time_s"], 2),
                    "tokens_per_second":    round(result["tokens_per_second"], 2),
                    "num_chunks":           result["num_chunks"],
                    "seed":                 seed,
                })
                log_file.flush()

    log_file.close()
    logger.info("Done. Log written to: %s", cfg.LOG_CSV)
    logger.info("MIDI files in: %s", cfg.MIDI_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xLSTM-3 Music Generation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.py file (optional)",
    )
    args = parser.parse_args()

    run(config_path=args.config)
