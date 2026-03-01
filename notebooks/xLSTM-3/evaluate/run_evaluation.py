"""
evaluate/run_evaluation.py
==========================
CLI entry point: reads generation_log.csv, produces aggregated metrics and plots.

Usage:
    python notebooks/xLSTM-3/evaluate/run_evaluation.py

Output:
    results/<RUN_NAME>/metrics/summary_by_condition.csv
    results/<RUN_NAME>/metrics/plots/*.png
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import argparse
import importlib.util

# ---------------------------------------------------------------------------
# Make sibling packages importable regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../xLSTM-3/evaluate/
_XLSTM3_DIR = _SCRIPT_DIR.parent                       # .../xLSTM-3/
sys.path.insert(0, str(_XLSTM3_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

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
    # Inject into sys.modules to satisfy internal imports
    sys.modules["config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_evaluation")

def run(config_path: str | None = None) -> None:
    cfg = load_config(config_path)

    from grammar_eval import (  # noqa: E402
        aggregate,
        generate_plots,
        load_log,
        write_summary_csv,
    )

    log_csv = cfg.LOG_CSV
    if not Path(log_csv).exists():
        logger.error("generation_log.csv not found at: %s", log_csv)
        logger.error("Run generate/run_generation.py first.")
        sys.exit(1)

    logger.info("Loading generation log: %s", log_csv)
    rows = load_log(log_csv)
    if not rows:
        logger.error("No valid rows found in log. Aborting.")
        sys.exit(1)

    logger.info("Aggregating %d rows …", len(rows))
    aggregated = aggregate(rows)

    # Summary CSV
    summary_path = str(Path(cfg.METRICS_DIR) / "summary_by_condition.csv")
    write_summary_csv(aggregated, summary_path)

    # Plots
    plots_dir = str(Path(cfg.METRICS_DIR) / "plots")
    generate_plots(aggregated, plots_dir)

    # Print a quick overview to stdout
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Target':>8} {'Success%':>9} {'ErrRate':>9} {'TokReach%':>10}")
    print("-" * 60)
    for key in sorted(aggregated.keys(), key=lambda k: (k[0], k[1])):
        v = aggregated[key]
        print(
            f"{v['strategy']:<15} {v['target_tokens']:>8} "
            f"{v['success_rate'] * 100:>8.1f}% "
            f"{v['grammar_error_rate_mean']:>9.4f} "
            f"{v['target_reached_rate'] * 100:>9.1f}%"
        )
    print("=" * 60)
    print(f"\nDetailed CSV: {summary_path}")
    print(f"Plots dir:    {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xLSTM-3 Generation Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.py file (optional)",
    )
    args = parser.parse_args()

    run(config_path=args.config)
