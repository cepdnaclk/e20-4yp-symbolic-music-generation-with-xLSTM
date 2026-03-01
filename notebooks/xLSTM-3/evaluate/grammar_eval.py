"""
evaluate/grammar_eval.py
========================
Reads generation_log.csv and computes per-condition aggregated statistics + plots.

Key metrics computed per (strategy, target_tokens) condition:
  - success_rate:          fraction of pieces that decoded to valid MIDI
  - grammar_error_rate:    mean error rate (errors / total_tokens)
  - target_reached_rate:   fraction where actual_tokens >= target_tokens
  - tokens_per_second:     mean generation throughput
  - tokens_per_bar:        mean tokens per bar across pieces

Outputs:
  - metrics/summary_by_condition.csv   — per-condition aggregated table
  - metrics/plots/                     — PNG plots for the paper
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports for plotting (graceful fallback if matplotlib missing)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for server use
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plots will not be generated.")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_log(log_csv: str) -> List[Dict]:
    """Load generation_log.csv into a list of dicts. Skips incomplete rows."""
    rows = []
    with open(log_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing key fields (e.g. from crashed pieces)
            if not row.get("strategy") or not row.get("target_tokens"):
                continue
            rows.append(row)
    logger.info("Loaded %d rows from %s", len(rows), log_csv)
    return rows


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def aggregate(rows: List[Dict]) -> Dict[tuple, Dict]:
    """
    Aggregate per-piece rows into per-condition statistics.

    Returns:
        Dict keyed by (strategy, target_tokens) with aggregated stats.
    """
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (row["strategy"], int(row["target_tokens"]))
        groups[key].append(row)

    results = {}
    for key, pieces in groups.items():
        n = len(pieces)
        success_list        = [_safe_bool(p["success"]) for p in pieces]
        target_reached_list = [_safe_bool(p["target_reached"]) for p in pieces]
        error_rates         = [_safe_float(p["grammar_error_rate"]) for p in pieces]
        tps_list            = [_safe_float(p["tokens_per_second"]) for p in pieces]
        tpb_list            = [_safe_float(p["tokens_per_bar_mean"]) for p in pieces]
        gen_time_list       = [_safe_float(p["generation_time_s"]) for p in pieces]
        actual_tok_list     = [_safe_float(p["actual_tokens"]) for p in pieces]
        num_bars_list       = [_safe_float(p["num_bars"]) for p in pieces]

        results[key] = {
            "strategy":             key[0],
            "target_tokens":        key[1],
            "n_pieces":             n,
            "success_rate":         sum(success_list) / n,
            "target_reached_rate":  sum(target_reached_list) / n,
            "grammar_error_rate_mean":   _mean(error_rates),
            "grammar_error_rate_std":    _std(error_rates),
            "tokens_per_second_mean":    _mean(tps_list),
            "tokens_per_bar_mean":       _mean(tpb_list),
            "generation_time_s_mean":    _mean(gen_time_list),
            "actual_tokens_mean":        _mean(actual_tok_list),
            "num_bars_mean":             _mean(num_bars_list),
        }

    return results


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "strategy", "target_tokens", "n_pieces",
    "success_rate", "target_reached_rate",
    "grammar_error_rate_mean", "grammar_error_rate_std",
    "tokens_per_second_mean", "tokens_per_bar_mean",
    "generation_time_s_mean", "actual_tokens_mean", "num_bars_mean",
]


def write_summary_csv(aggregated: Dict[tuple, Dict], output_path: str) -> None:
    """Write the per-condition summary table to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        # Sort by strategy then target_tokens for readability
        for key in sorted(aggregated.keys(), key=lambda k: (k[0], k[1])):
            row = aggregated[key]
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in row.items()})
    logger.info("Summary CSV written to: %s", output_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STRATEGY_COLOURS = {
    "single_shot": "#4C72B0",
    "chunked":     "#DD8452",
}
STRATEGY_LABELS = {
    "single_shot": "Single-shot",
    "chunked":     "Chunked",
}


def _plot_metric_vs_tokens(
    aggregated: Dict[tuple, Dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: str,
    ylim: tuple | None = None,
) -> None:
    """Generic helper: plot metric_key vs target_tokens, one line per strategy."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for strategy in ["single_shot", "chunked"]:
        pairs = sorted(
            [(k[1], v[metric_key]) for k, v in aggregated.items() if k[0] == strategy],
            key=lambda x: x[0],
        )
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLOURS[strategy], linewidth=2, markersize=7)

    ax.set_xlabel("Target tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    if ylim:
        ax.set_ylim(ylim)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def generate_plots(aggregated: Dict[tuple, Dict], plots_dir: str) -> None:
    """Generate all paper-ready plots from aggregated statistics."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping plots (matplotlib not available).")
        return

    pd = plots_dir

    _plot_metric_vs_tokens(
        aggregated, "grammar_error_rate_mean",
        ylabel="Grammar error rate (errors / tokens)",
        title="Grammar Error Rate vs Target Tokens",
        output_path=f"{pd}/grammar_error_rate.png",
        ylim=(0, None),
    )

    _plot_metric_vs_tokens(
        aggregated, "success_rate",
        ylabel="Success rate",
        title="MIDI Decode Success Rate vs Target Tokens",
        output_path=f"{pd}/success_rate.png",
        ylim=(0, 1.05),
    )

    _plot_metric_vs_tokens(
        aggregated, "target_reached_rate",
        ylabel="Target-reached rate",
        title="Fraction of Pieces Reaching Target Token Count",
        output_path=f"{pd}/target_reached_rate.png",
        ylim=(0, 1.05),
    )

    _plot_metric_vs_tokens(
        aggregated, "tokens_per_second_mean",
        ylabel="Tokens / second",
        title="Generation Throughput vs Target Tokens",
        output_path=f"{pd}/tokens_per_second.png",
    )

    _plot_metric_vs_tokens(
        aggregated, "tokens_per_bar_mean",
        ylabel="Mean tokens per bar",
        title="Token Density per Bar vs Target Tokens",
        output_path=f"{pd}/tokens_per_bar.png",
    )
