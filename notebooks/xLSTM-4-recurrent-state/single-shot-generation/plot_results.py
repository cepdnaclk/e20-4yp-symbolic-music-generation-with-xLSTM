"""
plot_results.py
===============
Standalone plotting script for xLSTM generation metrics.
Generates individual PDF plots organized by category.

Usage:
    # Manual plotting
    python plot_results.py --csv results/quick_test/generation_metrics.csv

    # Specify custom output directory
    python plot_results.py --csv results/quick_test/generation_metrics.csv --output-dir results/quick_test

    # Or called automatically from run_generation.py
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'pdf'


def load_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load and validate generation metrics CSV.

    Args:
        csv_path: Path to generation_metrics.csv

    Returns:
        Pandas DataFrame with metrics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = [
        'target_tokens', 'generation_time_s', 'tokens_per_second',
        'grammar_error_rate', 'success', 'num_bars', 'num_notes'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Target lengths: {sorted(df['target_tokens'].unique())}")
    print(f"Pieces per length: {df.groupby('target_tokens').size().to_dict()}")

    return df


def save_plot(fig, output_path: str):
    """Save plot as PDF only."""
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(output_path)}")


def plot_tokens_per_second(df: pd.DataFrame, output_dir: str):
    """
    Plot generation speed (tokens/second) vs target tokens.
    Should be constant for O(N) scaling.
    """
    grouped = df.groupby('target_tokens').agg({
        'tokens_per_second': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['tokens_per_second']['mean'].values
    y_std = grouped['tokens_per_second']['std'].values

    ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Generation Speed')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_val = y_mean.mean()
    ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_val:.1f} tok/s')
    ax.legend()

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'performance', 'tokens_per_second.pdf'))


def plot_generation_time(df: pd.DataFrame, output_dir: str):
    """
    Plot generation time vs target tokens.
    Should be linear for O(N) scaling.
    """
    grouped = df.groupby('target_tokens').agg({
        'generation_time_s': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['generation_time_s']['mean'].values
    y_std = grouped['generation_time_s']['std'].values

    ax.errorbar(x, y_mean, yerr=y_std, marker='s', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Generation Time (seconds)')
    ax.set_title('Generation Time')
    ax.grid(True, alpha=0.3)

    # Fit linear trend
    z = np.polyfit(x, y_mean, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.5,
            label=f'Linear fit: {z[0]:.2e}·N + {z[1]:.2f}')
    ax.legend()

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'performance', 'generation_time.pdf'))


def plot_grammar_error_rate(df: pd.DataFrame, output_dir: str):
    """
    Plot grammar error rate vs target tokens.
    """
    grouped = df.groupby('target_tokens').agg({
        'grammar_error_rate': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['grammar_error_rate']['mean'].values * 100  # Percentage
    y_std = grouped['grammar_error_rate']['std'].values * 100

    ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#C73E1D')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Grammar Error Rate (%)')
    ax.set_title('Token Grammar Quality')
    ax.grid(True, alpha=0.3)

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'quality', 'grammar_error_rate.pdf'))


def plot_num_bars(df: pd.DataFrame, output_dir: str):
    """
    Plot number of musical bars vs target tokens.
    """
    grouped = df.groupby('target_tokens').agg({
        'num_bars': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['num_bars']['mean'].values
    y_std = grouped['num_bars']['std'].values

    ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#6A4C93')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Number of Bars')
    ax.set_title('Musical Bars Generated')
    ax.grid(True, alpha=0.3)

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'musical', 'num_bars.pdf'))


def plot_num_notes(df: pd.DataFrame, output_dir: str):
    """
    Plot number of musical notes vs target tokens.
    """
    grouped = df.groupby('target_tokens').agg({
        'num_notes': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['num_notes']['mean'].values
    y_std = grouped['num_notes']['std'].values

    ax.errorbar(x, y_mean, yerr=y_std, marker='s', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#1B9AAA')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Number of Notes')
    ax.set_title('Musical Notes Generated')
    ax.grid(True, alpha=0.3)

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'musical', 'num_notes.pdf'))


def plot_tokens_per_bar(df: pd.DataFrame, output_dir: str):
    """
    Plot tokens per bar vs target tokens.
    """
    grouped = df.groupby('target_tokens').agg({
        'tokens_per_bar_mean': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = grouped['target_tokens'].values
    y_mean = grouped['tokens_per_bar_mean']['mean'].values
    y_std = grouped['tokens_per_bar_mean']['std'].values

    ax.errorbar(x, y_mean, yerr=y_std, marker='^', capsize=5,
                capthick=2, linewidth=2, markersize=8, color='#7D5A50')
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Tokens per Bar')
    ax.set_title('Bar Density')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_val = y_mean.mean()
    ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_val:.1f}')
    ax.legend()

    # Use exact x values
    ax.set_xticks(x)

    save_plot(fig, os.path.join(output_dir, 'musical', 'tokens_per_bar.pdf'))


def generate_plots(csv_path: str, output_dir: str = None) -> str:
    """
    Main function: generate all plots from metrics CSV.

    Args:
        csv_path: Path to generation_metrics.csv
        output_dir: Directory to save plots (default: same dir as CSV)

    Returns:
        Path to plots directory
    """
    # Default output_dir to CSV's parent directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # Create plots directory structure
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "performance"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "quality"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "musical"), exist_ok=True)

    print(f"\nGenerating plots from: {csv_path}")
    print(f"Saving to: {plots_dir}\n")

    # Load data
    df = load_metrics(csv_path)

    print("\nGenerating plots:")
    print("Performance:")
    plot_tokens_per_second(df, plots_dir)
    plot_generation_time(df, plots_dir)

    print("\nQuality:")
    plot_grammar_error_rate(df, plots_dir)

    print("\nMusical:")
    plot_num_bars(df, plots_dir)
    plot_num_notes(df, plots_dir)
    plot_tokens_per_bar(df, plots_dir)

    print(f"\n✓ All plots saved to: {plots_dir}")
    print(f"  - performance/ (2 plots)")
    print(f"  - quality/ (1 plot)")
    print(f"  - musical/ (3 plots)")
    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from xLSTM generation metrics CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for quick_test results
  python plot_results.py --csv results/quick_test/generation_metrics.csv

  # Specify custom output directory
  python plot_results.py --csv results/quick_test/generation_metrics.csv --output-dir custom_plots/
        """
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to generation_metrics.csv file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as CSV location)"
    )

    args = parser.parse_args()

    try:
        plots_dir = generate_plots(args.csv, args.output_dir)
        print(f"\nSuccess! View plots in: {plots_dir}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
