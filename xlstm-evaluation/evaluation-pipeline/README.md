# xLSTM Perplexity Evaluation Pipeline

A comprehensive, reusable pipeline for evaluating xLSTM model perplexity across multiple checkpoints and context lengths.

## Quick Start

1. Open `perplexity_evaluation.ipynb` in Jupyter
2. Configure the parameters in the first cell
3. Run all cells
4. Find results in the auto-generated `results/` folder

## Configuration

Edit the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    "model_run_path": "path/to/helibrunna/output/run_YYYYMMDD-HHMM",
    "context_lengths": [1024, 2048, 3072, 4096, 5120, 10240],
    "checkpoint_selection": "auto",
    # ... more options
}
```

## Features

- **Automatic checkpoint discovery** - Finds all checkpoints in training run
- **Smart checkpoint selection** - Auto-selects evenly spaced checkpoints
- **Comprehensive evaluation** - Tests all checkpoint × context combinations
- **Publication-ready visualizations** - 6 different chart types
- **Reproducible results** - Saves config + results with timestamped folders

## Output

Results are saved to `results/{model_name}_{timestamp}/`:
- `config.json` - Configuration used
- `raw_results.json` - All PPL values
- `summary.md` - Auto-generated report
- `plots/` - All visualization charts
- `tables/` - CSV exports

## Documentation

- [`perplexity-pipeline-plan.md`](./perplexity-pipeline-plan.md) - Implementation plan
- [`perplexity-pipeline-doc.md`](./perplexity-pipeline-doc.md) - Implementation log
