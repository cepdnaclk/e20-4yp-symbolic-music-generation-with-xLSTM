# xLSTM Perplexity Evaluation Pipeline - Implementation Plan

> **Goal**: Create a reusable, configurable Jupyter notebook pipeline for comprehensive perplexity evaluation of any trained xLSTM model.

---

## Overview

This pipeline will automate the perplexity evaluation process we developed previously, making it:
- **Configurable**: Easy to adapt for different models, hyperparameters, and datasets
- **Comprehensive**: Test multiple checkpoints × multiple context lengths
- **Visual**: Generate publication-ready charts and tables
- **Reproducible**: Save all results with proper naming conventions
- **Automatic**: Auto-detect model configuration from config.yaml

---

## Data Split Usage Strategy

> **Important**: Following standard ML practice, we use different data splits for different purposes.

| Split | Purpose | Charts |
|-------|---------|--------|
| **valid.txt** | Find best checkpoint (model selection) | Chart 1, 3, 4, 6 |
| **test.txt** | Final evaluation of best checkpoint (reporting) | Chart 2, 5 |

**Rationale**:
- Validation set is used for hyperparameter tuning and checkpoint selection (prevents data leakage)
- Test set is used only for final evaluation to get unbiased performance metrics
- This is standard practice in ML to ensure reported results are not overfit to the selection process

---

## Configuration Design

### Centralized Configuration Cell
All configurable parameters in one place at the top of the notebook:

```python
CONFIG = {
    # Model paths (ONLY REQUIRED INPUT)
    "model_run_path": "path/to/helibrunna/output/run_YYYYMMDD-HHMM",
    
    # Evaluation settings
    "context_lengths": [1024, 2048, 3072, 4096, 5120, 10240],
    "checkpoint_selection": "auto",  # "auto", "all", or list of checkpoint names
    "auto_checkpoint_count": 6,      # Number of checkpoints if "auto"
    
    # Output settings
    "results_base_dir": "./results",
    "save_plots": True,
    "plot_format": "png",  # or "pdf" for publications
}

# AUTO-DETECTED FROM config.yaml (no manual input needed):
# - model_name: from training.model_name
# - training_context_length: from model.context_length
# - tokenizer_path: {model_run_path}/tokenizer.json
# - valid_data_path: from dataset.local_valid
# - test_data_path: from dataset.local_test
# - embedding_dim, num_blocks, vocab_size, etc.
```

### Auto-Detection from config.yaml

The notebook will automatically read `config.yaml` from the run folder:

```yaml
# Example config.yaml structure:
training:
  model_name: xlstm_lmd_512d_4096ctx_12b  # → auto-detect
  batch_size: 1
  lr: 0.001
  ...
model:
  embedding_dim: 512           # → auto-detect
  context_length: 4096         # → auto-detect (training_context_length)
  num_blocks: 12               # → auto-detect
  vocab_size: 675              # → auto-detect
  ...
dataset:
  local_valid: path/to/valid.txt  # → auto-detect
  local_test: path/to/test.txt    # → auto-detect
```

### Output Folder Naming Convention
```
results/
└── {model_name}_{timestamp}/
    ├── config.json              # Copy of configuration used + auto-detected values
    ├── model_info.json          # Auto-detected model metadata
    ├── checkpoint_search/       # Validation results for checkpoint selection
    │   ├── all_results.json
    │   ├── all_results.csv
    │   └── plots/
    ├── final_evaluation/        # Test results for best checkpoint  
    │   ├── results.json
    │   ├── results.csv
    │   └── plots/
    └── summary.md               # Auto-generated summary report
```

---

## Implementation Phases

### Phase 1: Project Setup ✅
- [x] Create folder structure: `evaluation-pipeline/`
- [x] Create `perplexity-pipeline-plan.md` (this file)
- [x] Create `perplexity-pipeline-doc.md` for documentation

### Phase 2: Notebook Foundation ✅
- [x] Create `perplexity_evaluation.ipynb` 
- [x] Add configuration cell with minimal required inputs
- [x] Implement `load_config_yaml()` to auto-detect model settings
- [x] Display auto-detected configuration to user with nice formatting
- [x] Add imports and utility functions
- [x] Add model loading helper with config override support

### Phase 3: Checkpoint Discovery & Selection ✅
- [x] Implement `discover_checkpoints()` - find all checkpoints in run directory
- [x] Implement `select_checkpoints()` - auto-select evenly spaced checkpoints
- [x] Display discovered checkpoints to user with visualization
- [x] Allow manual override of selection

### Phase 4: Data Loading ✅
- [x] Implement `load_data()` supporting both valid and test splits
- [x] Display data statistics with visualizations:
  - [x] Histogram of sequence lengths
  - [x] Summary statistics table (min, max, mean, median, count)
  - [x] Pie chart of sequences by length category (short/medium/long)
- [x] Cache tokenized sequences for efficiency

### Phase 5: Perplexity Computation ✅
- [x] Implement `compute_perplexity()` using existing logic
- [x] Add progress bars for long evaluations
- [x] Handle context length override for contexts > training context
- [x] Implement batch evaluation: all checkpoints × all contexts
- [x] Build results DataFrame with all combinations

### Phase 6: Best Checkpoint Selection ✅
- [x] **Use VALIDATION set** for checkpoint selection
- [x] Selection strategy: **Average PPL across all context lengths**
  - For each checkpoint, compute avg(PPL@1024, PPL@2048, ..., PPL@10240)
  - Select checkpoint with lowest average PPL
- [x] Detect overfitting point (where PPL starts increasing)
- [x] Display selection reasoning to user

### Phase 7: Final Evaluation ✅
- [x] **Use TEST set** for final evaluation of best checkpoint
- [x] Evaluate best checkpoint at ALL context lengths
- [x] Compute extrapolation quality metrics

### Phase 8: Visualization ✅

| Chart# | Name | Data Split | Purpose |
|--------|------|------------|---------|
| 1 | Checkpoint vs PPL | Validation | Find best checkpoint across training progress |
| 2 | Context vs PPL | Test | Show extrapolation ability of best checkpoint |
| 3 | Heatmap | Validation | Quick overview of all checkpoint × context results |
| 4 | PPL Degradation Rate | Validation | Quantify extrapolation quality by checkpoint |
| 5 | Training Progress | Both | Correlate training loss with validation PPL |
| 6 | Best Checkpoint Bar | Validation | Clear identification of optimal checkpoint |

- [x] **Chart 1**: Checkpoint vs PPL (Multi-line)
- [x] **Chart 2**: Context Length vs PPL (Best Checkpoint)
- [x] **Chart 3**: Heatmap of Checkpoint × Context Length
- [x] **Chart 4**: PPL Degradation Rate
- [x] **Chart 5**: Training Progress vs PPL
- [x] **Chart 6**: Best Checkpoint Identification

### Phase 9: Result Export ✅
- [x] Save raw results to JSON
- [x] Save summary tables to CSV
- [x] Save all plots to files
- [x] Generate auto-summary markdown report
- [x] Copy config + model_info to results folder for reproducibility

### Phase 10: Documentation & Polish ✅
- [x] Add comprehensive markdown cells explaining each step
- [x] Add example outputs and interpretations
- [x] Document error handling and troubleshooting
- [x] Test with current model to validate pipeline

---

## Visualization Details

### Chart 1: Checkpoint vs PPL (Multi-line)
- **Data Split**: Validation
- **X-axis**: Checkpoint step number
- **Y-axis**: Perplexity
- **Lines**: One line per context length (different colors)
- **Purpose**: See how PPL evolves over training for each context length
- **Highlights**: Mark minimum PPL point on each line, identify best checkpoint

### Chart 2: Context Length vs PPL (Best Checkpoint)
- **Data Split**: Test
- **X-axis**: Context length (log scale optional)
- **Y-axis**: Perplexity (log scale)
- **Line**: Single line for the best checkpoint
- **Purpose**: Show extrapolation ability of the best checkpoint
- **Highlights**: 
  - Mark training context length with vertical line
  - Show PPL value annotations at each point
  - Shade "within training context" vs "extrapolation" regions

### Chart 3: Heatmap (Checkpoint × Context)
- **Data Split**: Validation
- **X-axis**: Context lengths
- **Y-axis**: Checkpoints (sorted by step)
- **Color**: PPL value (color scale with best = green, worst = red)
- **Purpose**: Quick visual overview of all results
- **Highlights**: Star or border on best cell

### Chart 4: PPL Degradation Rate
- **Data Split**: Validation
- **X-axis**: Context length
- **Y-axis**: PPL increase rate (% change per 1024 tokens from training context)
- **Bars/Lines**: Multiple checkpoints for comparison
- **Purpose**: Quantify how fast models degrade beyond training context

### Chart 5: Training Progress Correlation
- **Data Split**: Training (history.json) + Validation (our PPL)
- **X-axis**: Training step
- **Y-axis (left)**: Training loss (from history.json)
- **Y-axis (right)**: Validation PPL (from our evaluation @ training context length)
- **Purpose**: Correlate training loss with actual validation performance
- **Data source**: `history.json` format:
  ```json
  {
    "loss": [],
    "lr": [],
    "epoch": [],
    "step": []
  }
  ```

### Chart 6: Best Checkpoint Bar Chart
- **Data Split**: Validation
- **X-axis**: Checkpoints
- **Y-axis**: Average PPL across all context lengths
- **Bars**: All checkpoints with best highlighted in different color
- **Purpose**: Clear identification of optimal checkpoint

---

## Best Checkpoint Selection Algorithm

```
1. Load validation data
2. For each selected checkpoint:
   a. For each context length in context_lengths:
      - Compute PPL on validation set
   b. Compute avg_ppl = mean(PPL@ctx1, PPL@ctx2, ..., PPL@ctxN)
3. Select checkpoint with lowest avg_ppl as "best"
4. Final evaluation: Evaluate best checkpoint on TEST set (all contexts)
```

### Why Average PPL?
- Using average PPL across contexts balances:
  - Performance at training context (most common use case)
  - Extrapolation ability (important for long music generation)
- Alternative strategies available if needed:
  - Weighted average (prioritize certain context lengths)
  - Minimum at specific context (e.g., training context only)

---

## Automatic Checkpoint Selection Algorithm

```
1. List all checkpoint-XXXXX folders in run directory
2. Parse step numbers from folder names
3. Sort by step number
4. If "auto" selection:
   a. Always include first checkpoint (earliest)
   b. Always include last checkpoint (final)
   c. Select (N-2) evenly spaced checkpoints in between
5. Return selected checkpoint names
```

---

## Error Handling

- **CUDA OOM**: Catch and skip context lengths that cause OOM, log warning
- **Missing checkpoints**: Validate all selected checkpoints exist before starting
- **Missing config.yaml**: Fail early with clear error message
- **Corrupted results**: Save intermediate results after each checkpoint evaluation
- **Interrupted evaluation**: Support resuming from saved intermediate results

---

## Dependencies

- `torch` - Model inference
- `matplotlib` / `seaborn` - Visualization
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `tqdm` - Progress bars (standard, not notebook version)
- `tabulate` - Required for `DataFrame.to_markdown()`
- `json` / `yaml` - Config and results
- `helibrunna` - Model loading (existing)
- `transformers` - Tokenizer

---

## Verification Plan

### Manual Verification
1. **Config Auto-Detection**: Verify model settings are correctly read from config.yaml
2. **Checkpoint Discovery**: Verify all checkpoints are found
3. **Data Loading**: Verify both valid and test sets load correctly
4. **PPL Computation**: Compare results with previous manual runs (should match)
5. **Visualization**: Verify all 6 charts render correctly with proper labels
6. **Export**: Verify all files are saved to correct locations

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `evaluation-pipeline/perplexity-pipeline-plan.md` | This plan document | ✅ |
| `evaluation-pipeline/perplexity-pipeline-doc.md` | Implementation documentation | ✅ |
| `evaluation-pipeline/perplexity_evaluation.ipynb` | Main evaluation notebook | ✅ |
| `evaluation-pipeline/README.md` | Quick start guide | ✅ |

---

## Reference

This pipeline is based on the manual evaluation process documented in:
- `xlstm-evaluation/perplexity/perplexity-doc.md`
- `xlstm-evaluation/perplexity/evaluate_perplexity.py`
