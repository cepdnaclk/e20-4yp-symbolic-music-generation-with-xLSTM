# xLSTM Perplexity Evaluation Pipeline - Implementation Documentation

> This document tracks progress during implementation of the evaluation pipeline.

---

## Status Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project Setup | ✅ Complete |
| Phase 2 | Notebook Foundation | ✅ Complete |
| Phase 3 | Checkpoint Discovery & Selection | ✅ Complete |
| Phase 4 | Data Loading | ✅ Complete |
| Phase 5 | Perplexity Computation | ✅ Complete |
| Phase 6 | Best Checkpoint Selection | ✅ Complete |
| Phase 7 | Final Evaluation | ✅ Complete |
| Phase 8 | Visualization | ✅ Complete |
| Phase 9 | Result Export | ✅ Complete |
| Phase 10 | Documentation & Polish | ✅ Complete |

---

## Phase 1: Project Setup ✅

### Completed
- Created `evaluation-pipeline/` folder
- Created `perplexity-pipeline-plan.md`
- Created `perplexity-pipeline-doc.md` (this file)
- Created `README.md`

---

## Phase 2: Notebook Foundation ✅

### Implemented
- Configuration cell with centralized `CONFIG` dict
- Imports cell with all required libraries
- `load_model_config()` function for auto-detection from config.yaml

### Key Functions
- `load_model_config(run_path)` → Returns dict with model_name, training_context_length, etc.

---

## Phase 3: Checkpoint Discovery & Selection ✅

### Implemented
- `discover_checkpoints(run_path)` → Returns sorted list of (name, step) tuples
- `select_checkpoints(all_checkpoints, selection, count)` → Returns selected checkpoint names

### Selection Strategies
- `"auto"`: Evenly spaced checkpoints including first and last
- `"all"`: All checkpoints
- `list`: Specific checkpoint names

---

## Phase 4: Data Loading ✅

### Implemented
- `load_tokenizer(tokenizer_path)` → PreTrainedTokenizerFast
- `load_and_tokenize_data(data_path, tokenizer)` → List of token sequences
- `visualize_data_stats(sequences, title)` → Histogram, box plot, pie chart

### Visualizations
1. Histogram of sequence lengths with median line
2. Box plot of sequence lengths
3. Pie chart of length categories (<1024, 1024-4096, >=4096)

---

## Phase 5: Perplexity Computation ✅

### Implemented
- `compute_perplexity(model, sequences, context_length)` → float
- `load_model_checkpoint(run_path, checkpoint_name, context_override)` → LanguageModel
- `run_batch_evaluation(checkpoints, context_lengths, sequences, run_path)` → DataFrame

### Features
- Progress bars with tqdm
- CUDA OOM handling (catches and logs, continues)
- Context length override for long contexts
- Memory cleanup between checkpoints

---

## Phase 6: Best Checkpoint Selection ✅

### Implemented
- Average PPL calculation across all context lengths
- Ranking checkpoints by average PPL
- Display best checkpoint with reasoning

### Selection Logic
```python
avg_ppl = VALID_RESULTS.groupby('checkpoint')['perplexity'].mean()
BEST_CHECKPOINT = avg_ppl.idxmin()
```

---

## Phase 7: Final Evaluation ✅

### Implemented
- Test set evaluation for best checkpoint only
- All context lengths evaluated

---

## Phase 8: Visualization ✅

### Charts Implemented

| # | Chart | Description |
|---|-------|-------------|
| 1 | Checkpoint vs PPL | Multi-line plot with minimum markers |
| 2 | Context vs PPL | Best checkpoint with training context marker |
| 3 | Heatmap | Checkpoint × Context with color scale |
| 4 | Degradation Rate | PPL increase % beyond training context |
| 5 | Training Progress | Dual-axis: training loss + validation PPL |
| 6 | Best Checkpoint Bar | Bar chart with best highlighted |

---

## Phase 9: Result Export ✅

### Output Structure
```
results/{model_name}_{timestamp}/
├── config.json
├── validation_results.json
├── test_results.json
├── summary.md
├── tables/
│   ├── validation_results.csv
│   └── test_results.csv
└── plots/
    └── (charts if save_plots=True)
```

---

## Notebook Cell Summary

| Cell # | Type | Description |
|--------|------|-------------|
| 1 | Markdown | Title and introduction |
| 2 | Code | Configuration |
| 3 | Code | Imports |
| 4 | Markdown | Phase 2 header |
| 5 | Code | Config auto-detection |
| 6 | Markdown | Phase 3 header |
| 7 | Code | Checkpoint discovery |
| 8 | Markdown | Phase 4 header |
| 9 | Code | Data loading functions |
| 10 | Code | Load validation data |
| 11 | Code | Load test data |
| 12 | Markdown | Phase 5 header |
| 13 | Code | Perplexity functions |
| 14 | Code | Batch evaluation function |
| 15 | Markdown | Phase 6 header |
| 16 | Code | Run validation evaluation |
| 17 | Code | Find best checkpoint |
| 18 | Markdown | Phase 7 header |
| 19 | Code | Run test evaluation |
| 20 | Markdown | Phase 8 header |
| 21 | Code | Chart 1 |
| 22 | Code | Chart 2 |
| 23 | Code | Chart 3 |
| 24 | Code | Chart 4 |
| 25 | Code | Chart 5 |
| 26 | Code | Chart 6 |
| 27 | Markdown | Phase 9 header |
| 28 | Code | Export results |
| 29 | Code | Generate summary |
| 30 | Markdown | Final summary |

---

## Problems Encountered

### Problem 1: tqdm.notebook ImportError
**Error**: `ImportError: IProgress not found. Please update jupyter and ipywidgets`

**Cause**: `tqdm.notebook` requires `ipywidgets` package which was not installed in the conda environment.

**Fix**: Changed the import to use a try/except fallback:
```python
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
```

**Date**: 2026-02-08

---

### Problem 2: Very Slow Evaluation (Hours for Full Run)
**Observed**: Cell 10 (validation evaluation) took ~9 hours for 36 evaluations (6 checkpoints × 6 contexts)

**Rate**: ~15 minutes per checkpoint-context combination

**Root Cause**: The `compute_perplexity()` function processes sequences **one at a time** without batching:
```python
for seq in sequences:  # Iterates through ~3000 sequences one by one
    input_ids = torch.tensor([seq[:-1]], device='cuda')  # Batch size = 1
    logits = model.model(input_ids)
```

**Why It's Slow**:
1. **No batching**: GPU processes one sequence at a time (very inefficient)
2. **Large dataset**: ~3000 validation sequences × 36 evaluations = 108,000 forward passes
3. **Longer contexts = slower**: Context 10240 takes 10× longer than 1024

**Future Improvements** (TODO):
- [ ] Add batched inference to `compute_perplexity()` 
- [ ] Add `max_samples` parameter to limit sequences for quick tests
- [ ] Multi-GPU parallelization (3× RTX 6000 available)

**Fix Applied**: Changed to two-phase evaluation strategy:
1. **Phase 6**: Evaluate all checkpoints at **training context only** → 6 evaluations
2. **Phase 7**: Evaluate **best checkpoint** at all context lengths → 6+6 evaluations (valid + test)

**Result**: Reduced from 36 evaluations to 18 evaluations (2× faster)

**Date**: 2026-02-08

---

### Problem 3: NameError - VALID_RESULTS not defined
**Error**: `NameError: name 'VALID_RESULTS' is not defined` in export cell

**Cause**: After refactoring to two-phase evaluation, we renamed `VALID_RESULTS` to `CHECKPOINT_RESULTS`, but the export cell wasn't updated.

**Fix**: Updated export cell to use:
- `CHECKPOINT_RESULTS` (checkpoint selection results)
- `TEST_RESULTS` (final test results)
- `BEST_PPL` (instead of `BEST_AVG_PPL`)

**Date**: 2026-02-08

---

### Problem 4: Missing 'tabulate' dependency
**Error**: `ImportError: Missing optional dependency 'tabulate'` in summary generation cell

**Cause**: `DataFrame.to_markdown()` requires the `tabulate` package which wasn't installed.

**Fix**: Installed tabulate in the xlstm conda environment:
```bash
/home/e20037/miniconda/envs/xlstm/bin/pip install tabulate
```

**Note**: Must use the environment-specific pip path, not just `pip install`.

**Date**: 2026-02-08

---

## Reference

Based on manual evaluation documented in:
- [`../perplexity/perplexity-doc.md`](../perplexity/perplexity-doc.md)
- [`../perplexity/evaluate_perplexity.py`](../perplexity/evaluate_perplexity.py)
