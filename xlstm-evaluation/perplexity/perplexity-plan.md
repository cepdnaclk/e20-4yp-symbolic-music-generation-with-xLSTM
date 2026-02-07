# Perplexity Evaluation Implementation Plan

## Overview

This document outlines the implementation plan for calculating perplexity on the **test set** for the trained xLSTM music generation model.

**Goal**: Measure how well the xLSTM model predicts the next token in unseen music sequences.

> **Note on Test vs Validation**:
> - **Test set**: Used for final perplexity reporting (primary evaluation)
> - **Validation set**: Used only for checkpoint selection (if needed)
> - This follows standard ML practice and Museformer's methodology

---

## Agent Instructions

> **IMPORTANT**: When implementing this plan, the agent should:
> 1. **Update this plan** - Mark checkboxes as completed `[x]` when each phase is done
> 2. **Document in `perplexity-doc.md`** - Create and update this file to record:
>    - Important decisions made during implementation
>    - Problems encountered and how they were solved
>    - Any deviations from the original plan
>    - Useful observations or insights
> 3. **Follow modular coding practices** - Split code into well-organized files

---

## Project Context Summary

| Component | Location |
|-----------|----------|
| **Trained Model** | `/repos/helibrunna/output/xlstm_lmd_512d_2048ctx_12b/run_20260126-0516/` |
| **Final Checkpoint** | `checkpoint-158760-last/` |
| **Test Data** | `/data/lmd_preprocessed/splits/test.txt` (~293MB, ~3000 songs) |
| **Validation Data** | `/data/lmd_preprocessed/splits/valid.txt` (~298MB) |
| **Training Config** | 512d embedding, 2048 context length, 12 blocks |
| **Tokenization** | REMIGEN2 format, whitespace-separated tokens |
| **Conda Environment** | `conda activate xlstm` (required before running scripts) |

---

## Context Lengths for Perplexity Testing

Following Museformer's evaluation methodology, we calculate PPL on the **first N tokens** of each song:

| Context Length | Purpose |
|----------------|---------|
| **1,024 tokens** | Short context - baseline comparison |
| **5,120 tokens** | Medium context - standard evaluation |
| **10,240 tokens** | Long context - Museformer's max (for direct comparison) |
| **16,384 tokens** | Extrapolation test - xLSTM's advantage beyond Museformer's limit |

### Why These Lengths?

- Museformer showed Transformers degrade at longer contexts (PPL explodes beyond training length)
- xLSTM should maintain low PPL at all lengths due to recurrent memory architecture
- Testing at 16k demonstrates xLSTM's extrapolation capability (trained on 2k context)

## What is Perplexity?

Perplexity measures how "surprised" a model is when predicting the next token:

```
PPL = exp(average negative log-likelihood)
    = exp(-1/N × Σ log P(token_i | context))
```

- **Lower perplexity = better model** (less surprised by correct tokens)
- Standard metric for language models, used by Museformer for comparison

---

## Implementation Strategy: Teacher Forcing

Following Museformer's approach in `musformer_ppl_eval.py`:

```python
# For each sequence:
inputs  = tokens[:-1]   # All tokens except last
targets = tokens[1:]    # All tokens except first

logits = model.predict(inputs)   # Get predictions
nll = -log_softmax(logits)[targets]  # Negative log-likelihood
```

---

## Files to Create (Modular Structure)

```
xlstm-evaluation/perplexity/
├── perplexity-plan.md           # This document (update checkboxes during implementation)
├── perplexity-doc.md            # Implementation notes and problem documentation
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # Configuration constants and paths
│   ├── data_loader.py           # Load and process test/valid data
│   ├── perplexity_calculator.py # Core PPL computation logic
│   └── utils.py                 # Helper functions (logging, file I/O)
├── evaluate_perplexity.py       # Main entry point script
├── run_all_contexts.sh          # Bash script to run all context lengths
└── results/                     # Output directory (gitignored)
    └── (generated results)
```

---

## Implementation Phases

### Phase 1: Project Setup
- [x] Create directory structure (`src/`, `results/`)
- [x] Create `perplexity-doc.md` documentation file
- [x] Create `src/__init__.py`
- [x] Create `src/config.py` with paths and constants

### Phase 2: Data Loading Module
- [x] Create `src/data_loader.py`
- [x] Implement `load_tokenized_sequences()` function
- [x] Implement `get_sequence_stats()` for data analysis
- [x] Test data loading with a small sample (5-10 sequences)

### Phase 3: Perplexity Calculator Module
- [x] Create `src/perplexity_calculator.py`
- [x] Implement `compute_perplexity()` function
- [x] Implement `compute_perplexity_multi_context()` for multiple context lengths
- [x] Handle edge cases (short sequences, padding tokens)

### Phase 4: Utility Functions
- [x] Create `src/utils.py`
- [x] Implement logging setup
- [x] Implement results saving (JSON format)
- [x] Implement table formatting

### Phase 5: Main Evaluation Script
- [x] Create `evaluate_perplexity.py` with CLI arguments
- [x] Integrate all modules
- [x] Add comprehensive logging
- [x] Test with `--max-samples 10` first

### Phase 5.5: Checkpoint Evaluation Support (NEW)
- [x] Add `--checkpoint` argument to `evaluate_perplexity.py` to select specific checkpoint
- [x] Modify Helibrunna `LanguageModel` with `checkpoint_name` parameter
- [ ] Create `evaluate_checkpoints.py` script to run PPL on multiple checkpoints
- [ ] Parse `history.json` to extract training loss at checkpoint steps

### Phase 6: Run Evaluations
**Final Checkpoint (for Museformer comparison):**
- [ ] Run PPL at 1,024 tokens on test set (final checkpoint)
- [ ] Run PPL at 5,120 tokens on test set (final checkpoint)
- [ ] Run PPL at 10,240 tokens on test set (final checkpoint)
- [ ] Run PPL at 16,384 tokens on test set (extrapolation demo)

**Multiple Checkpoints (Training Dynamics Analysis):**
- [ ] Run PPL at 1,024 tokens on checkpoints: 20k, 40k, 80k, 120k, 150k, -last
- [ ] Extract training loss from history.json for these steps

### Phase 7: Analysis and Documentation
- [ ] Compile results into summary tables
- [ ] Create visualization: PPL vs context length (final checkpoint)
- [ ] Create visualization: Training loss vs Validation PPL (checkpoint analysis)
- [ ] Update `perplexity-doc.md` with findings
- [ ] Compare with Museformer results (if available)

---

## Checkpoint Evaluation Strategy

### Current Checkpoint Behavior (Helibrunna)

The `LanguageModel` class in `repos/helibrunna/source/languagemodel.py` (lines 85-100) currently **only loads checkpoints ending with `-last`**:

```python
# Current behavior - only finds "-last" checkpoints
checkpoint_folders = glob.glob(os.path.join(model_path_or_repo, "checkpoint-*"))
for checkpoint_folder in checkpoint_folders:
    if checkpoint_folder.endswith("-last"):
        model_path = checkpoint_folder
        break
```

> [!IMPORTANT]
> To evaluate intermediate checkpoints, we need to either:
> 1. Modify Helibrunna's `LanguageModel` to accept a `checkpoint_name` parameter, OR
> 2. Pass the **full checkpoint path** directly instead of the run directory

### Available Checkpoints

The training run has **80 checkpoints** saved every 2,000 steps:

| Checkpoint | Training Step | ~Progress |
|------------|---------------|-----------|
| `checkpoint-10000` | 10,000 | ~6% |
| `checkpoint-40000` | 40,000 | ~25% |
| `checkpoint-80000` | 80,000 | ~50% |
| `checkpoint-120000` | 120,000 | ~75% |
| `checkpoint-158760-last` | 158,760 | 100% |

**Location**: `/repos/helibrunna/output/xlstm_lmd_512d_2048ctx_12b/run_20260126-0516/`

### Training History

The `history.json` file in the run directory contains:
- **loss**: Training loss at each step
- **lr**: Learning rate schedule
- **epoch**: Current epoch
- **step**: Training step number

This allows us to **correlate training loss with validation perplexity** to analyze:
1. Is the model overfitting? (training loss ↓ but validation PPL ↑)
2. When does the model achieve best validation PPL?
3. Does PPL improvement plateau before training ends?

### Recommended Checkpoint Selection

For a comprehensive analysis, evaluate **5-6 checkpoints** spanning training:

| Checkpoint | Step | Progress | Purpose |
|------------|------|----------|---------|
| `checkpoint-20000` | 20,000 | 13% | Early training |
| `checkpoint-40000` | 40,000 | 25% | Quarter |
| `checkpoint-80000` | 80,000 | 50% | Midpoint |
| `checkpoint-120000` | 120,000 | 75% | Three-quarters |
| `checkpoint-150000` | 150,000 | 95% | Near-final |
| `checkpoint-158760-last` | 158,760 | 100% | Final |

### Implementation Changes Needed

1. **Modify `src/config.py`**: Add checkpoint selection support
2. **Update `evaluate_perplexity.py`**: Add `--checkpoint` CLI argument
3. **Create `evaluate_checkpoints.py`**: Script to run PPL on multiple checkpoints
4. **Create analysis script**: Plot training loss vs validation PPL

---

## Key Implementation Details

### Data Loading (`src/data_loader.py`)

```python
from typing import List, Tuple
import torch
from tqdm import tqdm

def load_tokenized_sequences(
    file_path: str, 
    tokenizer,
    max_samples: int = None,
    show_progress: bool = True
) -> List[torch.Tensor]:
    """
    Load tokenized sequences from file.
    Each line is one song, tokens are space-separated.
    """
    sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if max_samples:
        lines = lines[:max_samples]
    
    iterator = tqdm(lines, desc="Loading sequences") if show_progress else lines
    for line in iterator:
        line = line.strip()
        if line:
            token_ids = tokenizer.encode(line, return_tensors="pt")
            sequences.append(token_ids.squeeze(0))
    
    return sequences

def get_sequence_stats(sequences: List[torch.Tensor]) -> dict:
    """Get statistics about sequence lengths."""
    lengths = [len(seq) for seq in sequences]
    return {
        "num_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "sequences_above_1k": sum(1 for l in lengths if l >= 1024),
        "sequences_above_5k": sum(1 for l in lengths if l >= 5120),
        "sequences_above_10k": sum(1 for l in lengths if l >= 10240),
        "sequences_above_16k": sum(1 for l in lengths if l >= 16384),
    }
```

### Perplexity Calculator (`src/perplexity_calculator.py`)

```python
import math
import torch
from tqdm import tqdm
from typing import List, Optional

def compute_perplexity(
    model,
    sequences: List[torch.Tensor],
    max_context: Optional[int] = None,
    show_progress: bool = True
) -> dict:
    """
    Compute perplexity over token sequences.
    
    Args:
        model: Helibrunna LanguageModel instance
        sequences: List of tokenized sequences
        max_context: Truncate to first N tokens (None = full sequence)
    """
    total_nll = 0.0
    total_tokens = 0
    skipped_sequences = 0
    pad_idx = model.tokenizer.pad_token_id
    
    model.model.eval()
    
    iterator = tqdm(sequences, desc=f"PPL@{max_context or 'full'}") if show_progress else sequences
    
    with torch.no_grad():
        for tokens in iterator:
            # Truncate if specified
            if max_context and len(tokens) > max_context:
                tokens = tokens[:max_context]
            
            # Skip very short sequences
            if len(tokens) < 2:
                skipped_sequences += 1
                continue
            
            tokens = tokens.to(model.device)
            
            # Teacher forcing shift
            inputs = tokens[:-1].unsqueeze(0)
            targets = tokens[1:]
            
            # Get logits
            logits = model.predict(inputs).squeeze(0)
            
            # Compute NLL
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[torch.arange(len(targets), device=model.device), targets]
            
            # Mask padding if present
            if pad_idx is not None:
                mask = targets != pad_idx
                nll = nll[mask]
            
            total_nll += nll.sum().item()
            total_tokens += len(nll)
    
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_nll) if avg_nll < 100 else float('inf')
    
    return {
        "perplexity": ppl,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "num_sequences": len(sequences),
        "skipped_sequences": skipped_sequences,
        "max_context": max_context
    }
```

---

## Evaluation Commands

### Test at All Context Lengths

```bash
# Run all context length evaluations
MODEL_PATH="/scratch1/.../repos/helibrunna/output/xlstm_lmd_512d_2048ctx_12b/run_20260126-0516"
TEST_FILE="/scratch1/.../data/lmd_preprocessed/splits/test.txt"

for ctx in 1024 5120 10240 16384; do
    python evaluate_perplexity.py \
        --model-path "$MODEL_PATH" \
        --data-file "$TEST_FILE" \
        --max-context $ctx \
        --output-dir results/context_${ctx}
done
```

### Optional: Checkpoint Selection Using Validation Set

```bash
VALID_FILE="/scratch1/.../data/lmd_preprocessed/splits/valid.txt"

for ckpt in 50000 100000 150000 158760-last; do
    python evaluate_perplexity.py \
        --model-path "${MODEL_PATH%/*}/checkpoint-${ckpt}" \
        --data-file "$VALID_FILE" \
        --max-context 5120 \
        --output-dir results/checkpoint_selection/${ckpt}
done
```

---

## Expected Output Format

```json
{
    "perplexity": 12.345,
    "avg_nll": 2.514,
    "total_tokens": 15000000,
    "num_sequences": 2500,
    "skipped_sequences": 3,
    "max_context": 10240,
    "model_checkpoint": "checkpoint-158760-last",
    "data_file": "test.txt",
    "timestamp": "2026-02-07T14:30:00Z"
}
```

---

## Memory Management

For very long sequences (>8k tokens):

1. **Batch size = 1**: Process one sequence at a time
2. **Mixed precision**: Consider using fp16 for inference
3. **Gradient disabled**: `torch.no_grad()` context
4. **CUDA cache clearing**: Call `torch.cuda.empty_cache()` periodically if needed

---

## Verification Checklist

- [ ] Model loads correctly with Helibrunna LanguageModel
- [ ] Tokenizer produces correct token IDs for REMIGEN2 format
- [ ] Teacher forcing shift is correct (inputs[:-1], targets[1:])
- [ ] Padding tokens are properly masked
- [ ] NLL computation matches Museformer's approach
- [ ] Results are saved in reproducible JSON format
- [ ] PPL values are reasonable (not inf or nan)

---

## Results Summary Table (To Be Filled)

| Context Length | Perplexity | Total Tokens | Notes |
|----------------|------------|--------------|-------|
| 1,024 | - | - | |
| 5,120 | - | - | |
| 10,240 | - | - | Museformer comparison |
| 16,384 | - | - | xLSTM extrapolation |

---

## Next Steps After Implementation

1. Compare xLSTM perplexity with Museformer baseline (if available)
2. Create visualization of PPL vs context length
3. Document findings for research paper/thesis
4. Proceed to structural analysis (Similarity Error metric)
