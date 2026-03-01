# Single-Shot Music Generation Plan

> **Goal**: Evaluate the new $O(N)$ `xLSTMGenerator` on the task of scaling up to full-length musical pieces (up to 12,288 tokens) in a pure single-shot manner. Evaluate generation process quality, speed, and success metrics using the exact same evaluation metrics as prior experiments.

---

## 1. Context & Motivation
Previously in `xLSTM-3`, generating long sequences required a complex "chunked" sliding-window strategy because the underlying Helibrunna generator used $O(N^2)$ scaling, which caused Out-Of-Memory (OOM) crashes and took 30+ minutes for a single piece. 

Now, with our custom `inference.py` (`xLSTMGenerator`) using the recurrent formulation, sequence generation holds a constant memory footprint and scales linearly in time $O(N)$. Therefore, **chunked generation is retired**. We will evaluate purely **single-shot generation** at all lengths.

---

## 2. Model & Checkpoint
- **Model:** `xlstm_lmd_512d_4096ctx_12b`
- **Checkpoint:** `checkpoint-66000-last` (Best PPL)
- **Context Override:** Loaded with `INFERENCE_CONTEXT_LENGTH = 16_384` to prevent arbitrary length warnings.

---

## 3. What We Are Generating

### Token-Based Targets
We will evaluate pure extrapolation capabilities: 
| Target Tokens | Relationship to training context (4096) | Expected Generation Time (per piece) |
|---|---|---|
| `1024` | 0.25× | ~6.5 seconds |
| `2048` | 0.5× | ~13 seconds |
| `4096` | 1× | ~25 seconds |
| `8192` | 2× | ~51 seconds |
| `12288` | 3× | ~77 seconds |

### Conditions
- **Strategy:** `single_shot` ONLY.
- **Target Tokens:** `[1024, 2048, 4096, 8192, 12288]`
- **Temperature:** `0.8` (Fixed)
- **Pieces per condition:** `50`
- **Total Pieces:** 1 × 5 × 50 = **250 MIDI files.**
- **Prompt:** `s-9 o-0 t-38`
- **Total Estimated Generation Time:** ~2.4 Hours.

### File Naming Convention
```text
single_shot_{tokens:05d}tok_{piece_id:03d}.mid
# Example: single_shot_12288tok_047.mid
```

---

## 4. Evaluation Metrics (Reused from xLSTM-3)

We will reuse the pure-function evaluation scripts from `xLSTM-3` to guarantee comparable results.
- **Grammar / Token Quality (`token_analysis.py`):**
  - `grammar_error_rate`, `incomplete_triplets`, `orphan_tokens`, `actual_bars`.
- **Generation Cost:**
  - `generation_time_s`, `tokens_per_second`, `success` rate.

---

## 5. Architectural Implementation Plan

We do *not* want to mutate `xLSTM-3`. Instead, we will construct a clean pipeline in `single-shot-generation/`:

```text
notebooks/xLSTM-4-recurrent-state/single-shot-generation/
├── single-shot-generation-plan.md  ← This file
├── config.py                       ← Hardcoded paths, lengths, and hyperparameters
└── run_generation.py               ← CLI loop to generate the 250 files and log to CSV
```

### Script Execution Flow (`run_generation.py`)
1. Iterates over `TARGET_TOKENS` and runs exactly 50 iterations per target.
2. Initializes our recurrent `xLSTMGenerator` from `inference.py`.
3. Calls `model.generate()` to generate the raw text tokens.
4. Uses `token_analysis.py` (imported from `xLSTM-3/generate`) to calculate grammar statistics.
5. Uses `converter.py` (imported from `xLSTM-3/generate`) to decode the text into a `.mid` file.
6. Writes performance details, time, and statistics to `generation_log.csv`. 

### Evaluating Extrapolation & Grammar
Once generation is done, we can simply point `xLSTM-3/evaluate/run_evaluation.py` to the generated `generation_log.csv` OR adapt a quick evaluation script here to output our graphs mapping `tokens_per_second` and `grammar_error_rate` against target tokens.

---

## 6. Next Steps for Implementation
1. Copy the layout parameters into `single-shot-generation/config.py`.
2. Write `run_generation.py` allowing it to import the required files from `xLSTM-3` and `inference.py`.
3. Smoke test 1 short generation condition (1024 tokens) and 1 long condition (8192 tokens) to ensure logging and MIDI conversion operates without errors.
4. Launch the full 250-file bulk generation.
