# xLSTM Music Generation – Research Plan (xLSTM-3)

> **Goal**: Generate a systematic set of MIDI files with the trained xLSTM model, fix known
> generation bugs, and evaluate the *generation process itself* (grammar quality, success rate,
> speed). Musical quality and long-range structure analysis is handled by a teammate.

---

## Division of Work

| Task | Owner |
|---|---|
| Fix generation bugs, implement pipeline | **You** |
| Generate MIDI files (clean, well-named) | **You** |
| Grammar / token quality metrics | **You** |
| Generation speed & cost metrics | **You** |
| Musical quality metrics (key stability, pitch, rhythm …) | Teammate |
| Long-range structure analysis (SSM, Similarity Error) | Teammate |

---

## REMIGEN2 Token Format Reference

Confirmed from `repos/MidiProcessor/midiprocessor/const.py` and `enc_remigen2_utils.py`:

| Token | Abbr | Meaning | Emitted |
|---|---|---|---|
| `s-X` | `TS_ABBR` | **Time Signature** (not "section"!) | Start of **every** bar |
| `o-X` | `POS_ABBR` | Onset/position within bar | Per note group |
| `t-X` | `TEMPO_ABBR` | Tempo class | Start of **every** bar |
| `i-X` | `INST_ABBR` | Instrument (MIDI program) | Per note group |
| `p-X` | `PITCH_ABBR` | Pitch (MIDI note number) | Per note |
| `d-X` | `DURATION_ABBR` | Duration class | Per note |
| `v-X` | `VELOCITY_ABBR` | Velocity class | Per note |
| `b-1` | `BAR_ABBR` | Bar boundary marker | **End** of each bar |

A complete bar always looks like:
```
s-9 t-38 o-0 i-X p-X d-X v-X o-12 i-X p-X d-X v-X ... b-1
 ↑ time sig  ↑ tempo  ↑ note groups                    ↑ bar end
```

The encoder emits `s-X t-X` at the start of **every** bar, not just when they change.
This means every bar is self-contained — the model always knows the meter and tempo.

> **Note**: `b-1` marks the **end** of a bar. The next bar starts immediately after with `s-X t-X`.
> The grammar check for a "complete bar" is: `s-X t-X ... b-1` (opens with time sig + tempo,
> closes with bar marker).

---

## Model & Checkpoint Selection

**Model**: `xlstm_lmd_512d_4096ctx_12b` — 512-dim, 12 blocks, trained on LMD with 4096-token context.

**Best checkpoint**: `checkpoint-66000` — confirmed by perplexity evaluation
(`xlstm-evaluation/evaluation-pipeline/results/xlstm_lmd_512d_4096ctx_12b_20260210_182112/summary.md`).

| Step | PPL @ 4096 ctx (training ctx) | Notes |
|---|---|---|
| 34,000 | 1.741 | |
| **66,000** | **1.643** | **← Best — used for all generation** |
| 98,000 | 1.647 | slight overfit begins |
| 130,000 | 1.744 | overfitting |
| 158,760 | 1.824 | clear overfit (final checkpoint) |

**Extrapolation behaviour** (test PPL at checkpoint-66000 across context lengths):

| Context length | PPL | vs Training ctx |
|---|---|---|
| 1024 | 1.868 | +0.24 |
| 2048 | 1.715 | +0.08 |
| 3072 | 1.658 | +0.03 |
| 4096 ← training ctx | 1.630 | baseline |
| **5120** | **1.622** | **−0.008 — better!** |
| 10240 | 1.976 | +0.35 |

> Perplexity actually improves slightly at 5120 tokens (25% beyond training context) before
> degrading at 10240. This suggests the model extrapolates gracefully up to ~5k tokens.
> Our generation targets (1024–12288) deliberately span this extrapolation curve.

**Checkpoint loading**: Helibrunna auto-loads the checkpoint ending in `-last`. The folder has
been renamed to `checkpoint-66000-last`. `MODEL_PATH` is set to the run directory and
Helibrunna discovers the checkpoint automatically.

---

## Bug Investigation & Fixes

Thorough code review of `xLSTM-2/xlstm_music_generation.py` and
`repos/helibrunna/source/languagemodel.py` found **5 bugs** and **1 configuration issue**.

### Implementation Checklist

- [ ] **Config**: Load model with `INFERENCE_CONTEXT_LENGTH = 16_384`
- [ ] **Bug 1**: Replace hard `break` (no new tokens) with retry logic
- [ ] **Bug 2**: Replace hard `break` (no b-1 in chunk) with widen-and-retry logic
- [ ] **Bug 3**: Compute `max_iterations` dynamically from `target_tokens`
- [ ] **Bug 4**: Move token filtering to after slicing; return raw tokens from `generate()`
- [ ] **Bug 5**: Snap context left edge to nearest `b-1` using look-back buffer

---

### Configuration — Context-Length Must Be Set at Load Time

Helibrunna's generate loop exits when `inputs.shape[1] >= self.config.context_length`.
`MusicGenerator.__init__` already supports overriding this via `config_overrides`.

**Action**: Always load with `INFERENCE_CONTEXT_LENGTH = 16_384`. This keeps the wall far
beyond anything we generate.

---

### BUG 1 — Silent Early Exit: No New Tokens Generated

**File**: `xLSTM-2/xlstm_music_generation.py`, lines 172–175

```python
# CURRENT (buggy):
if len(chunk_tokens) <= len(context_tokens):
    break   # aborts the ENTIRE run from one bad chunk

# FIX:
# Retry up to MAX_RETRIES times, reducing CONTEXT_SIZE by 20% each attempt.
# Only break after all retries exhausted.
```

---

### BUG 2 — Silent Early Exit: No Complete Bar in Chunk

**File**: `xLSTM-2/xlstm_music_generation.py`, lines 182–185

```python
# CURRENT (buggy):
if last_bar_idx is None:
    break   # aborts the entire run

# FIX:
# Retry up to MAX_RETRIES times, increasing NEW_TOKENS by 50% each attempt.
# Only break after all retries exhausted.
```

---

### BUG 3 — `max_iterations` Too Conservative

**File**: `xLSTM-2/xlstm_music_generation.py`, line 111

```python
# CURRENT (buggy):
max_iterations = 50   # hardcoded, insufficient for large targets

# FIX:
max_iterations = max(100, int(target_tokens / NEW_TOKENS) + 20)
```

---

### BUG 4 — Token Filtering Before Slicing

**File**: `xLSTM-2/xlstm_music_generation.py`, inside `generate()`

```python
# CURRENT (buggy):
# Filtering happens inside generate(), BEFORE generate_long() slices for new tokens.
# If any tokens in the context portion are filtered out, the slice index is wrong.

# FIX:
# generate() returns raw tokens (no filtering).
# generate_long() slices first → then filters only the new-token portion.
# clean_tokens() is called separately, only before MIDI decoding.
```

---

### BUG 5 — Context Window Left Edge Starts Mid-Bar

**File**: `xLSTM-2/xlstm_music_generation.py`, lines 148–154

```python
# CURRENT (buggy):
context_tokens = all_tokens[-CONTEXT_SIZE:]   # left edge lands mid-bar

# FIX: snap left edge to nearest b-1 using a look-back buffer
candidates = all_tokens[-(CONTEXT_SIZE + CONTEXT_BUFFER):]
context_tokens = candidates   # fallback if no b-1 found in buffer
for i, tok in enumerate(candidates[:CONTEXT_BUFFER]):
    if tok == "b-1":
        context_tokens = candidates[i + 1:]   # start right after bar marker
        break
```

Since every bar starts with `s-X t-X`, snapping after a `b-1` gives the model a complete,
self-contained bar header at the start of its context — exactly as during training.

**How it works (both edges bar-aligned):**
- **Right edge**: always at the last `b-1` of `all_tokens` (guaranteed by append logic)
- **Left edge**: snapped to the first `b-1` within the look-back buffer zone

---

### Summary of All Issues

| # | Issue | Severity | Fixed In |
|---|---|---|---|
| Config | `context_length` not set at load time | Config fix | `config.py` |
| 1 | "No new tokens" → immediate `break` | **High** | `generator.py` |
| 2 | "No b-1 in chunk" → immediate `break` | **High** | `generator.py` |
| 3 | `max_iterations` hardcoded at 50 | Medium | `generator.py` |
| 4 | Token filtering before slicing | Medium | `generator.py` + `converter.py` |
| 5 | Context left edge starts mid-bar | **High** | `generator.py` |

---

## What We Generate

### Model
`xlstm_lmd_512d_4096ctx_12b` (training context: 4096 tokens), loaded with
`INFERENCE_CONTEXT_LENGTH = 16_384`.

### Token-Based Targets

Token count is the **primary target** — it is what the model operates on directly.
Bar count is an *output metric*, measured after generation and logged in `generation_log.csv`.

| Target | Relationship to training context (4096) |
|---|---|
| `1024` | 0.25× — well within |
| `2048` | 0.5× — within |
| `4096` | 1× — at training boundary |
| `8192` | 2× — extrapolation |
| `12288` | 3× — strong extrapolation (4k safety margin below 16k limit) |

### Strategies

#### `single_shot`
One model call: `generate(prompt, max_length=target_tokens)`. The full sequence is generated
in one forward pass. Tests raw extrapolation capability.

> Memory note: At 12288 tokens, single-shot is memory-intensive. If it OOMs, that result
> is documented — it is itself informative.

#### `chunked` — Bar-Aware Sliding Window

```
Loop until len(all_tokens) >= target_tokens:
  1. context = all_tokens bar-aligned left edge, ~1500 tokens (Bug 5 fix)
  2. generate ~400 new tokens beyond context
  3. cut at last b-1 in new tokens (right edge, always bar-aligned)
  4. append only complete new bars to all_tokens
  5. if stuck: retry with smaller context (Bug 1) or wider chunk (Bug 2)
```

Both edges of the context window are always bar-aligned, so model always sees complete
bars — same structure as training data.

### Conditions

| Factor | Values |
|---|---|
| **Strategy** | `single_shot`, `chunked` |
| **Target tokens** | `1024`, `2048`, `4096`, `8192`, `12288` |
| **Temperature** | `0.8` (fixed) |
| **Pieces per condition** | `50` |

**Total: 2 × 5 × 50 = 500 MIDI files.**

### Prompt (Fixed for All Runs)
```
s-9 o-0 t-38
```

### File Naming Convention
```
{strategy}_{tokens:05d}tok_{piece_id:03d}.mid

Examples:
  single_shot_01024tok_001.mid
  single_shot_04096tok_023.mid
  chunked_08192tok_047.mid
  chunked_12288tok_050.mid
```

All MIDI files → `results/<run_name>/midi/` (flat, for teammate)
Raw token files → `results/<run_name>/tokens/`
Per-piece metadata → `results/<run_name>/generation_log.csv`

---

## What We Evaluate (Generation Side Only)

### Grammar / Token Quality

Computed from raw REMIGEN token stream — **before** MIDI conversion.

| Metric | Definition |
|---|---|
| `grammar_error_rate` | `num_errors / total_tokens` |
| `incomplete_triplets` | `p-` not followed by `d-` then `v-` |
| `orphan_tokens` | Standalone `d-` or `v-` without preceding `p-` |
| `tokens_per_bar_mean` | Mean tokens per completed bar |
| `tokens_per_bar_std` | Std of tokens per bar |
| `actual_bars` | Count of `b-1` tokens in output |
| `repair_needed` | Did `clean_tokens()` run before MIDI decode? |
| `success` | Did the piece decode to valid MIDI? |

**Key plot**: `grammar_error_rate` and `success_rate` vs `target_tokens` — shows where xLSTM
extrapolation degrades and quantifies single-shot vs chunked tradeoff.

### Generation Cost

| Metric | Definition |
|---|---|
| `generation_time_s` | Wall-clock seconds |
| `actual_tokens` | Tokens actually produced |
| `target_reached` | `actual_tokens >= target_tokens` (False = stopped early) |
| `tokens_per_second` | Throughput |
| `num_chunks` | Chunked only: iterations used |

---

## Repository Structure

```
notebooks/xLSTM-3/
├── xlstm-generation-plan.md        ← this file
├── README.md                       ← reproduction guide
├── config.py                       ← all paths & hyperparameters
│
├── generate/
│   ├── __init__.py
│   ├── generator.py                ← MusicGenerator (single_shot + chunked, all bugs fixed)
│   ├── converter.py                ← MIDIConverter (decode + clean, separate from generation)
│   ├── token_analysis.py           ← grammar_error_rate(), analyse_tokens() — pure functions
│   └── run_generation.py           ← CLI: generates all 500 conditions
│
├── evaluate/
│   ├── __init__.py
│   ├── grammar_eval.py             ← reads generation_log.csv → stats + plots
│   └── run_evaluation.py           ← CLI: produces metrics CSVs and plots
│
└── results/
    └── <run_name>/
        ├── midi/                   ← 500 MIDIs for teammate (flat directory)
        ├── tokens/                 ← raw token .txt files
        ├── generation_log.csv      ← one row per piece, all metrics
        └── metrics/                ← aggregated CSVs + plots
```

---

## `config.py` Template

```python
# ─── Model ───────────────────────────────────────────────────────────────────
# Pass the RUN directory — Helibrunna auto-discovers the checkpoint ending in "-last"
# Currently: checkpoint-66000-last (best validation checkpoint)
MODEL_PATH               = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/xlstm_lmd_512d_4096ctx_12b/run_20260207-1908"
INFERENCE_CONTEXT_LENGTH = 16_384   # keeps Helibrunna context wall far away

# ─── Run ─────────────────────────────────────────────────────────────────────
RUN_NAME         = "xlstm_512d_4096ctx"
PROMPT           = "s-9 o-0 t-38"
TEMPERATURES     = [0.8]
TARGET_TOKENS    = [1024, 2048, 4096, 8192, 12288]
STRATEGIES       = ["single_shot", "chunked"]
PIECES_PER_COND  = 50
SEED             = 42               # for reproducibility (seed + piece_id per piece)

# ─── Chunked parameters ───────────────────────────────────────────────────────
CONTEXT_TOKENS   = 1500   # target window size (actual may vary slightly due to bar alignment)
CONTEXT_BUFFER   = 300    # look-back buffer to find b-1 for left-edge alignment
NEW_TOKENS       = 400    # tokens to generate per chunk
MAX_RETRIES      = 3      # retries before giving up on a stuck chunk
```

---

## CLI Workflow

```bash
# 1. Update config.py (set MODEL_PATH)

# 2. Generate all 500 pieces
python notebooks/xLSTM-3/generate/run_generation.py

# 3. Evaluate grammar metrics and produce plots
python notebooks/xLSTM-3/evaluate/run_evaluation.py

# 4. Hand off results/<run_name>/midi/ and generation_log.csv to teammate
```

Both scripts are **resumable**: if interrupted, they skip files that already exist.

---

## Key Research Questions

| Question | Metric |
|---|---|
| Does grammar degrade with length? | `grammar_error_rate` vs `target_tokens` |
| Where does generation fail entirely? | `success_rate` vs `target_tokens` |
| Does chunked preserve grammar vs single-shot? | Compare per-strategy error rates |
| Does the model reach the requested length? | `target_reached` rate |
| Which strategy is faster? | `tokens_per_second` |

---

## Implementation Progress

### Phase 1 — Planning ✅
- [x] Code review of xLSTM-2 and Helibrunna
- [x] Identify all 5 bugs + config issue
- [x] Confirm REMIGEN2 token format from source
- [x] Agree on token-based targets (1024–12288)
- [x] Agree on bar-aligned context for chunked strategy
- [x] This plan finalised and reviewed

### Phase 2 — Implementation
- [ ] `config.py`
- [ ] `generate/token_analysis.py`
- [ ] `generate/generator.py` (all 5 bug fixes)
- [ ] `generate/converter.py`
- [ ] `generate/run_generation.py`
- [ ] `evaluate/grammar_eval.py`
- [ ] `evaluate/run_evaluation.py`
- [ ] `README.md`

### Phase 3 — Verification
- [ ] Single-shot smoke test (1024 tokens, 1 piece)
- [ ] Chunked smoke test (2048 tokens, 1 piece) — check bar-aligned joins
- [ ] Full 500-piece generation run
- [ ] Evaluation run → `generation_log.csv` + plots

---

## Out of Scope (Teammate's Job)

- Musical quality metrics (key stability, chord diversity, pitch range, note density)
- Self-Similarity Matrix (SSM) analysis
- Similarity Error (SE) evaluation
- Listening / subjective evaluations




