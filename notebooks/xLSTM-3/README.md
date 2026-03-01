# xLSTM-3 Music Generation Pipeline

Generates 500 MIDI files with xLSTM (checkpoint-66000) and evaluates generation quality.

## Quick Start

```bash
# From the repo root
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen

# 1. (Optional) Edit MODEL_PATH or RUN_NAME in config.py if using a different run.

# 2. Generate all 500 pieces (~hours on GPU, resumable)
python notebooks/xLSTM-3/generate/run_generation.py

# 3. Evaluate grammar metrics and produce plots
python notebooks/xLSTM-3/evaluate/run_evaluation.py
```

## Repository Structure

```
notebooks/xLSTM-3/
├── xlstm-generation-plan.md       — research plan + bug documentation
├── README.md                      — this file
├── config.py                      — ALL paths and hyperparameters (edit here)
│
├── generate/
│   ├── token_analysis.py          — pure REMIGEN2 grammar analysis functions
│   ├── generator.py               — MusicGenerator (single_shot + chunked, bugs fixed)
│   ├── converter.py               — MIDIConverter (REMIGEN → MIDI via midiprocessor)
│   └── run_generation.py          — CLI: generates all 500 conditions
│
├── evaluate/
│   ├── grammar_eval.py            — aggregation + plots functions
│   └── run_evaluation.py          — CLI: produces metrics and plots
│
└── results/<RUN_NAME>/
    ├── midi/                      — 500 .mid files for teammate
    ├── tokens/                    — raw .txt token files
    ├── generation_log.csv         — one row per piece, all metrics
    └── metrics/
        ├── summary_by_condition.csv
        └── plots/
            ├── grammar_error_rate.png
            ├── success_rate.png
            ├── target_reached_rate.png
            ├── tokens_per_second.png
            └── tokens_per_bar.png
```

## Configuration (`config.py`)

The only file you need to edit:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `run_20260207-1908` | Helibrunna run directory (auto-discovers `-last` checkpoint) |
| `INFERENCE_CONTEXT_LENGTH` | `16_384` | Overrides model context limit at inference |
| `RUN_NAME` | `xlstm_512d_4096ctx_ck66k` | Results sub-directory name |
| `PROMPT` | `s-9 o-0 t-38` | Fixed prompt for all runs |
| `TEMPERATURE` | `0.8` | Sampling temperature |
| `TARGET_TOKENS` | `[1024, 2048, 4096, 8192, 12288]` | Generation targets |
| `PIECES_PER_COND` | `50` | Pieces per (strategy × target) condition |
| `SEED` | `42` | Base seed; piece `i` uses `SEED + i` |

## Strategies

### `single_shot`
One model call with `max_length = target_tokens`. Tests raw extrapolation.

### `chunked`
Bar-aware sliding-window iteration:
1. Snap context left edge to nearest `b-1` (complete-bar alignment)
2. Generate ~400 new tokens
3. Cut at last `b-1` in new tokens (right-edge alignment)
4. Append only complete bars; retry on failure

## Bug Fixes (vs xLSTM-2)

| Bug | Fix |
|---|---|
| Silent exit when no new tokens | Retry with reduced context (up to `MAX_RETRIES`) |
| Silent exit when no b-1 in chunk | Retry with wider chunk (up to `MAX_RETRIES`) |
| `max_iterations` too small | Computed dynamically: `target_tokens // NEW_TOKENS + 20` |
| Token filtering before slicing | Raw output sliced by position first; cleaning only before MIDI write |
| Context left edge starts mid-bar | Snapped to nearest `b-1` via look-back buffer |

## Output: `generation_log.csv` Fields

| Column | Description |
|---|---|
| `strategy` | `single_shot` or `chunked` |
| `target_tokens` | Requested token count |
| `piece_id` | 1–50 piece index |
| `actual_tokens` | Tokens actually produced |
| `target_reached` | Whether `actual_tokens >= target_tokens` |
| `num_bars` | Count of `b-1` tokens |
| `grammar_error_rate` | `num_errors / total_tokens` |
| `incomplete_triplets` | `p-X` not followed by `d-X v-X` |
| `orphan_tokens` | Standalone `d-X` or `v-X` |
| `success` | Decoded to valid MIDI |
| `generation_time_s` | Wall-clock seconds |
| `tokens_per_second` | Throughput |
| `num_chunks` | Chunked only: iterations used |

## Resuming an Interrupted Run

The generation script skips any MIDI file that already exists on disk.
Simply re-run `run_generation.py` after an interruption — it will pick up where it left off.

## Model Details

- **Architecture**: xLSTM, 512-dim, 12 blocks, context 4096
- **Dataset**: Lakh MIDI Dataset (LMD), REMIGEN2 encoding
- **Best checkpoint**: `checkpoint-66000` (PPL = 1.64 @ 4096 ctx)
- **Loaded as**: `checkpoint-66000-last` (Helibrunna `-last` naming convention)
