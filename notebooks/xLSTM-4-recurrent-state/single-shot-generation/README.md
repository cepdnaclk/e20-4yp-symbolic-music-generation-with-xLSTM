# xLSTM-4 Single-Shot Music Generation

## Overview

This pipeline generates MIDI files using the **xLSTMGenerator** with O(N) recurrent inference. Unlike xLSTM-3, which used Helibrunna's O(N²) parallel formulation and required chunking, this approach uses pure single-shot generation at all lengths.

## Key Improvements Over xLSTM-3

1. **10× Faster** - O(N) linear scaling vs O(N²) quadratic slowdown
2. **No Repetition** - Recurrent state eliminates Helibrunna's parallel mode bugs
3. **Simpler** - No chunked strategy, no bar-alignment edge cases
4. **Scalable** - Can generate 12,288 tokens in ~77 seconds
5. **Multi-Config** - Easy experimentation with different parameter sets

## File Structure

```
single-shot-generation/
├── configs/                    # Configuration directory
│   ├── default.py              # Full run: 50 pieces × 5 lengths = 250 pieces
│   ├── quick_test.py           # Quick test: 5 pieces × 5 lengths = 25 pieces
│   └── [your_experiment.py]    # Custom configs you create
│
├── token_analysis.py           # Grammar checking (REMIGEN2 validation)
├── converter.py                # Token → MIDI conversion
├── run_generation.py           # Main orchestrator (accepts --config argument)
├── smoke_test.py               # Ultra-quick test (1 piece at 1024 tokens)
│
├── results/                    # Auto-generated results (one folder per config)
│   ├── default/                # From configs/default.py
│   ├── quick_test/             # From configs/quick_test.py
│   └── smoke_test/             # From smoke_test.py
│
└── README.md                   # This file
```

## Usage

### Quick Start: Test with 25 pieces (~15 minutes)

```bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate xlstm
cd notebooks/xLSTM-4-recurrent-state/single-shot-generation

# Option 1: Run with live output + save logs (RECOMMENDED)
python run_generation.py --config configs/quick_test.py 2>&1 | tee results/quick_test/generation_run.log

# Option 2: Run without saving logs
python run_generation.py --config configs/quick_test.py
```

**What the `tee` command does:**
- Shows real-time progress in your terminal (so you can monitor the run)
- Saves complete console output to `results/quick_test/generation_run.log`
- Captures both stdout and stderr (`2>&1` redirects stderr to stdout)

### Full Run: 250 pieces (~2.4 hours)

```bash
# With logging (recommended for long runs)
python run_generation.py --config configs/default.py 2>&1 | tee results/default/generation_run.log

# Without logging
python run_generation.py --config configs/default.py

# Or use default config (same as above)
python run_generation.py 2>&1 | tee results/default/generation_run.log
```

### Ultra-Quick Smoke Test (1 piece)

```bash
python smoke_test.py
```

## Multi-Config System

### How It Works

1. **Each config file creates its own results folder**
   - `configs/default.py` → `results/default/`
   - `configs/quick_test.py` → `results/quick_test/`
   - `configs/my_experiment.py` → `results/my_experiment/`

2. **RUN_NAME is auto-inferred from config filename**
   - No need to manually set RUN_NAME
   - Prevents accidental overwrites

3. **All configs use the same run_generation.py pipeline**
   - Consistent evaluation metrics across experiments

### Creating Custom Configs

Copy an existing config and modify parameters:

```bash
cp configs/quick_test.py configs/temperature_sweep.py
```

Edit `configs/temperature_sweep.py`:

```python
# Example: Temperature experiment
TEMPERATURE = 1.2  # Higher temperature
TARGET_TOKENS = [4096]  # Only test one length
PIECES_PER_COND = 20    # 20 pieces
SEED = 99               # Different seed
```

Run it:

```bash
python run_generation.py --config configs/temperature_sweep.py
```

Results saved to `results/temperature_sweep/`.

### Available Configs

| Config | Description | Pieces | Estimated Time |
|--------|-------------|--------|----------------|
| **configs/default.py** | Full experiment | 250 (50 × 5) | ~2.4 hours |
| **configs/quick_test.py** | Quick validation | 25 (5 × 5) | ~15 minutes |
| **smoke_test.py** | Ultra-quick sanity check | 1 | ~10 seconds |

## Configuration Parameters

All configs support these parameters:

```python
# Model & Paths
MODEL_PATH              # Path to model checkpoint
CHECKPOINT_NAME         # Specific checkpoint to load
MIDIPROCESSOR_PATH      # Path to MidiProcessor library
INFERENCE_CONTEXT_LENGTH  # Max context (16,384 allows extrapolation)

# Generation Hyperparameters
PROMPT                  # Fixed prompt (e.g., "s-9 o-0 t-38")
TEMPERATURE             # Sampling temperature (0.8 = balanced)
TARGET_TOKENS           # List of token lengths [1024, 2048, ...]
PIECES_PER_COND         # How many pieces per target length
SEED                    # Random seed base

# Output (auto-configured)
RUN_NAME                # Auto-inferred from config filename
RESULTS_DIR             # Auto: results/{RUN_NAME}/
MIDI_DIR                # Auto: results/{RUN_NAME}/midi/
TOKEN_DIR               # Auto: results/{RUN_NAME}/tokens/
LOG_CSV                 # Auto: results/{RUN_NAME}/generation_metrics.csv
```

## Output Structure

After running `python run_generation.py --config configs/quick_test.py 2>&1 | tee results/quick_test/generation_run.log`:

```
results/quick_test/
├── midi/
│   ├── single_shot_01024tok_001.mid
│   ├── single_shot_01024tok_002.mid
│   ├── ...
│   └── single_shot_12288tok_005.mid   (25 total files)
│
├── tokens/
│   ├── single_shot_01024tok_001.txt   (raw REMIGEN2 tokens)
│   └── ...
│
├── plots/                              (auto-generated visualizations - PDF only)
│   ├── performance/
│   │   ├── tokens_per_second.pdf      (generation speed)
│   │   └── generation_time.pdf        (time scaling)
│   ├── quality/
│   │   └── grammar_error_rate.pdf     (token quality)
│   └── musical/
│       ├── num_bars.pdf               (musical bars)
│       ├── num_notes.pdf              (note count)
│       └── tokens_per_bar.pdf         (bar density)
│
├── generation_metrics.csv             (structured metrics: CSV format)
└── generation_run.log                 (console output: text log)
```

## CSV Metrics

Each row in `generation_log.csv` contains:

| Metric Category | Fields |
|----------------|--------|
| **Generation** | `actual_tokens`, `target_reached`, `generation_time_s`, `tokens_per_second` |
| **Grammar** | `grammar_error_rate`, `incomplete_triplets`, `orphan_tokens` |
| **Musical** | `num_bars`, `num_notes`, `num_instruments`, `tokens_per_bar_mean/std` |
| **Status** | `success`, `repair_needed`, `ends_with_bar` |
| **Identity** | `strategy`, `target_tokens`, `piece_id`, `filename`, `seed` |

## Automatic Visualization

**Publication-quality plots are automatically generated** after each run! All plots are saved as PDFs organized by category.

### Generated Plots (6 total)

**Performance** (`plots/performance/`)
1. **tokens_per_second.pdf** - Generation speed (should be constant ~155 tok/s for O(N) scaling)
2. **generation_time.pdf** - Time vs length (should be linear for O(N) scaling)

**Quality** (`plots/quality/`)
3. **grammar_error_rate.pdf** - Token grammar quality across lengths

**Musical** (`plots/musical/`)
4. **num_bars.pdf** - Musical bars generated per length
5. **num_notes.pdf** - Note count per length
6. **tokens_per_bar.pdf** - Bar density consistency

### Manual Plotting

You can also regenerate plots manually:

```bash
# Regenerate plots from existing CSV
python plot_results.py --csv results/quick_test/generation_metrics.csv

# Specify custom output directory
python plot_results.py --csv results/quick_test/generation_metrics.csv --output-dir custom_plots/
```

**Note**: Requires `matplotlib` and `seaborn`. If not installed, generation continues but skips plotting.

## Resumability

The pipeline is fully resumable:

```bash
# Start generation
python run_generation.py --config configs/default.py

# <Ctrl+C to interrupt>

# Resume from where it stopped (skips existing MIDI files)
python run_generation.py --config configs/default.py
```

## Performance Benchmarks

From validation tests ([../benchmarking/](../benchmarking/)):

| Length | Time | Speed | Speedup vs Helibrunna |
|--------|------|-------|----------------------|
| 1,000 | 6.5s | 155 tok/s | 2.6× |
| 2,000 | 12.8s | 156 tok/s | 4.2× |
| 4,000 | 25.7s | 156 tok/s | - |
| 8,000 | 51.4s | 156 tok/s | - |
| 12,000 | 77.1s | 156 tok/s | - |

**Linear O(N) scaling confirmed!**

## Model Details

- **Model**: `xlstm_lmd_512d_4096ctx_12b`
- **Checkpoint**: `checkpoint-66000-last` (best PPL: 1.643)
- **Training context**: 4,096 tokens
- **Inference context**: 16,384 tokens (allows 3× extrapolation)
- **Tokenization**: REMIGEN2 (Lakh MIDI Dataset encoding)
- **Dataset**: Lakh MIDI Dataset (LMD)

## Architecture Flow

```
User runs: python run_generation.py --config configs/quick_test.py
                ↓
1. load_config("configs/quick_test.py")
   → RUN_NAME = "quick_test" (auto-inferred)
   → TARGET_TOKENS = [1024, 2048, 4096, 8192, 12288]
   → PIECES_PER_COND = 5
                ↓
2. Initialize xLSTMGenerator (from ../inference.py)
   → Loads checkpoint-66000-last
   → O(N) recurrent formulation
                ↓
3. For each (target_tokens, piece_id) combination:

   a. generator.generate(prompt, temperature, max_length)
      → Returns: "s-9 o-0 t-38 i-0 p-60 d-2 v-2 ..."

   b. Parse text → token list

   c. analyse_tokens() → grammar metrics

   d. Save raw tokens → .txt file

   e. MIDIConverter.tokens_to_midi() → .mid file

   f. Log all metrics → CSV
                ↓
4. Results saved to results/quick_test/
```

## Troubleshooting

**Import errors**:
```bash
conda activate xlstm
```

**CUDA out of memory**:
- Reduce `PIECES_PER_COND` in your config
- Or generate one target length at a time

**Invalid MIDI files**:
- Check `generation_log.csv` for rows with `success=False`
- These indicate token grammar errors that couldn't be auto-repaired

**Config not found**:
```bash
# Make sure path is relative to run_generation.py location
python run_generation.py --config configs/my_config.py  # ✓ Correct
python run_generation.py --config my_config.py          # ✗ Wrong
```

## Examples

### Example 1: Temperature Sweep

Create `configs/temp_sweep_high.py`:

```python
TEMPERATURE = 1.5  # Very creative
TARGET_TOKENS = [4096]  # Mid-length only
PIECES_PER_COND = 10
SEED = 123
# (copy rest from default.py)
```

Run:
```bash
python run_generation.py --config configs/temp_sweep_high.py
```

### Example 2: Extrapolation Test

Create `configs/extrapolation.py`:

```python
TEMPERATURE = 0.8
TARGET_TOKENS = [4096, 8192, 12288, 16384]  # Beyond training context
PIECES_PER_COND = 20
SEED = 42
# (copy rest from default.py)
```

Run:
```bash
python run_generation.py --config configs/extrapolation.py
```

Results in `results/extrapolation/`.

## Citation

If you use this pipeline, please cite the xLSTM paper:

```
Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., ... & Hochreiter, S. (2024).
xLSTM: Extended long short-term memory. arXiv preprint arXiv:2405.04517.
```
