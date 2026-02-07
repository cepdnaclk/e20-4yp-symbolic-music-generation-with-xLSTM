# MuseFormer Preprocessing Pipeline

Unified preprocessing pipeline for MuseFormer MIDI data with storage-efficient production mode.

## Overview

This pipeline processes raw MIDI files through 5 stages:
1. **Parsing** - Extract metadata with miditoolkit
2. **Midiminer** - Detect melody tracks (manual step)
3. **Compression** - Compress to 6 instrument categories
4. **MuseScore Normalization** - Normalize MIDI files
5. **Filtering & Pitch Normalization** - Apply MuseFormer paper rules

## Quick Start

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/scripts/museformer-preprocess

# Run full pipeline in dev mode (keeps all intermediates)
python pipeline_main.py --stage all --mode dev

# Run full pipeline in prod mode (minimal storage for 170k files)
python pipeline_main.py --stage all --mode prod
```

## Processing Modes

### Development Mode (`--mode dev`)
- Keeps all intermediate files
- Good for testing and debugging
- Requires ~6x raw dataset size
- Default for datasets < 5000 files

### Production Mode (`--mode prod`)
- Minimal intermediate storage
- Streaming pipeline where possible
- Requires ~1.5x raw dataset size
- **Recommended for 170k files**
- Saves ~40GB vs dev mode

## Pipeline Stages

### Stage 1: Parsing

Parses MIDI files and extracts metadata to manifest.

```bash
python pipeline_main.py --stage 1
```

**What it does**:
- Discovers MIDI files in `00_raw/`
- Parses with miditoolkit
- Extracts: time signatures, tempo, pitch range, note count, etc.
- Creates/updates `manifest.csv`
- **Dev mode**: Copies parsed files to `01_parsed_results/`
- **Prod mode**: Only updates manifest

### Stage 2: Midiminer (Manual)

Detects melody tracks using midiminer.

#### Step 2a: Prepare for Midiminer

```bash
python pipeline_main.py --stage 2-pre
```

This copies parsed files to `02_a_midiminer/` and shows instructions.

#### Manual Step: Run Midiminer

```bash
# Activate midiminer environment
conda activate midiminer

# Run midiminer
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/midi-miner
python track_separate.py \
  --input_dir /path/to/02_a_midiminer \
  --output_json /path/to/02_a_midiminer/program_result.json
```

#### Step 2b: Process Midiminer Results

```bash
python pipeline_main.py --stage 2-post
```

**What it does**:
- Reads `program_result.json`
- Filters files with melody tracks
- Updates manifest with midiminer roles
- **Prod mode**: Deletes `02_a_midiminer/` after processing

### Stage 3: Compress to 6 Tracks

Compresses MIDI files to 6 instrument categories.

```bash
python pipeline_main.py --stage 3
```

**What it does**:
- Uses midiminer roles to identify melody and bass
- Compresses to: square_synth, piano, guitar, string, bass, drum
- Applies deduplication and polyphony capping
- **Dev mode**: Saves to `03_compressed6/`
- **Prod mode**: Streams to next stage

### Stage 4: MuseScore Normalization

Normalizes MIDI files using MuseScore AppImage.

```bash
python pipeline_main.py --stage 4
```

**What it does**:
- Runs MuseScore with custom import options
- Fixes track name inconsistencies ("Piano, piano" → "piano")
- **Dev mode**: Saves to `04_musescore_norm/`
- **Prod mode**: Streams to next stage

**Requirements**:
- MuseScore AppImage at `musescore/MuseScore-Studio-4.6.5.253511702-x86_64.AppImage`
- `xvfb-run` installed (`sudo apt-get install xvfb`)

### Stage 5: Filter and Pitch Normalization

Applies MuseFormer paper filtering rules and pitch normalization.

```bash
python pipeline_main.py --stage 5
```

**What it does**:
- Filters by: time signature, tempo, pitch range, empty bars, etc.
- Detects key using Krumhansl-Kessler profiles
- Transposes to C major / A minor
- **Dev mode**: Saves filtered to `05_filtered/`, final to `06_pitch_normalize/`
- **Prod mode**: Saves only final to `06_pitch_normalize/`

## Directory Structure

```
data/museformer_baseline/
├── 00_raw/                    # Raw MIDI files (required)
├── 01_parsed_results/         # Parsed files (dev mode only)
├── 02_a_midiminer/            # Midiminer input (temp, deleted in prod)
├── 02_b_midiminer_results/    # Files with melody (deleted after stage 3 in prod)
├── 03_compressed6/            # Compressed files (dev mode only)
├── 04_musescore_norm/         # Normalized files (dev mode only)
├── 05_filtered/               # Filtered files (dev mode only)
├── 06_pitch_normalize/        # FINAL OUTPUT (always kept)
└── logs/
    ├── manifest.csv           # Processing manifest (always kept)
    └── pipeline.log           # Pipeline log
```

## Manifest CSV

The `manifest.csv` tracks all files through the pipeline:

| Column | Description |
|--------|-------------|
| `file_id` | Unique file identifier |
| `raw_path` | Original file path |
| `raw_basename` | File basename |
| `stage` | Current processing stage |
| `status` | ok / fail / drop |
| `drop_reason` | Why file was dropped |
| `midiminer_has_melody` | Boolean: has melody track |
| `pitchnorm_mode` | Detected key (major/minor) |
| `pitchnorm_semitones` | Transposition applied |

## Advanced Usage

### Run Specific Stage Range

```bash
# Run stages 3-5 only
python pipeline_main.py --stage 3-5 --mode prod
```

### Custom Base Directory

```bash
python pipeline_main.py --stage all --base-dir /path/to/dataset
```

### Verbose Logging

```bash
python pipeline_main.py --stage all --verbose
```

### Dry Run

```bash
python pipeline_main.py --stage all --dry-run
```

## Configuration

Edit `pipeline_config.py` to customize:
- Base paths
- Stage-specific settings
- MuseScore AppImage location
- Filtering parameters (via `filter_config_v2.py`)

## Troubleshooting

### MuseScore Fails

**Error**: `xvfb-run not found`
```bash
sudo apt-get install xvfb
```

**Error**: `MuseScore AppImage not found`
- Check path in `pipeline_config.py`
- Ensure AppImage has execute permissions: `chmod +x MuseScore-*.AppImage`

### Midiminer Issues

**Error**: `midiminer_json_not_found`
- Make sure you ran midiminer manually (Stage 2 manual step)
- Check that `program_result.json` exists in `02_a_midiminer/`

### Out of Disk Space

- Use `--mode prod` for large datasets
- Delete intermediate directories manually if needed
- Check manifest to see which files passed each stage

## Performance

For 170k MIDI files (~50KB avg):

| Mode | Storage | Time (est) |
|------|---------|------------|
| Dev | ~60GB | ~24 hours |
| Prod | ~20GB | ~20 hours |

**Bottlenecks**:
- Stage 4 (MuseScore): ~1-2s per file
- Stage 5 (Filtering): ~0.5s per file

## Files Created

- `manifest_utils.py` - Manifest management
- `pipeline_config.py` - Configuration
- `stage_01_parsing.py` - Stage 1
- `stage_02_midiminer_helper.py` - Stage 2
- `stage_03_compress6.py` - Stage 3
- `stage_04_musescore_norm.py` - Stage 4
- `stage_05_filter_wrapper.py` - Stage 5
- `pipeline_main.py` - Main orchestrator

## See Also

- [Implementation Plan](../../docs/museformer_pipeline_implementation_plan.md)
- [Filter Configuration](filter_config_v2.py)
- [MuseScore Import Options](../../musescore/midi_import_options.xml)
