# MuseFormer MIDI Filtering and Pitch Normalization Pipeline

A robust Python pipeline for filtering and normalizing MIDI files based on the MuseFormer paper's filtering rules (Table 4). This implementation includes improvements for configurability, error handling, and detailed tracking.

## Features

- **Comprehensive Filtering**: Implements all filtering rules from the MuseFormer paper
- **Duplicate Detection**: Identifies and removes duplicate MIDI files based on musical content
- **Pitch Normalization**: Automatically detects key and transposes to C major or A minor
- **Manifest Tracking**: Maintains detailed CSV manifest of all processing stages
- **Configurable**: Easy-to-modify configuration file for custom filtering rules
- **Progress Reporting**: Detailed console output and JSON summary of pipeline execution

## Installation

### Requirements

```bash
pip install miditoolkit pandas --break-system-packages
```

### Files

- `filter_and_normalize.py` - Main pipeline script
- `filter_config.py` - Configuration file with all filtering parameters
- `README.md` - This documentation file

## Quick Start

1. **Configure paths** in `filter_config.py`:
```python
INPUT_NORMALIZED_DIR = Path("data/musescore_normalized")
OUTPUT_FILTERED_DIR = Path("data/filtered")
OUTPUT_PITCH_NORM_DIR = Path("data/pitch_normalized")
MANIFEST_PATH = Path("data/manifest.csv")
```

2. **Run the pipeline**:
```bash
python filter_and_normalize.py
```

The pipeline will:
1. Extract statistics from all MIDI files
2. Detect and remove duplicates
3. Apply filtering rules
4. Copy filtered files to output directory
5. Apply pitch normalization
6. Update manifest with results

## Filtering Rules (MuseFormer Table 4)

### Time Signature
- **Rule**: Only keep 4/4 time signature
- **Config**: `ALLOWED_TIME_SIGNATURES = ["4/4"]`

### Track Requirements
- **Rule**: At least 2 non-empty tracks
- **Config**: `MIN_NONEMPTY_TRACKS = 2`
- **Rule**: Must have "square_synth" track (melody)
- **Config**: `REQUIRED_TRACK_NAME = "square_synth"`

### Tempo
- **Rule**: Tempo between 24 and 200 BPM
- **Config**: `TEMPO_MIN = 24`, `TEMPO_MAX = 200`

### Pitch Range
- **Rule**: Pitches between 21 (A0) and 108 (C8)
- **Config**: `PITCH_MIN = 21`, `PITCH_MAX = 108`

### Note Duration
- **Rule**: Maximum note duration of 16 beats (4 bars)
- **Config**: `MAX_NOTE_DURATION_BEATS = 16.0`

### Empty Bars
- **Rule**: Remove files with 4+ consecutive empty bars
- **Config**: `MAX_EMPTY_BARS_ALLOWED = 3`
- **Methods**:
  - `"onset"`: Bar is empty if no notes START in it
  - `"sounding"`: Bar is empty if no notes are SOUNDING during it (stricter)

### Degenerate Content
- **Rule**: Remove files where all notes have same pitch or duration
- **Config**: `DROP_DEGENERATE_CONTENT = True`

### Duplicates
- **Rule**: Remove files with identical duration, bar count, note count, onsets, and instruments
- **Detection**: Automatic based on musical signature

## Pitch Normalization

The pipeline normalizes all pieces to either C major or A minor using:

1. **Key Detection**: Krumhansl-Kessler algorithm
   - Computes duration-weighted pitch class histogram
   - Correlates with major/minor key profiles
   - Selects best matching key

2. **Transposition**:
   - Major keys → C major (pitch class 0)
   - Minor keys → A minor (pitch class 9)
   - Minimal semitone shift (within ±6 semitones)

3. **Octave Adjustment**:
   - Shifts octaves to fit within `PITCH_MIN` to `PITCH_MAX` range
   - Fails if unable to fit (drops file)

## Configuration Presets

### Strict MuseFormer (Default)
Exact rules from the paper:
```python
import filter_config as config
config.load_museformer_strict()
```

### Permissive
More flexible filtering:
```python
import filter_config as config
config.load_permissive()
```

Differences:
- Tempo: 20-240 BPM (vs 24-200)
- Pitch: 12-120 (vs 21-108)
- Empty bars: 7 (vs 3)
- Time signatures: Multiple allowed
- Minimum tracks: 1 (vs 2)

### Custom Configuration

Edit `filter_config.py` directly or modify in code:
```python
import filter_config as config
config.TEMPO_MIN = 30
config.TEMPO_MAX = 180
config.MAX_EMPTY_BARS_ALLOWED = 5
config.EMPTY_BAR_METHOD = "sounding"
config.validate_config()  # Always validate after changes
```

## Output Files

### Filtered Directory
Contains files that passed all filtering rules before pitch normalization.

### Pitch Normalized Directory
Contains final output files with key normalization applied.

### Manifest CSV
Updated with detailed information for each file:

**Filter Stage Columns**:
- `stage`: Current processing stage
- `status`: "ok" or "dropped"
- `drop_reason`: Why file was dropped (if applicable)
- `filter_is_duplicate`: Boolean for duplicate detection
- `filter_empty_bars`: Number of empty bars
- `filter_consecutive_empty_bars`: Max consecutive empty bars
- `filter_num_bars`: Total bars
- `filter_duration_beats`: Duration in beats
- `filter_nonempty_tracks`: Number of non-empty tracks
- `filter_pitch_min`, `filter_pitch_max`: Pitch range
- `filter_tempo_min`, `filter_tempo_max`: Tempo range
- `filter_max_note_dur_beats`: Longest note duration
- `filter_distinct_onsets`: Number of unique note start times
- `filter_num_notes`: Total note count

**Pitch Normalization Columns**:
- `pitchnorm_mode`: "major" or "minor"
- `pitchnorm_tonic_pc`: Detected tonic pitch class (0-11)
- `pitchnorm_semitones`: Semitone shift applied
- `pitchnorm_path`: Path to normalized file

### Summary JSON
`pipeline_summary.json` in output directory with:
- Timestamp
- File counts at each stage
- Drop reason statistics
- Pitch normalization statistics

## Advanced Usage

### Empty Bar Detection Methods

**Onset Method** (original):
- A bar is empty if no notes START in that bar
- More permissive (allows sustained notes from previous bars)

**Sounding Method** (stricter):
- A bar is empty if no notes are SOUNDING during that bar
- Better captures actual musical silence

Configure with:
```python
config.EMPTY_BAR_METHOD = "sounding"  # or "onset"
```

### Drum Handling in Empty Bars

Choose whether drum notes count toward "non-empty" bars:
```python
config.COUNT_DRUMS_FOR_EMPTY_BARS = True   # Drums count
config.COUNT_DRUMS_FOR_EMPTY_BARS = False  # Ignore drums
```

### Required Track Name

If your dataset uses a different track name for melody:
```python
config.REQUIRED_TRACK_NAME = "melody"  # or "lead" or other
```

## Pipeline Stages

### Stage 1: Extract Statistics
- Parses all MIDI files
- Extracts tempo, pitch range, duration, empty bars, etc.
- Handles errors gracefully

### Stage 2: Duplicate Detection
- Generates signature from musical content
- Identifies duplicates (keeps first occurrence)

### Stage 3: Apply Filtering Rules
- Checks each file against all filter criteria
- Records drop reason for each filtered file

### Stage 4: Copy Filtered Files
- Copies passing files to filtered directory
- Preserves original files

### Stage 5: Pitch Normalization
- Detects key using Krumhansl-Kessler
- Transposes to target key (C major / A minor)
- Adjusts octaves to fit pitch range
- Drops files that can't fit in range

### Stage 6: Update Manifest
- Updates CSV with all processing results
- Creates backup of original manifest
- Adds detailed statistics for each file

## Troubleshooting

### No MIDI files found
Check that `INPUT_NORMALIZED_DIR` points to correct directory and contains `.mid` or `.midi` files.

### Manifest not found
Ensure `MANIFEST_PATH` points to valid CSV file or create one with at minimum a `raw_path` column.

### All files filtered out
- Check if filtering rules are too strict
- Review drop reason statistics in console output
- Consider using permissive preset
- Adjust specific parameters in `filter_config.py`

### Import errors
Install required packages:
```bash
pip install miditoolkit pandas --break-system-packages
```

### Configuration validation errors
Run validation after changes:
```python
import filter_config as config
config.validate_config()
```

## Performance Notes

- Processing speed: ~10-50 files/second (depends on file complexity)
- Memory usage: Low (files processed one at a time)
- Progress updates: Every 50 files by default

## References

- **MuseFormer Paper**: Filtering rules from Table 4
- **Krumhansl-Kessler**: Key detection algorithm
- **Miditoolkit**: MIDI parsing library

## Example Output

```
================================================================================
MuseFormer MIDI Filtering and Pitch Normalization Pipeline
================================================================================

Input directory: data/musescore_normalized
Output directory (filtered): data/filtered
Output directory (pitch norm): data/pitch_normalized
MIDI files found: 1523

Manifest backup: data/manifest.backup_20250122_143022.csv

--------------------------------------------------------------------------------
STAGE 1: Extracting statistics from MIDI files...
--------------------------------------------------------------------------------
  Processed 50/1523 files...
  Processed 100/1523 files...
  ...

Statistics extracted: 1520 OK, 3 failed

--------------------------------------------------------------------------------
STAGE 2: Detecting duplicates...
--------------------------------------------------------------------------------
  Duplicate: song_002.mid (same as song_001.mid)
  Duplicate: song_045.mid (same as song_012.mid)

Duplicates found: 23

--------------------------------------------------------------------------------
STAGE 3: Applying filtering rules...
--------------------------------------------------------------------------------
Files to keep: 1205
Files to drop: 315

Drop reasons breakdown:
  filter_duplicate_signature: 23
  filter_invalid_time_signature: 45
  filter_lt_2_tracks: 12
  filter_no_square_synth: 67
  filter_tempo_too_slow: 8
  filter_tempo_too_fast: 15
  filter_pitch_too_low: 3
  filter_pitch_too_high: 2
  filter_note_too_long: 18
  filter_too_many_empty_bars: 115
  filter_degenerate_all_same_pitch: 5
  filter_degenerate_all_same_duration: 2

--------------------------------------------------------------------------------
STAGE 4: Copying filtered files...
--------------------------------------------------------------------------------
Copied 1205 files to data/filtered

--------------------------------------------------------------------------------
STAGE 5: Applying pitch normalization...
--------------------------------------------------------------------------------
  Normalized 50/1205 files...
  Normalized 100/1205 files...
  ...

Pitch normalization: 1198 OK, 7 dropped

--------------------------------------------------------------------------------
STAGE 6: Updating manifest...
--------------------------------------------------------------------------------
Manifest updated: data/manifest.csv

================================================================================
PIPELINE COMPLETE
================================================================================
Total input files: 1523
  Parse failures: 3
  Duplicates removed: 23
  Filtered (kept): 1205
  Filtered (dropped): 315
  Pitch normalized: 1198
  Pitch norm failures: 7

Final output: 1198 files in data/pitch_normalized
================================================================================
Summary saved to: data/pitch_normalized/pipeline_summary.json
```

## License

This implementation is based on the MuseFormer paper's filtering methodology.
See original paper for citation information.