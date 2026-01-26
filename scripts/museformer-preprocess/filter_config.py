"""
Configuration for MuseFormer MIDI Filtering and Pitch Normalization Pipeline

This file contains all configurable parameters for the filtering pipeline.
Adjust these values to customize the filtering behavior for your dataset.

Based on Table 4 from the MuseFormer paper with additional configurability.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Input: Directory containing MuseScore-normalized MIDI files
INPUT_NORMALIZED_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/04_musescore_norm")

# Output: Directory for filtered files (before pitch normalization)
OUTPUT_FILTERED_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/05_filtered")

# Output: Directory for pitch-normalized files (final output)
OUTPUT_PITCH_NORM_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/06_pitch_normalize")

# Manifest CSV file to update with processing results
MANIFEST_PATH = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/logs/sample_manifest.csv")

# Stage names for manifest tracking
STAGE_FILTER = "filter"
STAGE_PITCH_NORM = "pitch_norm"


# =============================================================================
# TIME SIGNATURE FILTERING
# =============================================================================

# Allowed time signatures (MuseFormer paper: only 4/4)
ALLOWED_TIME_SIGNATURES = ["4/4"]

# If True, files with no time signature will be treated as valid (assumed 4/4)
# If False, files must have an explicit time signature
ALLOW_MISSING_TIME_SIGNATURE = True


# =============================================================================
# TRACK REQUIREMENTS
# =============================================================================

# Minimum number of non-empty tracks required
# MuseFormer paper: at least 2 instruments
MIN_NONEMPTY_TRACKS = 2

# Required track name (MuseFormer: "square_synth" for melody)
# Files without this track will be filtered out
REQUIRED_TRACK_NAME = "square_synth"


# =============================================================================
# TEMPO FILTERING
# =============================================================================

# Minimum tempo in BPM (MuseFormer paper: 24 BPM)
TEMPO_MIN = 24

# Maximum tempo in BPM (MuseFormer paper: 200 BPM)
TEMPO_MAX = 200


# =============================================================================
# PITCH RANGE FILTERING
# =============================================================================

# Minimum MIDI pitch allowed (MuseFormer paper: 21 = A0)
PITCH_MIN = 21

# Maximum MIDI pitch allowed (MuseFormer paper: 108 = C8)
PITCH_MAX = 108


# =============================================================================
# NOTE DURATION FILTERING
# =============================================================================

# Maximum note duration in beats (MuseFormer paper: 16 beats = 4 bars in 4/4)
MAX_NOTE_DURATION_BEATS = 16.0


# =============================================================================
# EMPTY BARS FILTERING
# =============================================================================

# Maximum number of consecutive empty bars allowed
# MuseFormer paper: filter files with 4 or more consecutive empty bars
MAX_EMPTY_BARS_ALLOWED = 3

# Method for counting empty bars:
#   "onset" - A bar is empty if no notes START in that bar (original)
#   "sounding" - A bar is empty if no notes are SOUNDING during that bar (stricter)
EMPTY_BAR_METHOD = "sounding"

# Whether to count drum notes when determining if a bar is empty
# True = drums count as notes (bar with only drums is not empty)
# False = ignore drums (bar with only drums is considered empty)
COUNT_DRUMS_FOR_EMPTY_BARS = True


# =============================================================================
# DEGENERATE CONTENT FILTERING
# =============================================================================

# If True, filter out files where all notes have the same pitch
# or all notes have the same duration (degenerate/repetitive content)
DROP_DEGENERATE_CONTENT = True


# =============================================================================
# PITCH NORMALIZATION
# =============================================================================

# Target pitch class for major keys (MuseFormer: C major = 0)
MAJOR_TARGET_PC = 0  # C

# Target pitch class for minor keys (MuseFormer: A minor = 9)
MINOR_TARGET_PC = 9  # A

# Krumhansl-Kessler key profiles for major and minor keys
# These are used for automatic key detection
# Values represent the perceived stability of each pitch class in the key

# Major profile (C major: C=1st position has highest weight)
KK_MAJOR_PROFILE = [
    6.35,  # C  (tonic)
    2.23,  # C#
    3.48,  # D  (supertonic)
    2.33,  # D#
    4.38,  # E  (mediant)
    4.09,  # F  (subdominant)
    2.52,  # F#
    5.19,  # G  (dominant)
    2.39,  # G#
    3.66,  # A  (submediant)
    2.29,  # A#
    2.88,  # B  (leading tone)
]

# Minor profile (A minor: A=1st position has highest weight)
KK_MINOR_PROFILE = [
    6.33,  # A  (tonic)
    2.68,  # A#
    3.52,  # B  (supertonic)
    5.38,  # C  (mediant)
    2.60,  # C#
    3.53,  # D  (subdominant)
    2.54,  # D#
    4.75,  # E  (dominant)
    3.98,  # F  (submediant)
    2.69,  # F#
    3.34,  # G  (subtonic)
    3.17,  # G#
]


# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

# Duplicate detection signature components
# Duplicates are detected by comparing:
#   - Number of bars
#   - Total duration in beats
#   - Number of notes
#   - Number of distinct note onsets
#   - Instrument program signature
# Files with identical signatures are considered duplicates

# Logging verbosity
# 0 = Errors only
# 1 = Progress updates every 50 files
# 2 = Detailed per-file information
VERBOSITY = 1


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration parameters.
    Raises ValueError if any parameters are invalid.
    """
    errors = []
    
    # Path validation
    if not INPUT_NORMALIZED_DIR.exists():
        errors.append(f"Input directory does not exist: {INPUT_NORMALIZED_DIR}")
    
    if MANIFEST_PATH.exists() and not MANIFEST_PATH.is_file():
        errors.append(f"Manifest path exists but is not a file: {MANIFEST_PATH}")
    
    # Numeric range validation
    if TEMPO_MIN <= 0:
        errors.append(f"TEMPO_MIN must be positive, got {TEMPO_MIN}")
    
    if TEMPO_MAX <= TEMPO_MIN:
        errors.append(f"TEMPO_MAX ({TEMPO_MAX}) must be greater than TEMPO_MIN ({TEMPO_MIN})")
    
    if PITCH_MIN < 0 or PITCH_MIN > 127:
        errors.append(f"PITCH_MIN must be in range [0, 127], got {PITCH_MIN}")
    
    if PITCH_MAX < 0 or PITCH_MAX > 127:
        errors.append(f"PITCH_MAX must be in range [0, 127], got {PITCH_MAX}")
    
    if PITCH_MAX <= PITCH_MIN:
        errors.append(f"PITCH_MAX ({PITCH_MAX}) must be greater than PITCH_MIN ({PITCH_MIN})")
    
    if MAX_NOTE_DURATION_BEATS <= 0:
        errors.append(f"MAX_NOTE_DURATION_BEATS must be positive, got {MAX_NOTE_DURATION_BEATS}")
    
    if MAX_EMPTY_BARS_ALLOWED < 0:
        errors.append(f"MAX_EMPTY_BARS_ALLOWED must be non-negative, got {MAX_EMPTY_BARS_ALLOWED}")
    
    if MIN_NONEMPTY_TRACKS < 1:
        errors.append(f"MIN_NONEMPTY_TRACKS must be at least 1, got {MIN_NONEMPTY_TRACKS}")
    
    # String validation
    if EMPTY_BAR_METHOD not in ["onset", "sounding"]:
        errors.append(f"EMPTY_BAR_METHOD must be 'onset' or 'sounding', got '{EMPTY_BAR_METHOD}'")
    
    if not ALLOWED_TIME_SIGNATURES:
        errors.append("ALLOWED_TIME_SIGNATURES cannot be empty")
    
    if not REQUIRED_TRACK_NAME:
        errors.append("REQUIRED_TRACK_NAME cannot be empty")
    
    # Pitch class validation
    if MAJOR_TARGET_PC < 0 or MAJOR_TARGET_PC > 11:
        errors.append(f"MAJOR_TARGET_PC must be in range [0, 11], got {MAJOR_TARGET_PC}")
    
    if MINOR_TARGET_PC < 0 or MINOR_TARGET_PC > 11:
        errors.append(f"MINOR_TARGET_PC must be in range [0, 11], got {MINOR_TARGET_PC}")
    
    # Key profile validation
    if len(KK_MAJOR_PROFILE) != 12:
        errors.append(f"KK_MAJOR_PROFILE must have 12 elements, got {len(KK_MAJOR_PROFILE)}")
    
    if len(KK_MINOR_PROFILE) != 12:
        errors.append(f"KK_MINOR_PROFILE must have 12 elements, got {len(KK_MINOR_PROFILE)}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))


# Run validation when config is imported
validate_config()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def load_museformer_strict():
    """
    Load strict MuseFormer paper configuration.
    This matches the exact filtering rules from Table 4.
    """
    global TEMPO_MIN, TEMPO_MAX, PITCH_MIN, PITCH_MAX
    global MAX_NOTE_DURATION_BEATS, MAX_EMPTY_BARS_ALLOWED
    global ALLOWED_TIME_SIGNATURES, MIN_NONEMPTY_TRACKS
    global DROP_DEGENERATE_CONTENT, EMPTY_BAR_METHOD
    global ALLOW_MISSING_TIME_SIGNATURE
    
    TEMPO_MIN = 24
    TEMPO_MAX = 200
    PITCH_MIN = 21
    PITCH_MAX = 108
    MAX_NOTE_DURATION_BEATS = 16.0
    MAX_EMPTY_BARS_ALLOWED = 3
    ALLOWED_TIME_SIGNATURES = ["4/4"]
    MIN_NONEMPTY_TRACKS = 2
    DROP_DEGENERATE_CONTENT = True
    EMPTY_BAR_METHOD = "onset"
    ALLOW_MISSING_TIME_SIGNATURE = False
    
    validate_config()
    print("Loaded: MuseFormer Strict configuration")


def load_permissive():
    """
    Load more permissive configuration.
    Allows wider ranges and more flexibility.
    """
    global TEMPO_MIN, TEMPO_MAX, PITCH_MIN, PITCH_MAX
    global MAX_NOTE_DURATION_BEATS, MAX_EMPTY_BARS_ALLOWED
    global ALLOWED_TIME_SIGNATURES, MIN_NONEMPTY_TRACKS
    global DROP_DEGENERATE_CONTENT, EMPTY_BAR_METHOD
    global ALLOW_MISSING_TIME_SIGNATURE
    
    TEMPO_MIN = 20
    TEMPO_MAX = 240
    PITCH_MIN = 12
    PITCH_MAX = 120
    MAX_NOTE_DURATION_BEATS = 32.0
    MAX_EMPTY_BARS_ALLOWED = 7
    ALLOWED_TIME_SIGNATURES = ["4/4", "3/4", "2/4", "6/8", "5/4"]
    MIN_NONEMPTY_TRACKS = 1
    DROP_DEGENERATE_CONTENT = False
    EMPTY_BAR_METHOD = "onset"
    ALLOW_MISSING_TIME_SIGNATURE = True
    
    validate_config()
    print("Loaded: Permissive configuration")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Example usage in your pipeline:

1. Default configuration (as specified above):
   python filter_and_normalize.py

2. Use strict MuseFormer configuration:
   In your code, before running:
   
   import filter_config as config
   config.load_museformer_strict()
   
3. Use permissive configuration:
   
   import filter_config as config
   config.load_permissive()

4. Custom configuration:
   
   import filter_config as config
   config.TEMPO_MIN = 30
   config.TEMPO_MAX = 180
   config.MAX_EMPTY_BARS_ALLOWED = 5
   config.validate_config()  # Always validate after changes

5. Override from command line:
   You can add argument parsing to filter_and_normalize.py to override
   specific parameters without modifying this file.
"""