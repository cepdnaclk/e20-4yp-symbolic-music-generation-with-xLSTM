"""
MuseFormer MIDI Filtering and Pitch Normalization Pipeline (Improved)

This script implements the filtering rules from Table 4 of the MuseFormer paper
with improvements for robustness and configurability.

Usage:
    python filter_and_normalize.py

Configuration:
    Edit filter_config.py to customize filtering parameters
"""

from pathlib import Path
from dataclasses import dataclass
import json
import math
import shutil
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import miditoolkit
from collections import defaultdict

# Import configuration
try:
    import filter_config as config
except ImportError:
    print("ERROR: filter_config.py not found!")
    print("Please ensure filter_config.py is in the same directory as this script.")
    exit(1)


# =============================================================================
# KEY DETECTION (Krumhansl-Kessler)
# =============================================================================

def rotate(lst: list, k: int) -> list:
    """Rotate list by k positions"""
    k %= len(lst)
    return lst[k:] + lst[:k]


def correlation(a: list, b: list) -> float:
    """Compute cosine-like correlation between two lists"""
    sa = sum(x * x for x in a) ** 0.5
    sb = sum(x * x for x in b) ** 0.5
    if sa == 0 or sb == 0:
        return -1e9
    return sum(x * y for x, y in zip(a, b)) / (sa * sb)


def minimal_mod12_shift(target_pc: int, tonic_pc: int) -> int:
    """
    Calculate minimal semitone shift from tonic_pc to target_pc
    Returns value in range [-6, +5]
    """
    d = (target_pc - tonic_pc) % 12
    if d > 6:
        d -= 12
    return d


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MidiStats:
    """Statistics extracted from a MIDI file for filtering"""
    basename: str
    path: Path
    ticks_per_beat: int
    time_sigs: List[str]
    is_time_sig_valid: bool
    tempo_min: float
    tempo_max: float
    has_tempo_data: bool
    pitch_min: int
    pitch_max: int
    max_note_dur_beats: float
    num_notes: int
    distinct_onsets: int
    programs_sig: tuple
    nonempty_tracks: int
    has_required_track: bool
    empty_bars: int
    consecutive_empty_bars: int
    unique_pitch_count: int
    unique_dur_count: int
    num_bars: int
    duration_beats: float

    def dup_signature(self) -> tuple:
        """Generate signature for duplicate detection"""
        return (
            self.num_bars,
            round(self.duration_beats, 4),
            self.num_notes,
            self.distinct_onsets,
            self.programs_sig,
        )


# =============================================================================
# MIDI LOADING AND STATS EXTRACTION
# =============================================================================

def load_midi(path: Path) -> miditoolkit.MidiFile:
    """Load MIDI file using miditoolkit"""
    return miditoolkit.MidiFile(str(path))


def count_empty_bars_sounding(midi: miditoolkit.MidiFile, tpb: int, num_bars: int, 
                               count_drums: bool = True) -> Tuple[int, int]:
    """
    Count empty bars using "sounding notes" method.
    
    A bar is empty if NO notes are sounding during any part of that bar.
    
    Args:
        midi: MidiFile object
        tpb: Ticks per beat
        num_bars: Total number of bars
        count_drums: Whether to include drum notes
    
    Returns:
        (total_empty_bars, max_consecutive_empty_bars)
    """
    bar_ticks = 4 * tpb  # Assuming 4/4 time
    
    if bar_ticks == 0 or num_bars == 0:
        return 0, 0
    
    # Collect all notes
    all_notes = []
    for inst in midi.instruments:
        if inst.is_drum and not count_drums:
            continue
        all_notes.extend(inst.notes)
    
    if not all_notes:
        return num_bars, num_bars
    
    # Check each bar
    empty_count = 0
    consecutive = 0
    max_consecutive = 0
    
    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = (bar_idx + 1) * bar_ticks
        
        # Check if any note is sounding during this bar
        # Note is sounding if: note.start < bar_end AND note.end > bar_start
        has_sound = False
        for note in all_notes:
            if note.start < bar_end and note.end > bar_start:
                has_sound = True
                break
        
        if not has_sound:
            empty_count += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    return empty_count, max_consecutive


def count_empty_bars_onset(midi: miditoolkit.MidiFile, tpb: int, num_bars: int,
                           count_drums: bool = True) -> Tuple[int, int]:
    """
    Count empty bars using "onset" method (original implementation).
    
    A bar is empty if NO notes start in that bar.
    
    Args:
        midi: MidiFile object
        tpb: Ticks per beat
        num_bars: Total number of bars
        count_drums: Whether to include drum notes
    
    Returns:
        (total_empty_bars, max_consecutive_empty_bars)
    """
    bar_ticks = 4 * tpb
    
    if bar_ticks == 0 or num_bars == 0:
        return 0, 0
    
    # Collect onsets
    onsets = set()
    for inst in midi.instruments:
        if inst.is_drum and not count_drums:
            continue
        for note in inst.notes:
            onsets.add(int(note.start))
    
    if not onsets:
        return num_bars, num_bars
    
    # Sort onsets for efficient scanning
    starts = sorted(onsets)
    
    empty_count = 0
    consecutive = 0
    max_consecutive = 0
    si = 0
    
    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = (bar_idx + 1) * bar_ticks
        
        # Advance pointer to first onset >= bar_start
        while si < len(starts) and starts[si] < bar_start:
            si += 1
        
        # Check if there's an onset in this bar
        if si >= len(starts) or starts[si] >= bar_end:
            empty_count += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    return empty_count, max_consecutive


def extract_stats(path: Path) -> MidiStats:
    """Extract all statistics from a MIDI file"""
    midi = load_midi(path)
    tpb = midi.ticks_per_beat
    
    # -------------------------------------------------------------------------
    # Time Signatures
    # -------------------------------------------------------------------------
    ts = getattr(midi, "time_signature_changes", []) or []
    time_sigs = [f"{x.numerator}/{x.denominator}" for x in ts] if ts else []
    
    # Validate time signature
    if config.ALLOW_MISSING_TIME_SIGNATURE and len(time_sigs) == 0:
        # Treat as valid (assume 4/4)
        is_time_sig_valid = True
    elif len(time_sigs) == 0:
        # Strict mode: must have time signature
        is_time_sig_valid = False
    else:
        # Check all time sigs are in allowed list
        is_time_sig_valid = all(ts in config.ALLOWED_TIME_SIGNATURES for ts in time_sigs)
    
    # -------------------------------------------------------------------------
    # Tempo
    # -------------------------------------------------------------------------
    tempos = getattr(midi, "tempo_changes", []) or []
    if tempos:
        bpm_vals = [float(t.tempo) for t in tempos]
        tempo_min = min(bpm_vals)
        tempo_max = max(bpm_vals)
        has_tempo_data = True
    else:
        # No tempo data
        tempo_min = float("inf")
        tempo_max = float("-inf")
        has_tempo_data = False
    
    # -------------------------------------------------------------------------
    # Instruments Signature
    # -------------------------------------------------------------------------
    prog_list = []
    for inst in midi.instruments:
        if inst.is_drum:
            prog_list.append(("drum", 0))
        else:
            # Distinguish square_synth if needed
            if inst.name == config.REQUIRED_TRACK_NAME:
                prog_list.append(("required", inst.program))
            else:
                prog_list.append(("inst", inst.program))
    programs_sig = tuple(sorted(prog_list))
    
    # -------------------------------------------------------------------------
    # Collect Notes
    # -------------------------------------------------------------------------
    all_notes = []
    non_drum_notes = []
    onsets = set()
    pitches = []
    durs = []
    max_end_tick = 0
    
    nonempty_tracks = 0
    has_required = False
    
    for inst in midi.instruments:
        if inst.notes:
            nonempty_tracks += 1
        
        if inst.name == config.REQUIRED_TRACK_NAME and len(inst.notes) > 0:
            has_required = True
        
        for n in inst.notes:
            all_notes.append(n)
            onsets.add(int(n.start))
            max_end_tick = max(max_end_tick, int(n.end))
            
            if not inst.is_drum:
                non_drum_notes.append(n)
                pitches.append(int(n.pitch))
                durs.append(int(n.end) - int(n.start))
    
    num_notes = len(all_notes)
    distinct_onsets = len(onsets)
    
    # -------------------------------------------------------------------------
    # Pitch and Duration Stats
    # -------------------------------------------------------------------------
    if non_drum_notes:
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        max_note_dur_beats = max(d / tpb for d in durs) if tpb > 0 else float("inf")
        unique_pitch_count = len(set(pitches))
        unique_dur_count = len(set(durs))
    else:
        pitch_min = 999
        pitch_max = -999
        max_note_dur_beats = float("inf")
        unique_pitch_count = 0
        unique_dur_count = 0
    
    # -------------------------------------------------------------------------
    # Duration and Bars
    # -------------------------------------------------------------------------
    duration_beats = max_end_tick / tpb if tpb > 0 else 0.0
    num_bars = int(math.ceil(duration_beats / 4.0)) if duration_beats > 0 else 0
    
    # -------------------------------------------------------------------------
    # Empty Bars
    # -------------------------------------------------------------------------
    if config.EMPTY_BAR_METHOD == "sounding":
        empty_bars, consecutive_empty = count_empty_bars_sounding(
            midi, tpb, num_bars, config.COUNT_DRUMS_FOR_EMPTY_BARS
        )
    else:  # "onset"
        empty_bars, consecutive_empty = count_empty_bars_onset(
            midi, tpb, num_bars, config.COUNT_DRUMS_FOR_EMPTY_BARS
        )
    
    return MidiStats(
        basename=path.name,
        path=path,
        ticks_per_beat=tpb,
        time_sigs=time_sigs,
        is_time_sig_valid=is_time_sig_valid,
        tempo_min=tempo_min,
        tempo_max=tempo_max,
        has_tempo_data=has_tempo_data,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        max_note_dur_beats=max_note_dur_beats,
        num_notes=num_notes,
        distinct_onsets=distinct_onsets,
        programs_sig=programs_sig,
        nonempty_tracks=nonempty_tracks,
        has_required_track=has_required,
        empty_bars=empty_bars,
        consecutive_empty_bars=consecutive_empty,
        unique_pitch_count=unique_pitch_count,
        unique_dur_count=unique_dur_count,
        num_bars=num_bars,
        duration_beats=duration_beats,
    )


# =============================================================================
# FILTERING LOGIC
# =============================================================================

def filter_reason(s: MidiStats) -> Optional[str]:
    """
    Determine if a MIDI file should be filtered out.
    
    Returns:
        None if file passes all filters
        String describing the reason for filtering if it fails
    """
    # Time signature check
    if not s.is_time_sig_valid:
        return "filter_invalid_time_signature"
    
    # Track count check
    if s.nonempty_tracks < config.MIN_NONEMPTY_TRACKS:
        return f"filter_lt_{config.MIN_NONEMPTY_TRACKS}_tracks"
    
    # Required track check
    if not s.has_required_track:
        return f"filter_no_{config.REQUIRED_TRACK_NAME}"
    
    # Tempo check (improved)
    if not s.has_tempo_data:
        return "filter_no_tempo_data"
    
    if s.tempo_min < config.TEMPO_MIN:
        return "filter_tempo_too_slow"
    
    if s.tempo_max > config.TEMPO_MAX:
        return "filter_tempo_too_fast"
    
    # Pitch range check
    if s.pitch_min < config.PITCH_MIN:
        return "filter_pitch_too_low"
    
    if s.pitch_max > config.PITCH_MAX:
        return "filter_pitch_too_high"
    
    # Note duration check
    if s.max_note_dur_beats > config.MAX_NOTE_DURATION_BEATS:
        return "filter_note_too_long"
    
    # Empty bars check (using consecutive count for better detection)
    if s.consecutive_empty_bars > config.MAX_EMPTY_BARS_ALLOWED:
        return "filter_too_many_empty_bars"
    
    # Degenerate content check
    if config.DROP_DEGENERATE_CONTENT:
        if s.unique_pitch_count == 1:
            return "filter_degenerate_all_same_pitch"
        if s.unique_dur_count == 1:
            return "filter_degenerate_all_same_duration"
    
    return None


# =============================================================================
# PITCH NORMALIZATION
# =============================================================================

def pitch_class_histogram(midi: miditoolkit.MidiFile) -> List[float]:
    """
    Compute duration-weighted pitch class histogram over non-drum notes.
    
    Returns:
        List of 12 floats representing the weight of each pitch class (0-11)
    """
    hist = [0.0] * 12
    tpb = midi.ticks_per_beat
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc = int(n.pitch) % 12
            dur_beats = (int(n.end) - int(n.start)) / tpb if tpb > 0 else 0.0
            hist[pc] += max(dur_beats, 0.0)
    
    return hist


def detect_key_and_shift(midi: miditoolkit.MidiFile) -> Tuple[Optional[str], Optional[int], int]:
    """
    Detect key using Krumhansl-Kessler profiles and calculate transposition.
    
    Returns:
        (mode, tonic_pc, semitone_shift)
        - mode: "major" or "minor" (None if no pitched content)
        - tonic_pc: Pitch class of detected tonic 0-11 (None if no content)
        - semitone_shift: Semitones to transpose (0 if no content)
    """
    hist = pitch_class_histogram(midi)
    
    if sum(hist) == 0:
        return None, None, 0  # No pitched content
    
    # Find best matching major key
    best_major = (-1e9, None)
    for tonic in range(12):
        score = correlation(hist, rotate(config.KK_MAJOR_PROFILE, tonic))
        if score > best_major[0]:
            best_major = (score, tonic)
    
    # Find best matching minor key
    best_minor = (-1e9, None)
    for tonic in range(12):
        score = correlation(hist, rotate(config.KK_MINOR_PROFILE, tonic))
        if score > best_minor[0]:
            best_minor = (score, tonic)
    
    # Choose major vs minor
    if best_major[0] >= best_minor[0]:
        mode = "major"
        tonic = best_major[1]
        target = config.MAJOR_TARGET_PC
    else:
        mode = "minor"
        tonic = best_minor[1]
        target = config.MINOR_TARGET_PC
    
    shift = minimal_mod12_shift(target, tonic)
    return mode, tonic, shift


def apply_pitch_norm(midi: miditoolkit.MidiFile, shift: int) -> bool:
    """
    Apply pitch normalization to MIDI file.
    
    1. Transpose all non-drum notes by 'shift' semitones
    2. Adjust octaves to fit within [PITCH_MIN, PITCH_MAX] range
    
    Args:
        midi: MidiFile object (modified in-place)
        shift: Semitones to transpose
    
    Returns:
        True if successful, False if unable to fit in range
    """
    # Apply key transposition
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(n.pitch) + int(shift)
    
    # Check if already in valid range
    all_pitched = [
        n.pitch 
        for inst in midi.instruments 
        if not inst.is_drum 
        for n in inst.notes
    ]
    
    if not all_pitched:
        return True  # No pitched notes, nothing to do
    
    mn = min(all_pitched)
    mx = max(all_pitched)
    
    # Early exit if already valid
    if config.PITCH_MIN <= mn and mx <= config.PITCH_MAX:
        return True
    
    # Try octave adjustments
    # Shift up if too low
    while mn < config.PITCH_MIN:
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                n.pitch += 12
        mn += 12
        mx += 12
        
        # Check if we've gone out of MIDI range
        if mx > 127:
            return False
    
    # Shift down if too high
    while mx > config.PITCH_MAX:
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                n.pitch -= 12
        mn -= 12
        mx -= 12
        
        # Check if we've gone out of MIDI range
        if mn < 0:
            return False
    
    # Final validation
    all_pitched = [
        n.pitch 
        for inst in midi.instruments 
        if not inst.is_drum 
        for n in inst.notes
    ]
    
    if any(p < 0 or p > 127 for p in all_pitched):
        return False
    
    return True


# =============================================================================
# MANIFEST MANAGEMENT
# =============================================================================

def update_manifest(df: pd.DataFrame, updates: dict):
    """
    Update manifest DataFrame with new data.
    
    Args:
        df: Manifest DataFrame
        updates: Dict mapping basename -> dict of column values
    """
    # Ensure raw_basename column exists
    if "raw_basename" not in df.columns:
        df["raw_basename"] = df["raw_path"].astype(str).map(lambda p: Path(p).name)
    
    # Add any new columns
    all_cols = set(k for u in updates.values() for k in u.keys())
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Apply updates
    for basename, cols in updates.items():
        mask = df["raw_basename"].astype(str) == basename
        for k, v in cols.items():
            df.loc[mask, k] = v


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("MuseFormer MIDI Filtering and Pitch Normalization Pipeline")
    print("=" * 80)
    print()
    
    # Create output directories
    config.OUTPUT_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_PITCH_NORM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    midi_files = sorted(
        list(config.INPUT_NORMALIZED_DIR.glob("*.mid")) + 
        list(config.INPUT_NORMALIZED_DIR.glob("*.midi"))
    )
    
    print(f"Input directory: {config.INPUT_NORMALIZED_DIR}")
    print(f"Output directory (filtered): {config.OUTPUT_FILTERED_DIR}")
    print(f"Output directory (pitch norm): {config.OUTPUT_PITCH_NORM_DIR}")
    print(f"MIDI files found: {len(midi_files)}")
    print()
    
    if len(midi_files) == 0:
        print("ERROR: No MIDI files found!")
        return
    
    # Load and backup manifest
    if not config.MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {config.MANIFEST_PATH}")
        return
    
    df = pd.read_csv(config.MANIFEST_PATH)
    backup_path = config.MANIFEST_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(backup_path, index=False)
    print(f"Manifest backup: {backup_path}")
    print()
    
    # =========================================================================
    # STAGE 1: Extract Statistics
    # =========================================================================
    print("-" * 80)
    print("STAGE 1: Extracting statistics from MIDI files...")
    print("-" * 80)
    
    stats = []
    failed = []
    
    for i, path in enumerate(midi_files, 1):
        try:
            s = extract_stats(path)
            stats.append(s)
            if i % 50 == 0:
                print(f"  Processed {i}/{len(midi_files)} files...")
        except Exception as e:
            failed.append((path.name, str(e)))
            print(f"  ERROR parsing {path.name}: {e}")
    
    print(f"\nStatistics extracted: {len(stats)} OK, {len(failed)} failed")
    print()
    
    # =========================================================================
    # STAGE 2: Duplicate Detection
    # =========================================================================
    print("-" * 80)
    print("STAGE 2: Detecting duplicates...")
    print("-" * 80)
    
    sig_to_first = {}
    is_dup = {}
    
    for s in stats:
        sig = s.dup_signature()
        if sig not in sig_to_first:
            sig_to_first[sig] = s.basename
            is_dup[s.basename] = False
        else:
            is_dup[s.basename] = True
            print(f"  Duplicate: {s.basename} (same as {sig_to_first[sig]})")
    
    num_dups = sum(is_dup.values())
    print(f"\nDuplicates found: {num_dups}")
    print()
    
    # =========================================================================
    # STAGE 3: Apply Filtering Rules
    # =========================================================================
    print("-" * 80)
    print("STAGE 3: Applying filtering rules...")
    print("-" * 80)
    
    keep = []
    drop_reason_map = {}
    drop_stats = defaultdict(int)
    
    for s in stats:
        # Check if duplicate
        if is_dup.get(s.basename, False):
            drop_reason_map[s.basename] = "filter_duplicate_signature"
            drop_stats["filter_duplicate_signature"] += 1
            continue
        
        # Apply filter rules
        reason = filter_reason(s)
        if reason is None:
            keep.append(s)
        else:
            drop_reason_map[s.basename] = reason
            drop_stats[reason] += 1
    
    print(f"Files to keep: {len(keep)}")
    print(f"Files to drop: {len(drop_reason_map)}")
    print()
    print("Drop reasons breakdown:")
    for reason, count in sorted(drop_stats.items()):
        print(f"  {reason}: {count}")
    print()
    
    # =========================================================================
    # STAGE 4: Copy Filtered Files
    # =========================================================================
    print("-" * 80)
    print("STAGE 4: Copying filtered files...")
    print("-" * 80)
    
    for s in keep:
        shutil.copy2(s.path, config.OUTPUT_FILTERED_DIR / s.basename)
    
    print(f"Copied {len(keep)} files to {config.OUTPUT_FILTERED_DIR}")
    print()
    
    # =========================================================================
    # STAGE 5: Pitch Normalization
    # =========================================================================
    print("-" * 80)
    print("STAGE 5: Applying pitch normalization...")
    print("-" * 80)
    
    pitch_updates = {}
    pitch_drops = {}
    
    for i, s in enumerate(keep, 1):
        src = config.OUTPUT_FILTERED_DIR / s.basename
        dst = config.OUTPUT_PITCH_NORM_DIR / s.basename
        
        try:
            midi = load_midi(src)
            mode, tonic, shift = detect_key_and_shift(midi)
            
            if mode is None:
                # No pitched content
                pitch_drops[s.basename] = "pitchnorm_no_pitched_content"
                continue
            
            ok = apply_pitch_norm(midi, shift)
            
            if not ok:
                pitch_drops[s.basename] = "pitchnorm_out_of_range"
                continue
            
            midi.dump(str(dst))
            pitch_updates[s.basename] = {
                "pitchnorm_mode": mode,
                "pitchnorm_tonic_pc": tonic,
                "pitchnorm_semitones": shift,
                "pitchnorm_path": str(dst),
            }
            
            if i % 50 == 0:
                print(f"  Normalized {i}/{len(keep)} files...")
                
        except Exception as e:
            pitch_drops[s.basename] = f"pitchnorm_error:{e}"
            print(f"  ERROR normalizing {s.basename}: {e}")
    
    print(f"\nPitch normalization: {len(pitch_updates)} OK, {len(pitch_drops)} dropped")
    print()
    
    # =========================================================================
    # STAGE 6: Update Manifest
    # =========================================================================
    print("-" * 80)
    print("STAGE 6: Updating manifest...")
    print("-" * 80)
    
    updates = {}
    kept_basenames = set(s.basename for s in keep)
    
    # Update filtering stage
    for s in stats:
        b = s.basename
        
        if b in kept_basenames:
            # File passed filtering
            updates[b] = {
                "stage": config.STAGE_FILTER,
                "status": "ok",
                "drop_reason": pd.NA,
                "error_msg": pd.NA,
                "mscore_norm_path": str(config.INPUT_NORMALIZED_DIR / b),
                "filtered_path": str(config.OUTPUT_FILTERED_DIR / b),
                "filter_is_duplicate": bool(is_dup.get(b, False)),
                "filter_empty_bars": s.empty_bars,
                "filter_consecutive_empty_bars": s.consecutive_empty_bars,
                "filter_num_bars": s.num_bars,
                "filter_duration_beats": s.duration_beats,
                "filter_nonempty_tracks": s.nonempty_tracks,
                "filter_pitch_min": s.pitch_min,
                "filter_pitch_max": s.pitch_max,
                "filter_tempo_min": s.tempo_min,
                "filter_tempo_max": s.tempo_max,
                "filter_max_note_dur_beats": s.max_note_dur_beats,
                "filter_distinct_onsets": s.distinct_onsets,
                "filter_num_notes": s.num_notes,
            }
        else:
            # File was dropped
            reason = drop_reason_map.get(b, "filter_unknown")
            updates[b] = {
                "stage": config.STAGE_FILTER,
                "status": "dropped",
                "drop_reason": reason,
                "error_msg": pd.NA,
                "mscore_norm_path": str(config.INPUT_NORMALIZED_DIR / b),
                "filtered_path": pd.NA,
            }
    
    # Update pitch normalization stage
    for b, cols in pitch_updates.items():
        updates.setdefault(b, {})
        updates[b].update({
            "stage": config.STAGE_PITCH_NORM,
            "status": "ok",
            "drop_reason": pd.NA,
            "error_msg": pd.NA,
        })
        updates[b].update(cols)
    
    for b, reason in pitch_drops.items():
        updates.setdefault(b, {})
        updates[b].update({
            "stage": config.STAGE_PITCH_NORM,
            "status": "dropped",
            "drop_reason": reason,
            "error_msg": pd.NA,
        })
    
    update_manifest(df, updates)
    df.to_csv(config.MANIFEST_PATH, index=False)
    
    print(f"Manifest updated: {config.MANIFEST_PATH}")
    print()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total input files: {len(midi_files)}")
    print(f"  Parse failures: {len(failed)}")
    print(f"  Duplicates removed: {num_dups}")
    print(f"  Filtered (kept): {len(keep)}")
    print(f"  Filtered (dropped): {len(drop_reason_map)}")
    print(f"  Pitch normalized: {len(pitch_updates)}")
    print(f"  Pitch norm failures: {len(pitch_drops)}")
    print()
    print(f"Final output: {len(pitch_updates)} files in {config.OUTPUT_PITCH_NORM_DIR}")
    print("=" * 80)
    
    # Save summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_files": len(midi_files),
        "parse_failures": len(failed),
        "duplicates_removed": num_dups,
        "filtered_kept": len(keep),
        "filtered_dropped": len(drop_reason_map),
        "pitch_normalized": len(pitch_updates),
        "pitch_norm_failures": len(pitch_drops),
        "drop_reasons": dict(drop_stats),
        "pitch_drop_reasons": dict(defaultdict(int, 
            [(k.split(":")[0], v) for k, v in 
             defaultdict(int, [(reason, 1) for reason in pitch_drops.values()]).items()]
        ))
    }
    
    summary_path = config.OUTPUT_PITCH_NORM_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()