"""
Stage 1: MIDI Parsing

Parses MIDI files using miditoolkit and extracts metadata to manifest.
In dev mode, copies successfully parsed files to output directory.
In prod mode, only updates manifest without copying files.
"""

from pathlib import Path
from typing import List, Tuple
import traceback
import logging

import pandas as pd
import numpy as np
import miditoolkit

from manifest_utils import ManifestManager
from pipeline_config import PipelineConfig, MIDI_EXTENSIONS, STAGE_NAMES

logger = logging.getLogger(__name__)


def get_time_signatures(midi: miditoolkit.MidiFile) -> List[str]:
    """Extract time signatures from MIDI file."""
    if not midi.time_signature_changes:
        return []
    
    sigs = []
    for ts in midi.time_signature_changes:
        sigs.append(f"{ts.numerator}/{ts.denominator}")
    
    # Unique but preserve order
    unique_sigs = []
    for sig in sigs:
        if sig not in unique_sigs:
            unique_sigs.append(sig)
    
    return unique_sigs


def get_tempo_stats(midi: miditoolkit.MidiFile) -> Tuple[float, float]:
    """Extract tempo statistics from MIDI file."""
    tempos = [t.tempo for t in getattr(midi, 'tempo_changes', [])] or [120.0]
    return float(np.min(tempos)), float(np.max(tempos))


def get_note_stats(midi: miditoolkit.MidiFile) -> dict:
    """Extract note statistics from MIDI file."""
    pitches = []
    durs = []
    onsets = set()
    n_notes = 0
    programs = set()
    has_drum = False
    n_tracks = len(midi.instruments)
    
    for inst in midi.instruments:
        if inst.is_drum:
            has_drum = True
        else:
            programs.add(int(inst.program))
        
        for note in inst.notes:
            n_notes += 1
            pitches.append(note.pitch)
            # Duration in beats
            durs.append((note.end - note.start) / midi.ticks_per_beat)
            onsets.add(note.start)
    
    if n_notes == 0:
        return None
    
    return {
        'pitch_min': float(np.min(pitches)),
        'pitch_max': float(np.max(pitches)),
        'max_note_dur_beats': float(np.max(durs)),
        'num_notes': float(n_notes),
        'num_tracks': float(n_tracks),
        'distinct_onsets': float(len(onsets)),
        'programs': sorted(list(programs)),
        'has_drum': bool(has_drum),
    }


def parse_midi_file(path: Path, raw_dir: Path) -> dict:
    """
    Parse a single MIDI file and extract metadata.
    
    Args:
        path: Path to MIDI file
        raw_dir: Base raw directory (for computing relative path)
    
    Returns:
        Dict with parsing results and metadata
    """
    # Use relative path from raw_dir as file_id for uniqueness
    # e.g., "0/00000ec8a66b6bd2ef809b0443eeae41.mid"
    try:
        rel_path = path.relative_to(raw_dir)
        file_id = str(rel_path.with_suffix(''))  # Remove .mid extension
    except ValueError:
        # Fallback if path is not relative to raw_dir
        file_id = path.stem
    
    result = {
        'file_id': file_id,
        'raw_path': str(path),
        'raw_basename': path.name,
        'stage': STAGE_NAMES['parsing'],
        'status': 'ok',
        'drop_reason': '',
        'error_msg': '',
        'time_signatures': '',
        'is_4_4_only': '',
        'tempo_min': '',
        'tempo_max': '',
        'pitch_min': '',
        'pitch_max': '',
        'max_note_dur_beats': '',
        'num_notes': '',
        'num_tracks': '',
        'distinct_onsets': '',
        'programs': '',
        'has_drum': '',
    }
    
    try:
        # Parse MIDI
        midi = miditoolkit.MidiFile(str(path))
        
        # Time signatures
        time_sigs = get_time_signatures(midi)
        result['time_signatures'] = ','.join(time_sigs) if time_sigs else ''
        result['is_4_4_only'] = bool(time_sigs and all(ts == '4/4' for ts in time_sigs))
        
        # Tempo
        tempo_min, tempo_max = get_tempo_stats(midi)
        result['tempo_min'] = tempo_min
        result['tempo_max'] = tempo_max
        
        # Notes
        note_stats = get_note_stats(midi)
        if note_stats is None:
            result['status'] = 'drop'
            result['drop_reason'] = 'no_notes'
        else:
            result.update(note_stats)
            # Convert programs list to string
            result['programs'] = ','.join(map(str, result['programs']))
    
    except Exception as e:
        result['status'] = 'fail'
        result['drop_reason'] = 'parse_error'
        result['error_msg'] = repr(e)
        logger.error(f"Failed to parse {path.name}: {e}")
    
    return result


def run_stage_1(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 1: Parse MIDI files and extract metadata.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with stage statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Parsing MIDI Files")
    logger.info("=" * 80)
    
    # Find all MIDI files in raw directory (recursively)
    midi_files = []
    for ext in MIDI_EXTENSIONS:
        midi_files.extend(config.dirs['raw'].rglob(ext))
    
    midi_files = sorted(midi_files)
    logger.info(f"Found {len(midi_files)} MIDI files in {config.dirs['raw']}")
    
    if len(midi_files) == 0:
        logger.warning("No MIDI files found!")
        return {'total': 0, 'ok': 0, 'fail': 0, 'drop': 0}
    
    # Parse each file
    results = []
    for i, path in enumerate(midi_files, 1):
        result = parse_midi_file(path, config.dirs['raw'])
        results.append(result)
        
        if i % 100 == 0:
            logger.info(f"Parsed {i}/{len(midi_files)} files...")
    
    # Check for duplicate file_ids
    file_ids = [r['file_id'] for r in results]
    unique_ids = set(file_ids)
    if len(file_ids) != len(unique_ids):
        logger.warning(f"Found {len(file_ids) - len(unique_ids)} duplicate file_ids!")
        # Find duplicates
        from collections import Counter
        id_counts = Counter(file_ids)
        duplicates = {fid: count for fid, count in id_counts.items() if count > 1}
        logger.warning(f"Duplicate file_ids: {list(duplicates.keys())[:10]}...")
    
    # Create/update manifest
    if manifest.df is None or len(manifest.df) == 0:
        # Create new manifest
        manifest.df = pd.DataFrame(results)
        logger.info(f"Created new manifest with {len(results)} files")
    else:
        # Update existing manifest using file_id as key (not basename)
        updates = {r['file_id']: r for r in results}
        manifest.update_rows_by_file_id(updates)
        logger.info(f"Updated manifest with {len(results)} files")
    
    # Save manifest
    manifest.save(backup=True)
    
    # Copy successfully parsed files in dev mode
    if config.mode == 'dev' and config.dirs['parsed'].exists():
        logger.info("Dev mode: Copying successfully parsed files...")
        copied = 0
        
        for result in results:
            if result['status'] == 'ok':
                src = Path(result['raw_path'])
                dst = config.dirs['parsed'] / result['raw_basename']
                
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                    copied += 1
        
        logger.info(f"Copied {copied} files to {config.dirs['parsed']}")
    
    # Calculate statistics
    stats_df = pd.DataFrame(results)
    stats = {
        'total': len(results),
        'ok': (stats_df['status'] == 'ok').sum(),
        'fail': (stats_df['status'] == 'fail').sum(),
        'drop': (stats_df['status'] == 'drop').sum(),
    }
    
    logger.info(f"Parsing complete: {stats['ok']} OK, {stats['fail']} FAIL, {stats['drop']} DROP")
    logger.info("")
    
    return stats


if __name__ == '__main__':
    # For testing
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from pipeline_config import get_default_config
    
    config = get_default_config()
    config.setup_directories()
    
    manifest = ManifestManager(config.manifest_path)
    manifest.load()
    
    stats = run_stage_1(config, manifest)
    print(f"\nStage 1 Statistics: {stats}")
