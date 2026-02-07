"""
Stage 3: Compress to 6 Tracks

Compresses MIDI files to 6 instrument categories using midiminer roles:
- square_synth (melody)
- piano
- guitar
- string
- bass
- drum

Extracted from museformer_preprocess.ipynb notebook.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

import miditoolkit

from manifest_utils import ManifestManager
from pipeline_config import (
    PipelineConfig, STAGE_NAMES, CANONICAL_PROGRAMS,
    FAMILY_RANGES, POLYPHONY_CAP, KEEP_TOP_K_PER_FAMILY,
    MIDIMINER_JSON_FILENAME, MIDIMINER_JSON_PATH
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMPRESSION UTILITIES (from notebook)
# =============================================================================

def note_stats(notes: List) -> Dict[str, Any]:
    """Calculate statistics for a list of notes."""
    if not notes:
        return {"n": 0, "mean_pitch": None}
    pitches = [n.pitch for n in notes]
    return {"n": len(notes), "mean_pitch": sum(pitches) / len(pitches)}


def pick_best_melody(cands: List[Tuple]) -> Optional[Any]:
    """Pick best melody instrument from candidates."""
    if not cands:
        return None
    # Take top-3 by note count, pick the one with highest mean pitch
    cands = sorted(cands, key=lambda x: x[1]["n"], reverse=True)[:3]
    cands = sorted(
        cands,
        key=lambda x: (x[1]["mean_pitch"] if x[1]["mean_pitch"] is not None else -1),
        reverse=True
    )
    return cands[0][0]


def pick_best_bass(cands: List[Tuple]) -> Optional[Any]:
    """Pick best bass instrument from candidates."""
    if not cands:
        return None
    cands = sorted(cands, key=lambda x: x[1]["n"], reverse=True)
    return cands[0][0]


def in_family(program: int, family: str) -> bool:
    """Check if MIDI program belongs to instrument family."""
    ranges = FAMILY_RANGES[family]
    for lo, hi in ranges:
        if lo <= program <= hi:
            return True
    return False


def dedupe_exact(notes: List) -> List:
    """Remove exact duplicate notes, keeping max velocity."""
    best = {}
    for n in notes:
        k = (int(n.start), int(n.end), int(n.pitch))
        if k not in best or n.velocity > best[k].velocity:
            best[k] = n
    out = list(best.values())
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out


def trim_same_pitch_overlaps(notes: List) -> List:
    """Trim overlapping notes of the same pitch."""
    by_pitch = defaultdict(list)
    for n in notes:
        by_pitch[int(n.pitch)].append(n)
    
    out = []
    for p, ns in by_pitch.items():
        ns.sort(key=lambda n: (n.start, n.end))
        prev = None
        for n in ns:
            if prev is None:
                prev = n
                continue
            if n.start < prev.end:
                prev.end = int(n.start)
            if prev.end > prev.start:
                out.append(prev)
            prev = n
        if prev is not None and prev.end > prev.start:
            out.append(prev)
    
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out


def cap_polyphony_by_onset(notes: List, cap: int) -> List:
    """Cap polyphony at each onset."""
    by_start = defaultdict(list)
    for n in notes:
        by_start[int(n.start)].append(n)
    
    out = []
    for s, ns in by_start.items():
        if len(ns) <= cap:
            out.extend(ns)
        else:
            # Keep top by velocity, tie by pitch
            ns.sort(key=lambda n: (n.velocity, n.pitch), reverse=True)
            out.extend(ns[:cap])
    
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out


def enforce_monophony(notes: List, mode: str = "highest") -> List:
    """Enforce monophony by keeping one note per onset."""
    by_start = defaultdict(list)
    for n in notes:
        by_start[int(n.start)].append(n)
    
    starts = sorted(by_start.keys())
    chosen = []
    for s in starts:
        ns = by_start[s]
        if mode == "highest":
            pick = max(ns, key=lambda n: (n.pitch, n.velocity))
        else:
            pick = min(ns, key=lambda n: (n.pitch, -n.velocity))
        chosen.append(pick)
    
    # Trim overlaps between chosen notes
    chosen.sort(key=lambda n: (n.start, n.end))
    out = []
    prev = None
    for n in chosen:
        if prev is None:
            prev = n
            continue
        if n.start < prev.end:
            prev.end = int(n.start)
        if prev.end > prev.start:
            out.append(prev)
        prev = n
    if prev is not None and prev.end > prev.start:
        out.append(prev)
    
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out


def build_empty_output_midi(src_midi: miditoolkit.MidiFile) -> miditoolkit.MidiFile:
    """Create empty MIDI with same metadata as source."""
    out = miditoolkit.MidiFile()
    out.ticks_per_beat = src_midi.ticks_per_beat
    out.tempo_changes = src_midi.tempo_changes
    out.time_signature_changes = src_midi.time_signature_changes
    out.key_signature_changes = getattr(src_midi, "key_signature_changes", [])
    out.markers = getattr(src_midi, "markers", [])
    out.lyrics = getattr(src_midi, "lyrics", [])
    return out


def compress_to_6tracks(
    midi_path: Path,
    roles: Dict[str, int],
    keep_top_k: int = KEEP_TOP_K_PER_FAMILY
) -> Tuple[miditoolkit.MidiFile, Dict[str, Any]]:
    """
    Compress MIDI file to 6 instrument tracks.
    
    Args:
        midi_path: Path to MIDI file
        roles: Midiminer roles dict with 'melody', 'bass', 'drum' programs
        keep_top_k: Number of top instruments to keep per family
    
    Returns:
        Tuple of (compressed MIDI, debug info dict)
    """
    midi = miditoolkit.MidiFile(str(midi_path))
    
    # Separate instruments
    drums = [inst for inst in midi.instruments if inst.is_drum]
    non_drums = [inst for inst in midi.instruments if not inst.is_drum]
    
    # Role programs from midiminer
    mel_prog = roles.get("melody", None)
    bass_prog = roles.get("bass", None)
    
    def collect_candidates(target_prog):
        if target_prog is None:
            return []
        cands_exact = []
        cands_offby1 = []
        for inst in non_drums:
            st = note_stats(inst.notes)
            if inst.program == target_prog:
                cands_exact.append((inst, st))
            elif inst.program == target_prog - 1:
                cands_offby1.append((inst, st))
        return cands_exact if cands_exact else cands_offby1
    
    mel_cands = collect_candidates(mel_prog)
    bass_cands = collect_candidates(bass_prog)
    
    melody_inst = pick_best_melody(mel_cands)
    bass_inst = pick_best_bass(bass_cands)
    
    used_ids = set()
    if melody_inst is not None:
        used_ids.add(id(melody_inst))
    if bass_inst is not None:
        used_ids.add(id(bass_inst))
    for d in drums:
        used_ids.add(id(d))
    
    # Remaining instruments → families
    fam_insts = {"piano": [], "guitar": [], "string": []}
    for inst in non_drums:
        if id(inst) in used_ids:
            continue
        for fam in fam_insts.keys():
            if in_family(inst.program, fam):
                fam_insts[fam].append(inst)
                break
    
    # Select top-K by activity (note count)
    for fam in fam_insts:
        fam_insts[fam] = sorted(
            fam_insts[fam],
            key=lambda inst: len(inst.notes),
            reverse=True
        )[:keep_top_k]
    
    # Gather notes per target
    notes_out = {
        "square_synth": list(melody_inst.notes) if melody_inst else [],
        "bass": list(bass_inst.notes) if bass_inst else [],
        "drum": [],
        "piano": [],
        "guitar": [],
        "string": [],
    }
    for d in drums:
        notes_out["drum"].extend(d.notes)
    for fam in ["piano", "guitar", "string"]:
        for inst in fam_insts[fam]:
            notes_out[fam].extend(inst.notes)
    
    # Clean each track
    def clean_track(name, notes):
        notes = dedupe_exact(notes)
        notes = trim_same_pitch_overlaps(notes)
        if name == "square_synth":
            notes = enforce_monophony(notes, mode="highest")
        elif name == "bass":
            notes = enforce_monophony(notes, mode="lowest")
        elif name in POLYPHONY_CAP:
            notes = cap_polyphony_by_onset(notes, POLYPHONY_CAP[name])
        elif name == "drum":
            pass  # No extra processing for drums
        return notes
    
    for k in list(notes_out.keys()):
        notes_out[k] = clean_track(k, notes_out[k])
    
    # Build output MIDI with exactly 6 instruments (fixed order)
    out = build_empty_output_midi(midi)
    
    insts = []
    # square_synth
    insts.append(miditoolkit.Instrument(
        program=CANONICAL_PROGRAMS["square_synth"],
        is_drum=False,
        name="square_synth"
    ))
    insts[-1].notes = notes_out["square_synth"]
    
    # piano
    insts.append(miditoolkit.Instrument(
        program=CANONICAL_PROGRAMS["piano"],
        is_drum=False,
        name="piano"
    ))
    insts[-1].notes = notes_out["piano"]
    
    # guitar
    insts.append(miditoolkit.Instrument(
        program=CANONICAL_PROGRAMS["guitar"],
        is_drum=False,
        name="guitar"
    ))
    insts[-1].notes = notes_out["guitar"]
    
    # string
    insts.append(miditoolkit.Instrument(
        program=CANONICAL_PROGRAMS["string"],
        is_drum=False,
        name="string"
    ))
    insts[-1].notes = notes_out["string"]
    
    # bass
    insts.append(miditoolkit.Instrument(
        program=CANONICAL_PROGRAMS["bass"],
        is_drum=False,
        name="bass"
    ))
    insts[-1].notes = notes_out["bass"]
    
    # drum
    insts.append(miditoolkit.Instrument(
        program=0,
        is_drum=True,
        name="drum"
    ))
    insts[-1].notes = notes_out["drum"]
    
    out.instruments = insts
    
    debug = {
        "melody_selected_program": mel_prog,
        "bass_selected_program": bass_prog,
        "melody_notes": len(notes_out["square_synth"]),
        "bass_notes": len(notes_out["bass"]),
        "piano_notes": len(notes_out["piano"]),
        "guitar_notes": len(notes_out["guitar"]),
        "string_notes": len(notes_out["string"]),
        "drum_notes": len(notes_out["drum"]),
        "orig_tracks": len(midi.instruments),
    }
    
    return out, debug


# =============================================================================
# STAGE 3 RUNNER
# =============================================================================

def run_stage_3(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 3: Compress MIDI files to 6 tracks.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with stage statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: Compressing to 6 Tracks")
    logger.info("=" * 80)
    
    # Load midiminer JSON
    #midiminer_json_path = config.dirs['midiminer_input'] / MIDIMINER_JSON_FILENAME
    midiminer_json_path = MIDIMINER_JSON_PATH
    
    # Try alternate location if not found
    if not midiminer_json_path.exists():
        midiminer_json_path = config.dirs['logs'] / MIDIMINER_JSON_FILENAME
    
    if not midiminer_json_path.exists():
        logger.error(f"Midiminer JSON not found: {midiminer_json_path}")
        return {'error': 'midiminer_json_not_found'}
    
    with midiminer_json_path.open('r') as f:
        midiminer_results = json.load(f)
    
    # Map basename -> roles
    roles_by_basename = {}
    for path_str, roles in midiminer_results.items():
        basename = Path(path_str).name
        if isinstance(roles, dict):
            roles_by_basename[basename] = roles
    
    logger.info(f"Loaded midiminer results for {len(roles_by_basename)} files")
    
    # Query files with melody from manifest
    files_with_melody = manifest.query_by_stage(STAGE_NAMES['midiminer'], status='ok')
    logger.info(f"Found {len(files_with_melody)} files with melody to compress")
    
    if len(files_with_melody) == 0:
        logger.warning("No files to compress!")
        return {'total': 0, 'ok': 0, 'fail': 0}
    
    # Create output directory (needed in both dev and prod modes)
    config.dirs['compressed'].mkdir(parents=True, exist_ok=True)
    
    # Process each file
    updates = {}
    stats = {'ok': 0, 'fail': 0}
    
    for i, (_, row) in enumerate(files_with_melody.iterrows(), 1):
        basename = row['raw_basename']
        raw_path = Path(row['raw_path'])
        
        roles = roles_by_basename.get(basename)
        if roles is None:
            logger.warning(f"No midiminer roles for {basename}")
            continue
        
        try:
            # Compress MIDI
            compressed_midi, debug = compress_to_6tracks(raw_path, roles)
            
            # Save compressed file (both dev and prod modes)
            out_path = config.dirs['compressed'] / basename
            compressed_midi.dump(str(out_path))
            update = {
                'stage': STAGE_NAMES['compress'],
                'status': 'ok',
                'compressed6_path': str(out_path),
            }
            
            # Add debug info
            update.update({
                f'compress_{k}': v for k, v in debug.items()
            })
            
            updates[basename] = update
            stats['ok'] += 1
            
            if i % 100 == 0:
                logger.info(f"Compressed {i}/{len(files_with_melody)} files...")
        
        except Exception as e:
            logger.error(f"Failed to compress {basename}: {e}")
            updates[basename] = {
                'stage': STAGE_NAMES['compress'],
                'status': 'fail',
                'error_msg': repr(e),
            }
            stats['fail'] += 1
    
    # Update manifest
    manifest.update_rows(updates)
    manifest.save(backup=True)
    
    logger.info(f"Compression complete: {stats['ok']} OK, {stats['fail']} FAIL")
    logger.info("")
    
    return {'total': len(updates), **stats}


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
    
    stats = run_stage_3(config, manifest)
    print(f"\nStage 3 Statistics: {stats}")
