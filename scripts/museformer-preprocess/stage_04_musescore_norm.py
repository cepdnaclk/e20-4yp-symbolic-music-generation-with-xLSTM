"""
Stage 4: MuseScore Normalization

Python wrapper around MuseScore AppImage for MIDI normalization.
Handles environment setup, subprocess execution, and track name fixing.
"""

from pathlib import Path
import subprocess
import shutil
import os
import tempfile
import logging
from typing import Dict, Any, Optional

import miditoolkit

from manifest_utils import ManifestManager
from pipeline_config import (
    PipelineConfig, STAGE_NAMES,
    MUSESCORE_APPIMAGE, MUSESCORE_IMPORT_OPTIONS, MUSESCORE_ENV,
    MUSESCORE_TIMEOUT, MUSESCORE_FIX_TRACK_NAMES
)

logger = logging.getLogger(__name__)


def fix_track_name(name: str) -> str:
    """
    Fix MuseScore track name formatting.
    
    Examples:
        "Piano, piano" -> "piano"
        "Square Synthesizer, square_synth" -> "square_synth"
        "Violins, string" -> "string"
    """
    if not name:
        return name
    
    # If name contains comma, extract part after comma
    if "," in name:
        parts = name.split(",")
        canonical = parts[-1].strip()
        return canonical
    
    return name


def fix_midi_track_names(midi_path: Path) -> bool:
    """
    Fix track names in a MIDI file in-place.
    
    Args:
        midi_path: Path to MIDI file
    
    Returns:
        True if any changes were made, False otherwise
    """
    try:
        midi = miditoolkit.MidiFile(str(midi_path))
        
        changes_made = False
        for inst in midi.instruments:
            original_name = inst.name
            fixed_name = fix_track_name(original_name)
            
            if original_name != fixed_name:
                inst.name = fixed_name
                changes_made = True
        
        if changes_made:
            midi.dump(str(midi_path))
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error fixing track names in {midi_path.name}: {e}")
        return False


def normalize_with_musescore(
    input_path: Path,
    output_path: Path,
    timeout: int = MUSESCORE_TIMEOUT
) -> Dict[str, Any]:
    """
    Normalize a MIDI file using MuseScore.
    
    Args:
        input_path: Path to input MIDI file
        output_path: Path to output MIDI file
        timeout: Timeout in seconds
    
    Returns:
        Dict with result info: {'success': bool, 'error': str, 'fixed_names': bool}
    """
    result = {'success': False, 'error': '', 'fixed_names': False}
    
    # Check inputs
    if not input_path.exists():
        result['error'] = f"Input file not found: {input_path}"
        return result
    
    if not MUSESCORE_APPIMAGE.exists():
        result['error'] = f"MuseScore AppImage not found: {MUSESCORE_APPIMAGE}"
        return result
    
    if not MUSESCORE_IMPORT_OPTIONS.exists():
        result['error'] = f"MIDI import options not found: {MUSESCORE_IMPORT_OPTIONS}"
        return result
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up environment
    env = os.environ.copy()
    env.update(MUSESCORE_ENV)
    
    # Ensure XDG directories exist
    for key in ['XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME']:
        if key in env:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        'xvfb-run', '-a', '--server-args=-screen 0 1024x768x24',
        str(MUSESCORE_APPIMAGE),
        '-M', str(MUSESCORE_IMPORT_OPTIONS),
        '-o', str(output_path),
        str(input_path)
    ]
    
    try:
        # Run MuseScore with timeout
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check if output file was created
        if output_path.exists() and output_path.stat().st_size > 0:
            result['success'] = True
            
            # Fix track names if enabled
            if MUSESCORE_FIX_TRACK_NAMES:
                if fix_midi_track_names(output_path):
                    result['fixed_names'] = True
        else:
            result['error'] = 'No output file created or empty output'
    
    except subprocess.TimeoutExpired:
        result['error'] = f'Timeout ({timeout}s)'
    except FileNotFoundError:
        result['error'] = 'xvfb-run not found (install xvfb)'
    except Exception as e:
        result['error'] = repr(e)
    
    return result


def run_stage_4(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 4: MuseScore normalization.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with stage statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: MuseScore Normalization")
    logger.info("=" * 80)
    
    
    # Query files from previous stage
    compressed_files = manifest.query_by_stage(STAGE_NAMES['compress'], status='ok')
    logger.info(f"Found {len(compressed_files)} compressed files to normalize")
    
    if len(compressed_files) == 0:
        logger.warning("No files to normalize!")
        return {'total': 0, 'ok': 0, 'fail': 0}
    
    # Create output directory (needed in both dev and prod modes)
    config.dirs['musescore_norm'].mkdir(parents=True, exist_ok=True)
    
    # Process each file
    updates = {}
    stats = {'ok': 0, 'fail': 0, 'timeout': 0, 'fixed_names': 0}
    
    for i, (_, row) in enumerate(compressed_files.iterrows(), 1):
        basename = row['raw_basename']
        
        # Get input path (always use compressed files from Stage 3)
        if 'compressed6_path' in row and row['compressed6_path']:
            input_path = Path(row['compressed6_path'])
        else:
            input_path = config.dirs['compressed'] / basename
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            continue
        
        # Determine output path (same for both modes)
        output_path = config.dirs['musescore_norm'] / basename
        
        # Skip if already processed
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug(f"Skipping {basename} (already exists)")
            updates[basename] = {
                'stage': STAGE_NAMES['musescore'],
                'status': 'ok',
                'musescore_norm_path': str(output_path),
            }
            stats['ok'] += 1
            continue
        
        try:
            # Normalize with MuseScore
            result = normalize_with_musescore(input_path, output_path)
            
            if result['success']:
                update_data = {
                    'stage': STAGE_NAMES['musescore'],
                    'status': 'ok',
                    'musescore_fixed_names': result['fixed_names'],
                }
                
                # Save the path (both modes)
                update_data['musescore_norm_path'] = str(output_path)
                
                updates[basename] = update_data
                stats['ok'] += 1
                if result['fixed_names']:
                    stats['fixed_names'] += 1
            else:
                updates[basename] = {
                    'stage': STAGE_NAMES['musescore'],
                    'status': 'fail',
                    'error_msg': result['error'],
                }
                stats['fail'] += 1
                if 'Timeout' in result['error']:
                    stats['timeout'] += 1
                
                logger.error(f"Failed to normalize {basename}: {result['error']}")
                

        
        except Exception as e:
            logger.error(f"Error processing {basename}: {e}")
            updates[basename] = {
                'stage': STAGE_NAMES['musescore'],
                'status': 'fail',
                'error_msg': repr(e),
            }
            stats['fail'] += 1
            
            # Clean up temp file in prod mode
            if config.mode == 'prod' and output_path.exists():
                output_path.unlink()
        
        if i % 50 == 0:
            logger.info(f"Normalized {i}/{len(compressed_files)} files...")
    
    # Update manifest
    manifest.update_rows(updates)
    manifest.save(backup=True)
    
    logger.info(f"Normalization complete: {stats['ok']} OK, {stats['fail']} FAIL")
    logger.info(f"  Timeouts: {stats['timeout']}, Fixed names: {stats['fixed_names']}")
    
    # Cleanup previous stage's files in production mode
    if config.mode == 'prod':
        compressed_dir = config.dirs['compressed']
        if compressed_dir.exists():
            shutil.rmtree(compressed_dir)
            logger.info(f"  Cleaned up compressed files: {compressed_dir}")
    
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
    
    stats = run_stage_4(config, manifest)
    print(f"\nStage 4 Statistics: {stats}")
