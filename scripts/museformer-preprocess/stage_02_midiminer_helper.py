"""
Stage 2: Midiminer Helper

Prepares files for midiminer processing and processes midiminer results.
This stage has two sub-steps:
1. Pre-midiminer: Copy parsed files to midiminer input directory
2. Post-midiminer: Process midiminer JSON results and update manifest
"""

from pathlib import Path
import json
import shutil
import logging
import subprocess
import os
from typing import Dict, Any

import pandas as pd

from manifest_utils import ManifestManager
from pipeline_config import (
    PipelineConfig, STAGE_NAMES, MIDIMINER_JSON_FILENAME, MIDIMINER_JSON_PATH,
    MIDIMINER_SCRIPT, MIDIMINER_CONDA_ENV, MIDIMINER_TEMP_DIR, MIDIMINER_JOBLIB_TEMP,
    MIDIMINER_TARGET_TRACK, MIDIMINER_NUM_CORES,
    MIDIMINER_BATCH_SIZE, MIDIMINER_ENABLE_BATCHING
)

logger = logging.getLogger(__name__)


def should_use_batching(num_files: int) -> bool:
    """
    Determine if batch processing should be used.
    
    Args:
        num_files: Total number of files to process
    
    Returns:
        True if batching should be used
    """
    if MIDIMINER_ENABLE_BATCHING is None:
        # Auto-detect: use batching if files > batch size
        return num_files > MIDIMINER_BATCH_SIZE
    else:
        return MIDIMINER_ENABLE_BATCHING


def get_file_batches(input_dir: Path, batch_size: int) -> list:
    """
    Split MIDI files into batches.
    
    Args:
        input_dir: Directory containing MIDI files
        batch_size: Number of files per batch
    
    Returns:
        List of lists, where each inner list contains Path objects for a batch
    """
    # Get all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(list(input_dir.glob(ext)))
    
    # Sort for consistent ordering
    midi_files.sort()
    
    # Split into batches
    batches = []
    for i in range(0, len(midi_files), batch_size):
        batch = midi_files[i:i + batch_size]
        batches.append(batch)
    
    return batches


def run_midiminer_batch(
    batch_files: list,
    batch_num: int,
    total_batches: int,
    config: PipelineConfig
) -> dict:
    """
    Run midiminer on a single batch of files.
    
    Args:
        batch_files: List of Path objects for this batch
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches
        config: Pipeline configuration
    
    Returns:
        Dict with batch results
    """
    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
    
    # Create temporary batch directory
    batch_temp_dir = MIDIMINER_TEMP_DIR / f"batch_{batch_num:03d}"
    batch_input_dir = batch_temp_dir / "input"
    batch_output_dir = batch_temp_dir / "output"
    
    batch_input_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get main output directory to check for already-processed files
    main_output_dir = config.dirs['midiminer_output']
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Filter out files that have already been processed
        # Check if output exists in MAIN output directory (not temp)
        files_to_process = []
        files_already_processed = 0
        
        for src_file in batch_files:
            # Check if this file has already been processed in the main output directory
            output_file = main_output_dir / src_file.name
            if output_file.exists():
                files_already_processed += 1
            else:
                files_to_process.append(src_file)
        
        logger.info(f"  Files in batch: {len(batch_files)}")
        logger.info(f"  Already processed: {files_already_processed}")
        logger.info(f"  To process: {len(files_to_process)}")
        
        # If all files already processed, skip this batch
        if len(files_to_process) == 0:
            logger.info(f"  ✓ Batch {batch_num} already complete, skipping")
            
            # Still need to return the results for these files
            # Read existing files to build the file_programs dict
            batch_file_programs = {}
            # We can't easily reconstruct the JSON here, so return empty
            # The final merge will handle this
            return {
                'status': 'success',
                'batch': batch_num,
                'file_programs': {},  # Empty since already merged
                'files_processed': 0,
                'files_with_melody': 0,
                'skipped': True
            }
        
        # Copy batch files to temp input directory (only files that need processing)
        logger.info(f"  Copying {len(files_to_process)} files to batch input directory...")
        for src_file in files_to_process:
            dst_file = batch_input_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        
        # Build midiminer command for this batch
        cmd = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \\
conda activate {MIDIMINER_CONDA_ENV} && \\
export TMPDIR={MIDIMINER_TEMP_DIR} && \\
export JOBLIB_TEMP_FOLDER={MIDIMINER_JOBLIB_TEMP} && \\
cd {MIDIMINER_SCRIPT.parent} && \\
python {MIDIMINER_SCRIPT.name} \\
    -i {batch_input_dir} \\
    -o {batch_output_dir} \\
    -t {MIDIMINER_TARGET_TRACK} \\
    -c {MIDIMINER_NUM_CORES}
"""
        
        logger.info(f"  Running midiminer on batch {batch_num}...")
        
        # Run the command
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"  Batch {batch_num} completed successfully")
        
        # Read batch results JSON
        batch_json = batch_output_dir / MIDIMINER_JSON_FILENAME
        if not batch_json.exists():
            logger.error(f"  Batch {batch_num} JSON not found: {batch_json}")
            return {'status': 'error', 'error': 'batch_json_not_found', 'batch': batch_num}
        
        with open(batch_json, 'r') as f:
            batch_results = json.load(f)
        
        # Move processed files from batch output to main output
        main_output_dir = config.dirs['midiminer_output']
        main_output_dir.mkdir(parents=True, exist_ok=True)
        
        files_moved = 0
        for output_file in batch_output_dir.glob('*.mid'):
            dst_file = main_output_dir / output_file.name
            shutil.move(str(output_file), str(dst_file))
            files_moved += 1
        
        logger.info(f"  Moved {files_moved} processed files to main output directory")
        logger.info(f"  Batch {batch_num}: {len(batch_results)} files with melody")
        
        return {
            'status': 'success',
            'batch': batch_num,
            'file_programs': batch_results,
            'files_processed': len(batch_files),
            'files_with_melody': len(batch_results)
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"  Batch {batch_num} failed with exit code {e.returncode}")
        logger.error(f"  STDERR: {e.stderr[-500:]}")  # Last 500 chars
        return {
            'status': 'error',
            'error': str(e),
            'batch': batch_num,
            'returncode': e.returncode
        }
    
    except Exception as e:
        logger.error(f"  Batch {batch_num} failed with exception: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'batch': batch_num
        }
    
    finally:
        # Clean up batch temp directory
        if batch_temp_dir.exists():
            shutil.rmtree(batch_temp_dir)
            logger.info(f"  Cleaned up batch {batch_num} temp directory")


def run_stage_2_midiminer(config: PipelineConfig) -> dict:
    """
    Run midiminer track separation automatically.
    
    Supports batch processing for large datasets to prevent OOM crashes.
    
    This function:
    1. Creates temp directories for joblib
    2. Determines if batch processing is needed
    3. If batching: processes files in batches and merges results
    4. If not batching: processes all files at once
    5. Validates output JSON exists
    
    Args:
        config: Pipeline configuration
    
    Returns:
        Dict with execution statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 2 MIDIMINER: Running track separation")
    logger.info("=" * 80)
    
    # Create temp directories
    MIDIMINER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    MIDIMINER_JOBLIB_TEMP.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temp directories:")
    logger.info(f"  TMPDIR: {MIDIMINER_TEMP_DIR}")
    logger.info(f"  JOBLIB_TEMP_FOLDER: {MIDIMINER_JOBLIB_TEMP}")
    
    # Input and output directories
    in_dir = config.dirs['midiminer_input']
    out_dir = config.dirs['midiminer_output']
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Count input files
    num_files = len(list(in_dir.glob('*.mid'))) + len(list(in_dir.glob('*.midi')))
    logger.info(f"Total files to process: {num_files}")
    logger.info(f"Input directory: {in_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("")
    
    # Determine if batch processing should be used
    use_batching = should_use_batching(num_files)
    
    if use_batching:
        logger.info(f"✓ Batch processing ENABLED")
        logger.info(f"  Batch size: {MIDIMINER_BATCH_SIZE} files")
        logger.info(f"  Total batches: {(num_files + MIDIMINER_BATCH_SIZE - 1) // MIDIMINER_BATCH_SIZE}")
        logger.info("")
        
        # Get file batches
        batches = get_file_batches(in_dir, MIDIMINER_BATCH_SIZE)
        total_batches = len(batches)
        
        logger.info(f"Split {num_files} files into {total_batches} batches")
        logger.info("")
        
        # Process each batch
        all_file_programs = {}
        batch_stats = []
        failed_batches = []
        skipped_batches = []
        
        for batch_num, batch_files in enumerate(batches, 1):
            logger.info("=" * 80)
            batch_result = run_midiminer_batch(
                batch_files, batch_num, total_batches, config
            )
            
            if batch_result['status'] == 'success':
                # Check if batch was skipped
                if batch_result.get('skipped', False):
                    skipped_batches.append(batch_num)
                    logger.info(f"⊘ Batch {batch_num} skipped (already processed)")
                else:
                    # Merge batch results
                    all_file_programs.update(batch_result['file_programs'])
                    batch_stats.append({
                        'batch': batch_num,
                        'files_processed': batch_result['files_processed'],
                        'files_with_melody': batch_result['files_with_melody']
                    })
                    logger.info(f"✓ Batch {batch_num} completed: {batch_result['files_with_melody']} files with melody")
            else:
                failed_batches.append(batch_num)
                logger.error(f"✗ Batch {batch_num} FAILED: {batch_result.get('error')}")
            
            logger.info("")
        
        # Save final merged JSON
        final_json = out_dir / MIDIMINER_JSON_FILENAME
        with open(final_json, 'w') as f:
            json.dump(all_file_programs, f, indent=2, ensure_ascii=False)
        
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Processed batches: {len(batch_stats)}")
        logger.info(f"Skipped batches: {len(skipped_batches)}")
        if skipped_batches:
            logger.info(f"  Skipped batch numbers: {skipped_batches}")
        logger.info(f"Failed batches: {len(failed_batches)}")
        if failed_batches:
            logger.info(f"  Failed batch numbers: {failed_batches}")
        logger.info(f"Total files with melody: {len(all_file_programs)}")
        logger.info(f"Final JSON: {final_json}")
        logger.info("")
        
        if failed_batches:
            return {
                'status': 'partial_success',
                'num_files': num_files,
                'num_batches': total_batches,
                'processed_batches': len(batch_stats),
                'skipped_batches': skipped_batches,
                'failed_batches': failed_batches,
                'files_with_melody': len(all_file_programs),
                'batch_stats': batch_stats
            }
        else:
            return {
                'status': 'success',
                'num_files': num_files,
                'num_batches': total_batches,
                'processed_batches': len(batch_stats),
                'skipped_batches': skipped_batches,
                'files_with_melody': len(all_file_programs),
                'batch_stats': batch_stats
            }
    
    else:
        # Single-run processing (original logic)
        logger.info(f"✓ Single-run processing (batch processing not needed)")
        logger.info("")
        
        # Build command to run midiminer with conda activation
        cmd = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \\
conda activate {MIDIMINER_CONDA_ENV} && \\
export TMPDIR={MIDIMINER_TEMP_DIR} && \\
export JOBLIB_TEMP_FOLDER={MIDIMINER_JOBLIB_TEMP} && \\
cd {MIDIMINER_SCRIPT.parent} && \\
python {MIDIMINER_SCRIPT.name} \\
    -i {in_dir} \\
    -o {out_dir} \\
    -t {MIDIMINER_TARGET_TRACK} \\
    -c {MIDIMINER_NUM_CORES}
"""
        
        logger.info("Executing midiminer command:")
        logger.info(f"  Conda env: {MIDIMINER_CONDA_ENV}")
        logger.info(f"  Script: {MIDIMINER_SCRIPT}")
        logger.info(f"  Target track: {MIDIMINER_TARGET_TRACK}")
        logger.info(f"  Cores: {MIDIMINER_NUM_CORES}")
        logger.info("")
        
        try:
            # Run the command
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Midiminer execution completed successfully")
            
            # Log output (last 20 lines)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                logger.info("Midiminer output (last 20 lines):")
                for line in lines[-20:]:
                    logger.info(f"  {line}")
            
            # Validate output JSON exists
            output_json = out_dir / MIDIMINER_JSON_FILENAME
            if output_json.exists():
                logger.info(f"✓ Output JSON created: {output_json}")
                file_size = output_json.stat().st_size
                logger.info(f"  Size: {file_size / 1024:.1f} KB")
            else:
                logger.error(f"✗ Output JSON not found: {output_json}")
                return {'status': 'error', 'error': 'output_json_not_found'}
            
            logger.info("")
            return {'status': 'success', 'num_files': num_files}
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Midiminer execution failed with exit code {e.returncode}")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            return {'status': 'error', 'error': str(e), 'returncode': e.returncode}
        
        except Exception as e:
            logger.error(f"Unexpected error running midiminer: {e}")
            return {'status': 'error', 'error': str(e)}




def run_stage_2_pre(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 2 Pre: Prepare files for midiminer.
    
    Copies successfully parsed files to midiminer input directory.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 2 PRE: Preparing files for midiminer")
    logger.info("=" * 80)
    
    # Query files that were successfully parsed
    parsed_ok = manifest.query_by_stage(STAGE_NAMES['parsing'], status='ok')
    logger.info(f"Found {len(parsed_ok)} successfully parsed files")
    
    if len(parsed_ok) == 0:
        logger.warning("No files to process!")
        return {'copied': 0}
    
    # Create midiminer input directory
    config.dirs['midiminer_input'].mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copied = 0
    skipped = 0
    
    for _, row in parsed_ok.iterrows():
        src = Path(row['raw_path'])
        dst = config.dirs['midiminer_input'] / row['raw_basename']
        
        if dst.exists():
            skipped += 1
            continue
        
        if not src.exists():
            logger.warning(f"Source file not found: {src}")
            continue
        
        shutil.copy2(src, dst)
        copied += 1
    
    logger.info(f"Copied {copied} files to {config.dirs['midiminer_input']}")
    logger.info(f"Skipped {skipped} files (already exist)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("MANUAL STEP REQUIRED:")
    logger.info("=" * 80)
    logger.info("1. Activate midiminer conda environment:")
    logger.info("   conda activate midiminer")
    logger.info("")
    logger.info("2. Run midiminer:")
    logger.info(f"   cd {config.dirs['midiminer_input'].parent.parent / 'repos' / 'midi-miner'}")
    logger.info(f"   python track_separate.py \\")
    logger.info(f"     --input_dir {config.dirs['midiminer_input']} \\")
    logger.info(f"     --output_json {config.dirs['midiminer_input'] / MIDIMINER_JSON_FILENAME}")
    logger.info("")
    logger.info("3. After midiminer completes, run:")
    logger.info("   python pipeline_main.py --stage 2-post")
    logger.info("=" * 80)
    logger.info("")
    
    return {'copied': copied, 'skipped': skipped}


def run_stage_2_post(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 2 Post: Process midiminer results.
    
    Reads midiminer JSON, filters files with melody, and updates manifest.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 2 POST: Processing midiminer results")
    logger.info("=" * 80)
    
    # Check if midiminer JSON exists
    #midiminer_json_path = config.dirs['midiminer_extract'] / MIDIMINER_JSON_FILENAME
    midiminer_json_path = MIDIMINER_JSON_PATH
    
    if not midiminer_json_path.exists():
        logger.error(f"Midiminer JSON not found: {midiminer_json_path}")
        logger.error("Please run midiminer first (see stage 2-pre instructions)")
        return {'error': 'midiminer_json_not_found'}
    
    # Load midiminer results
    with midiminer_json_path.open('r') as f:
        midiminer_results = json.load(f)
    
    logger.info(f"Loaded midiminer results: {len(midiminer_results)} entries")
    
    # Map basename -> roles (only if melody exists)
    basename_to_roles = {}
    for path_str, roles in midiminer_results.items():
        basename = Path(path_str).name
        if isinstance(roles, dict) and 'melody' in roles:
            basename_to_roles[basename] = roles
    
    logger.info(f"Files with melody: {len(basename_to_roles)}")
    
    # Update manifest
    updates = {}
    
    # Get all parsed files
    parsed_files = manifest.query_by_stage(STAGE_NAMES['parsing'], status='ok')
    
    for _, row in parsed_files.iterrows():
        basename = row['raw_basename']
        roles = basename_to_roles.get(basename)
        
        update = {
            'stage': STAGE_NAMES['midiminer'],
            'midiminer_stage': STAGE_NAMES['midiminer'],
        }
        
        if roles is not None:
            # Has melody
            update['status'] = 'ok'
            update['drop_reason'] = pd.NA
            update['midiminer_has_melody'] = True
            update['midiminer_melody_program'] = roles.get('melody', pd.NA)
            update['midiminer_bass_program'] = roles.get('bass', pd.NA)
            update['midiminer_drum_program'] = roles.get('drum', pd.NA)
            update['midiminer_roles_json'] = json.dumps(roles, sort_keys=True)
        else:
            # No melody
            update['status'] = 'drop'
            update['drop_reason'] = 'no_melody_midiminer'
            update['midiminer_has_melody'] = False
            update['midiminer_melody_program'] = pd.NA
            update['midiminer_bass_program'] = pd.NA
            update['midiminer_drum_program'] = pd.NA
            update['midiminer_roles_json'] = pd.NA
        
        updates[basename] = update
    
    manifest.update_rows(updates)
    manifest.save(backup=True)
    
    # Calculate statistics
    has_melody = sum(1 for u in updates.values() if u.get('midiminer_has_melody'))
    no_melody = len(updates) - has_melody
    
    logger.info(f"Updated manifest: {has_melody} with melody, {no_melody} without melody")
    
    # Cleanup midiminer input directory in prod mode
    if config.mode == 'prod' and config.dirs['midiminer_input'].exists():
        logger.info("Prod mode: Cleaning up midiminer input directory...")
        shutil.rmtree(config.dirs['midiminer_input'])
        logger.info(f"Deleted {config.dirs['midiminer_input']}")
    
    logger.info("")
    
    return {
        'total': len(updates),
        'has_melody': has_melody,
        'no_melody': no_melody,
    }


if __name__ == '__main__':
    # For testing
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['pre', 'post'], required=True,
                       help='Which step to run: pre or post')
    args = parser.parse_args()
    
    from pipeline_config import get_default_config
    
    config = get_default_config()
    config.setup_directories()
    
    manifest = ManifestManager(config.manifest_path)
    manifest.load()
    
    if args.step == 'pre':
        stats = run_stage_2_pre(config, manifest)
    else:
        stats = run_stage_2_post(config, manifest)
    
    print(f"\nStage 2 {args.step.upper()} Statistics: {stats}")
