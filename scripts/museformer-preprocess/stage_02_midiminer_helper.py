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
    MIDIMINER_TARGET_TRACK, MIDIMINER_NUM_CORES
)

logger = logging.getLogger(__name__)


def run_stage_2_midiminer(config: PipelineConfig) -> dict:
    """
    Run midiminer track separation automatically.
    
    This function:
    1. Creates temp directories for joblib
    2. Builds command to activate conda and run midiminer
    3. Executes the command
    4. Validates output JSON exists
    
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
    logger.info(f"Processing {num_files} MIDI files from {in_dir}")
    logger.info(f"Output directory: {out_dir}")
    
    # Build command to run midiminer with conda activation
    # We need to source conda and activate the environment
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
