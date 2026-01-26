#!/bin/bash

# MuseScore MIDI Batch Normalization Script (IMPROVED)
# Imports MIDI files with custom options, exports normalized MIDI,
# and fixes track naming inconsistencies
# 
# Usage:
#   ./batch_normalize_midi_improved.sh          # Normal mode (skip existing files)
#   ./batch_normalize_midi_improved.sh --force  # Force mode (reprocess all files)

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

FORCE_REPROCESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_REPROCESS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f    Force reprocessing of all files (ignore existing outputs)"
            echo "  --help, -h     Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MSCORE="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/musescore/MuseScore-Studio-4.6.5.253511702-x86_64.AppImage"
OPS="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/musescore/midi_import_options.xml"
INPUT_DIR="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/03_compressed6"
OUTPUT_DIR="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/sample/04_musescore_norm"
LOG_DIR="$OUTPUT_DIR/logs"

# Track name fixing (set to true to enable post-processing)
FIX_TRACK_NAMES=true
PYTHON_SCRIPT="$OUTPUT_DIR/fix_track_names.py"

# Timeout per file (seconds)
TIMEOUT=60

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

export SKIP_LIBJACK=1
export QT_QPA_PLATFORM=offscreen
export XDG_CONFIG_HOME="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/musescore/.config"
export XDG_DATA_HOME="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/musescore/.local/share"
export XDG_CACHE_HOME="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/musescore/.cache"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"

# ============================================================================
# CREATE TRACK NAME FIXING SCRIPT
# ============================================================================

if [ "$FIX_TRACK_NAMES" = true ]; then
    cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Fix track names in MuseScore-normalized MIDI files

MuseScore sometimes creates track names like:
  - "Square Synthesizer, square_synth" instead of "square_synth"
  - "Piano, piano" instead of "piano"

This script fixes these by extracting the canonical name (after comma).
"""

import sys
from pathlib import Path
import miditoolkit


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
        # Take the last part (canonical name)
        canonical = parts[-1].strip()
        return canonical
    
    return name


def fix_midi_file(input_path: Path, output_path: Path = None) -> bool:
    """
    Fix track names in a MIDI file.
    
    Args:
        input_path: Path to input MIDI file
        output_path: Path to output MIDI file (overwrites input if None)
    
    Returns:
        True if any changes were made, False otherwise
    """
    if output_path is None:
        output_path = input_path
    
    try:
        # Load MIDI
        midi = miditoolkit.MidiFile(str(input_path))
        
        # Track changes
        changes_made = False
        changes = []
        
        # Fix each instrument name
        for inst in midi.instruments:
            original_name = inst.name
            fixed_name = fix_track_name(original_name)
            
            if original_name != fixed_name:
                inst.name = fixed_name
                changes_made = True
                changes.append(f"{original_name} -> {fixed_name}")
        
        # Save if changes were made
        if changes_made:
            midi.dump(str(output_path))
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_track_names.py <midi_file>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Fix in place
    if fix_midi_file(input_path):
        print("FIXED")
        sys.exit(0)
    else:
        print("OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
PYTHON_EOF

    chmod +x "$PYTHON_SCRIPT"
    echo "Created track name fixing script: $PYTHON_SCRIPT"
fi

# ============================================================================
# VALIDATION
# ============================================================================

echo "==================================================================="
echo "MuseScore MIDI Batch Normalization (IMPROVED)"
echo "==================================================================="
echo ""

# Check if MuseScore exists
if [ ! -f "$MSCORE" ]; then
    echo "ERROR: MuseScore AppImage not found at: $MSCORE"
    exit 1
fi

# Check if options XML exists
if [ ! -f "$OPS" ]; then
    echo "ERROR: MIDI import options XML not found at: $OPS"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check if input directory has MIDI files
midi_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.mid" -type f | wc -l)
if [ "$midi_count" -eq 0 ]; then
    echo "ERROR: No .mid files found in: $INPUT_DIR"
    exit 1
fi

# Check if Python is available for fixing
if [ "$FIX_TRACK_NAMES" = true ]; then
    if ! command -v python3 &> /dev/null; then
        echo "WARNING: python3 not found, disabling track name fixing"
        FIX_TRACK_NAMES=false
    else
        # Check if miditoolkit is available
        if ! python3 -c "import miditoolkit" 2>/dev/null; then
            echo "WARNING: miditoolkit not found, disabling track name fixing"
            echo "  Install with: pip install miditoolkit --break-system-packages"
            FIX_TRACK_NAMES=false
        fi
    fi
fi

echo "Configuration:"
echo "  MuseScore: $MSCORE"
echo "  MIDI Options: $OPS"
echo "  Input Dir: $INPUT_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Log Dir: $LOG_DIR"
echo "  Files to process: $midi_count"
echo "  Timeout per file: ${TIMEOUT}s"
echo "  Force reprocess: $FORCE_REPROCESS"
echo "  Fix track names: $FIX_TRACK_NAMES"
echo ""

# ============================================================================
# BATCH PROCESSING
# ============================================================================

total=0
success=0
failed=0
skipped=0
fixed=0

start_time=$(date +%s)

echo "Starting batch conversion: $(date)"
echo "-------------------------------------------------------------------"
echo ""

for input_file in "$INPUT_DIR"/*.mid; do
    # Skip if no files found
    [[ -e "$input_file" ]] || continue
    
    total=$((total + 1))
    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$filename"
    log_file="$LOG_DIR/${filename%.mid}.log"
    
    # Skip if already processed (unless force mode is enabled)
    if [ "$FORCE_REPROCESS" = false ] && [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "[$total] SKIPPED (already exists): $filename"
        skipped=$((skipped + 1))
        continue
    fi
    
    # If force mode and file exists, indicate reprocessing
    if [ "$FORCE_REPROCESS" = true ] && [ -f "$output_file" ]; then
        echo "[$total] REPROCESSING: $filename"
    else
        echo "[$total] Processing: $filename"
    fi
    
    # Run conversion with timeout
    timeout ${TIMEOUT}s xvfb-run -a --server-args="-screen 0 1024x768x24" \
        "$MSCORE" -M "$OPS" -o "$output_file" "$input_file" > "$log_file" 2>&1
    
    exit_code=$?
    
    # Check results
    if [ $exit_code -eq 0 ] && [ -f "$output_file" ] && [ -s "$output_file" ]; then
        # Success - now try to fix track names
        file_size=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null)
        input_size=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null)
        
        fix_status=""
        if [ "$FIX_TRACK_NAMES" = true ]; then
            fix_result=$(python3 "$PYTHON_SCRIPT" "$output_file" 2>&1)
            if [ "$fix_result" = "FIXED" ]; then
                fix_status=" [FIXED NAMES]"
                fixed=$((fixed + 1))
            fi
        fi
        
        success=$((success + 1))
        echo "  ✓ SUCCESS (${file_size} bytes, input: ${input_size} bytes)${fix_status}"
    else
        # Failure
        failed=$((failed + 1))
        
        # Determine failure reason
        if [ $exit_code -eq 124 ]; then
            reason="TIMEOUT (>${TIMEOUT}s)"
        elif [ $exit_code -eq 139 ]; then
            reason="SEGFAULT"
        elif [ ! -f "$output_file" ]; then
            reason="NO OUTPUT FILE"
        elif [ ! -s "$output_file" ]; then
            reason="EMPTY OUTPUT"
        else
            reason="EXIT CODE $exit_code"
        fi
        
        echo "  ✗ FAILED: $reason"
        echo "     Log: $log_file"
        
        # Show relevant error lines
        if [ -f "$log_file" ]; then
            error_lines=$(grep -i "error\|fail\|warning" "$log_file" | head -3)
            if [ -n "$error_lines" ]; then
                echo "     Errors:"
                echo "$error_lines" | sed 's/^/       /'
            fi
        fi
        
        # Remove empty output files
        if [ -f "$output_file" ] && [ ! -s "$output_file" ]; then
            rm -f "$output_file"
        fi
    fi
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "==================================================================="
echo "Batch Conversion Complete: $(date)"
echo "==================================================================="
echo ""
echo "Summary:"
echo "  Total files found:      $total"
echo "  Successfully converted: $success"
echo "  Failed:                 $failed"
echo "  Skipped (existing):     $skipped"

if [ "$FIX_TRACK_NAMES" = true ]; then
    echo "  Track names fixed:      $fixed"
fi

echo "  Processing time:        ${minutes}m ${seconds}s"

if [ $total -gt 0 ]; then
    success_rate=$(awk "BEGIN {printf \"%.1f\", ($success/($total-$skipped))*100}")
    echo "  Success rate:           ${success_rate}%"
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log directory:    $LOG_DIR"
echo ""

# ============================================================================
# FAILURE REPORT
# ============================================================================

if [ $failed -gt 0 ]; then
    echo "-------------------------------------------------------------------"
    echo "Failed Files:"
    echo "-------------------------------------------------------------------"
    
    for log in "$LOG_DIR"/*.log; do
        [[ -e "$log" ]] || continue
        filename=$(basename "$log" .log)
        output_file="$OUTPUT_DIR/${filename}.mid"
        
        # Check if this file failed
        if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
            echo "  - ${filename}.mid"
            
            # Show last error line
            last_error=$(grep -i "error" "$log" | tail -1)
            if [ -n "$last_error" ]; then
                echo "    Error: $last_error"
            fi
        fi
    done
    echo ""
fi

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

if [ "$FIX_TRACK_NAMES" = true ] && [ $fixed -gt 0 ]; then
    echo "-------------------------------------------------------------------"
    echo "Track Name Fixing Summary:"
    echo "-------------------------------------------------------------------"
    echo ""
    echo "Fixed track names in $fixed files to ensure compatibility with"
    echo "the filtering pipeline. Use these config settings:"
    echo ""
    echo "  TRACK_DETECTION_METHOD = 'program_or_name'"
    echo "  REQUIRED_PROGRAM_NUMBER = 80"
    echo ""
fi

# Exit with error if any files failed
if [ $failed -gt 0 ]; then
    exit 1
else
    exit 0
fi