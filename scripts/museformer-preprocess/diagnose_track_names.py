"""
MIDI Track Name Diagnostic Tool

This script analyzes MuseScore-normalized MIDI files to identify
track naming inconsistencies and suggest corrections.

Usage:
    python diagnose_track_names.py <directory>
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
import miditoolkit


def analyze_track_names(midi_dir: Path):
    """Analyze all MIDI files in a directory for track naming patterns"""
    
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    print(f"Analyzing {len(midi_files)} MIDI files...")
    print("=" * 80)
    print()
    
    # Statistics
    track_names = Counter()
    track_programs = Counter()
    track_name_to_program = defaultdict(set)
    files_with_issues = []
    
    # Analyze each file
    for midi_file in midi_files:
        try:
            midi = miditoolkit.MidiFile(str(midi_file))
            
            file_has_issue = False
            
            for inst in midi.instruments:
                if inst.is_drum:
                    continue
                
                name = inst.name or "(unnamed)"
                program = inst.program
                
                track_names[name] += 1
                track_programs[program] += 1
                track_name_to_program[name].add(program)
                
                # Check for problematic patterns
                if "," in name:
                    file_has_issue = True
            
            if file_has_issue:
                files_with_issues.append(midi_file.name)
        
        except Exception as e:
            print(f"Error analyzing {midi_file.name}: {e}")
    
    # Report findings
    print("TRACK NAME PATTERNS")
    print("-" * 80)
    print(f"{'Track Name':<40} {'Count':<10} {'Programs'}")
    print("-" * 80)
    
    for name, count in track_names.most_common(20):
        programs = sorted(track_name_to_program[name])
        programs_str = ", ".join(str(p) for p in programs[:5])
        if len(programs) > 5:
            programs_str += f", ... ({len(programs)} total)"
        print(f"{name:<40} {count:<10} {programs_str}")
    
    print()
    print("=" * 80)
    print("PROGRAM NUMBER DISTRIBUTION")
    print("-" * 80)
    print(f"{'Program':<10} {'Count':<10} {'Common Name'}")
    print("-" * 80)
    
    program_names = {
        0: "Acoustic Grand Piano",
        24: "Acoustic Guitar (nylon)",
        25: "Acoustic Guitar (steel)",
        32: "Acoustic Bass",
        33: "Electric Bass (finger)",
        40: "Violin",
        41: "Viola",
        42: "Cello",
        43: "Contrabass",
        48: "String Ensemble 1",
        80: "Lead 1 (square)",
        81: "Lead 2 (sawtooth)",
    }
    
    for program, count in sorted(track_programs.items()):
        name = program_names.get(program, "")
        print(f"{program:<10} {count:<10} {name}")
    
    print()
    print("=" * 80)
    print("PROBLEMATIC FILES (with commas in track names)")
    print("-" * 80)
    
    if files_with_issues:
        for i, filename in enumerate(files_with_issues[:20], 1):
            print(f"{i:3}. {filename}")
        
        if len(files_with_issues) > 20:
            print(f"... and {len(files_with_issues) - 20} more files")
        
        print()
        print(f"Total files with issues: {len(files_with_issues)} ({len(files_with_issues)/len(midi_files)*100:.1f}%)")
    else:
        print("No problematic files found!")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Check if square_synth exists
    square_synth_patterns = [name for name in track_names if "square" in name.lower()]
    
    if square_synth_patterns:
        print("✓ Found square synth track patterns:")
        for pattern in square_synth_patterns:
            print(f"  - {pattern} (program: {sorted(track_name_to_program[pattern])})")
        print()
    
    # Check for program 80 (square wave)
    if 80 in track_programs or 81 in track_programs:
        print("✓ Found lead synth programs (80/81 - square/sawtooth wave)")
        print("  Recommendation: Use TRACK_DETECTION_METHOD = 'program' in config")
        print("  This is the most reliable method for MuseScore files")
    else:
        print("⚠ No lead synth programs (80/81) found")
        print("  Your MuseScore normalization may need adjustment")
    
    print()
    
    if files_with_issues:
        print("⚠ Found files with inconsistent track naming (e.g., 'Piano, piano')")
        print("  Recommendation: Use TRACK_DETECTION_METHOD = 'program_or_name'")
        print("  This handles both program matching and flexible name matching")
    
    print()
    print("CONFIGURATION SUGGESTIONS")
    print("-" * 80)
    print()
    print("# Add to filter_config.py:")
    print()
    print("# For most reliable detection with MuseScore files:")
    print("TRACK_DETECTION_METHOD = 'program_or_name'")
    print("REQUIRED_PROGRAM_NUMBER = 80  # Lead 1 (square)")
    print("REQUIRED_PROGRAM_RANGE = [80, 81]  # Include sawtooth variant")
    print()


def show_file_details(midi_path: Path):
    """Show detailed track information for a single MIDI file"""
    
    print(f"Analyzing: {midi_path.name}")
    print("=" * 80)
    print()
    
    try:
        midi = miditoolkit.MidiFile(str(midi_path))
        
        print(f"Ticks per beat: {midi.ticks_per_beat}")
        print(f"Number of tracks: {len(midi.instruments)}")
        print()
        
        print("TRACK DETAILS")
        print("-" * 80)
        print(f"{'#':<3} {'Type':<6} {'Name':<35} {'Program':<8} {'Notes'}")
        print("-" * 80)
        
        for i, inst in enumerate(midi.instruments, 1):
            track_type = "DRUM" if inst.is_drum else "INST"
            name = inst.name or "(unnamed)"
            program = "-" if inst.is_drum else str(inst.program)
            notes = len(inst.notes)
            
            # Truncate long names
            if len(name) > 34:
                name = name[:31] + "..."
            
            print(f"{i:<3} {track_type:<6} {name:<35} {program:<8} {notes}")
        
        print()
        
        # Check for issues
        issues = []
        for inst in midi.instruments:
            if not inst.is_drum and "," in inst.name:
                issues.append(f"Track '{inst.name}' has comma (may cause matching issues)")
        
        if issues:
            print("ISSUES FOUND")
            print("-" * 80)
            for issue in issues:
                print(f"⚠ {issue}")
            print()
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_track_names.py <midi_directory_or_file>")
        print()
        print("Examples:")
        print("  python diagnose_track_names.py data/musescore_normalized")
        print("  python diagnose_track_names.py data/musescore_normalized/song.mid")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file():
        # Analyze single file
        show_file_details(path)
    elif path.is_dir():
        # Analyze directory
        analyze_track_names(path)
    else:
        print(f"Error: Not a file or directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()