import os
import argparse
import random
from pathlib import Path

def generate_splits(data_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """
    Generates train.txt, valid.txt, and test.txt from files in data_dir.
    """
    random.seed(seed)
    
    # Get all files in directory (assuming these are the MIDI files or Token files base names)
    # If the process started with MIDIs, we usually list the MIDI filenames.
    # The subsequent tools expect the names in these lists to match the filenames 
    # that were encoded (e.g., 'song.mid' -> looks for 'song.mid.txt' in token dir).
    
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files.sort() # Ensure deterministic order before shuffle
    random.shuffle(files)
    
    total_files = len(files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    # Test gets the rest
    
    train_files = files[:train_count]
    valid_files = files[train_count:train_count+valid_count]
    test_files = files[train_count+valid_count:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train.txt': train_files,
        'valid.txt': valid_files,
        'test.txt': test_files
    }
    
    print(f"Found {total_files} files in {data_dir}")
    for filename, file_list in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            for line in file_list:
                f.write(line + '\n')
        print(f"Created {filename}: {len(file_list)} files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate train/valid/test splits for MuseFormer.")
    parser.add_argument('data_dir', help="Directory containing the source files (e.g., raw MIDI files or token files)")
    parser.add_argument('--output_dir', default='data/meta', help="Directory to save the split files (default: data/meta)")
    parser.add_argument('--train', type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument('--valid', type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    generate_splits(args.data_dir, args.output_dir, args.train, args.valid, args.seed)
