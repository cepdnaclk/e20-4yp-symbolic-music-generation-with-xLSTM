#!/bin/bash

SOURCE_DIR="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/raw/lmd_full"
TARGET_DIR="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/lmd_preprocessed/midi"

# Combine all file lists
cat train.txt valid.txt test.txt > all_files.txt

echo "Building file index (this takes ~3 minutes)..."
# Build hash map - use find with readable files only
declare -A file_map
while IFS= read -r filepath; do
    filename=$(basename "$filepath")
    file_map["$filename"]="$filepath"
done < <(find "$SOURCE_DIR" -type f -name "*.mid" -readable)

echo "Found ${#file_map[@]} readable MIDI files"
echo "Starting copy process..."

# Copy files
found=0
not_found=0
permission_denied=0

while IFS= read -r filename; do
    if [ -n "${file_map[$filename]}" ]; then
        if cp "${file_map[$filename]}" "$TARGET_DIR/" 2>/dev/null; then
            ((found++))
            if [ $((found % 500)) -eq 0 ]; then
                echo "Progress: $found files copied..."
            fi
        else
            echo "Permission denied: $filename"
            ((permission_denied++))
        fi
    else
        echo "Not found: $filename"
        ((not_found++))
    fi
done < all_files.txt

echo ""
echo "======= SUMMARY ======="
echo "Successfully copied: $found"
echo "Permission denied: $permission_denied"
echo "Not found: $not_found"
echo "Total requested: 29940"
echo "======================="
