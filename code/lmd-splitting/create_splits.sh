#!/bin/bash

TOKEN_DIR="tokens"
META_DIR="../../repos/muzic/museformer/data/meta"
OUTPUT_DIR="splits"

mkdir -p "$OUTPUT_DIR"

echo "Creating train/valid/test splits..."

for split in train valid test; do
    echo "Processing $split split..."

    output_file="$OUTPUT_DIR/${split}.txt"
    > "$output_file"  # Clear file if exists

    found=0
    missing=0

    while IFS= read -r filename; do
        token_file="$TOKEN_DIR/${filename}.txt"

        if [ -f "$token_file" ]; then
            # Read the single line and append to output
            cat "$token_file" >> "$output_file"
            ((found++))

            if [ $((found % 1000)) -eq 0 ]; then
                echo "  $split: $found files processed..."
            fi
        else
            ((missing++))
        fi
    done < "$META_DIR/${split}.txt"

    echo "✅ $split: $found songs found, $missing missing"
    echo "   Output: $output_file"
    echo ""
done

echo "Summary:"
wc -l "$OUTPUT_DIR"/*.txt
