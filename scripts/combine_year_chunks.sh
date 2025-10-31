#!/bin/bash
# Combine year-chunked CSV files into single files per symbol for ML optimization

set -e

# Configuration
CHUNKS_DIR="data/training_chunks"
OUTPUT_DIR="data/ml_training"
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Combine Year Chunks into Single Files${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if chunks directory exists
if [ ! -d "$CHUNKS_DIR" ]; then
    echo -e "${RED}ERROR: $CHUNKS_DIR does not exist!${NC}"
    exit 1
fi

echo "Source: $CHUNKS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Process each symbol
for symbol in "${SYMBOLS[@]}"; do
    echo -e "${GREEN}Processing $symbol...${NC}"

    # Find all year chunks for this symbol
    chunk_files=($(ls "$CHUNKS_DIR/${symbol}_"*.csv 2>/dev/null | sort))

    if [ ${#chunk_files[@]} -eq 0 ]; then
        echo -e "${RED}  ✗ No chunk files found for $symbol${NC}"
        continue
    fi

    echo "  Found ${#chunk_files[@]} chunk file(s)"

    output_file="$OUTPUT_DIR/${symbol}_transformed_features.csv"

    # Combine files
    # First file: copy with header
    head -1 "${chunk_files[0]}" > "$output_file"

    # All files: append data rows (skip header)
    for chunk in "${chunk_files[@]}"; do
        year=$(basename "$chunk" | sed "s/${symbol}_//" | sed 's/.csv//')
        rows=$(tail -n +2 "$chunk" | wc -l)
        echo "    + $year: $rows rows"
        tail -n +2 "$chunk" >> "$output_file"
    done

    # Count final rows
    total_rows=$(($(wc -l < "$output_file") - 1))
    size=$(du -h "$output_file" | cut -f1)

    echo -e "${GREEN}  ✓ Combined $symbol: $total_rows total rows, $size${NC}"
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Combination Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
echo "Combined Files Summary:"
for symbol in "${SYMBOLS[@]}"; do
    output_file="$OUTPUT_DIR/${symbol}_transformed_features.csv"

    if [ -f "$output_file" ]; then
        rows=$(($(wc -l < "$output_file") - 1))
        size=$(du -h "$output_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $symbol: $rows rows, $size"
    else
        echo -e "  ${RED}✗${NC} $symbol: Not created"
    fi
done

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Verify: ./scripts/verify_extraction.sh"
echo "  2. Optimize: ./scripts/parallel_optimization.sh"
