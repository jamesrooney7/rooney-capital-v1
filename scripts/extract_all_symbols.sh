#!/bin/bash
# Extract training data for all symbols in parallel
# Optimized for 16 cores / 125GB RAM

set -e

# Configuration
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)
START_DATE="2010-01-01"
END_DATE="2024-12-31"
OUTPUT_DIR="data/ml_training"
MAX_PARALLEL=16  # Match your CPU cores

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Extract Training Data (All Symbols)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Symbols to extract: ${#SYMBOLS[@]}"
echo "Date range: $START_DATE to $END_DATE"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to extract one symbol
extract_symbol() {
    local symbol=$1
    local output_file="$OUTPUT_DIR/${symbol}_transformed_features.csv"

    echo -e "${GREEN}[$(date '+%H:%M:%S')] Starting extraction for $symbol${NC}"

    python research/extract_training_data.py \
        --symbol "$symbol" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --output "$output_file" \
        2>&1 | tee "$OUTPUT_DIR/${symbol}_extraction.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ Completed $symbol${NC}"

        # Show file size and row count
        if [ -f "$output_file" ]; then
            local rows=$(wc -l < "$output_file" | xargs)
            local size=$(du -h "$output_file" | cut -f1)
            echo -e "${YELLOW}  → $symbol: $((rows-1)) trades, $size${NC}"
        fi
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ Failed $symbol${NC}"
    fi

    return $exit_code
}

export -f extract_symbol
export OUTPUT_DIR START_DATE END_DATE GREEN YELLOW RED NC

# Run extractions in parallel
echo -e "${GREEN}Starting parallel extraction...${NC}"
echo ""

printf '%s\n' "${SYMBOLS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'extract_symbol "$@"' _ {}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Extraction Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
echo "Extraction Summary:"
total=0
success=0
failed=0

for symbol in "${SYMBOLS[@]}"; do
    output_file="$OUTPUT_DIR/${symbol}_transformed_features.csv"
    total=$((total + 1))

    if [ -f "$output_file" ]; then
        rows=$(wc -l < "$output_file" | xargs)
        size=$(du -h "$output_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $symbol: $((rows-1)) trades, $size"
        success=$((success + 1))
    else
        echo -e "  ${RED}✗${NC} $symbol: FAILED"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Results: $success succeeded, $failed failed (out of $total total)"
echo ""

if [ $success -gt 0 ]; then
    echo -e "${GREEN}Next step:${NC}"
    echo "  ./scripts/parallel_optimization.sh"
fi
