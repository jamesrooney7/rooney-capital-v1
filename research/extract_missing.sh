#!/bin/bash

# Extract training data for symbols that don't have CSV files yet
# This script checks for existing files and only processes missing ones
# Usage: bash research/extract_missing.sh
#
# To run in background (survives SSH disconnect):
#   nohup bash research/extract_missing.sh > extraction_missing.log 2>&1 &

# All symbols
SYMBOLS=(
    "ES" "NQ" "RTY" "YM"           # Equity indices
    "GC" "SI" "HG" "CL" "NG" "PL"  # Commodities
    "6A" "6B" "6C" "6E" "6J" "6M" "6N" "6S"  # Currencies
)

START_DATE="2010-01-01"
END_DATE="2024-12-31"
DATA_DIR="data/resampled"
OUTPUT_DIR="data/training"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Data Extraction - Missing Only${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Start Date: $START_DATE"
echo "End Date: $END_DATE"
echo "Total Symbols: ${#SYMBOLS[@]}"
echo ""

# Count existing files
EXISTING_COUNT=0
for symbol in "${SYMBOLS[@]}"; do
    if [ -f "$OUTPUT_DIR/${symbol}_transformed_features.csv" ]; then
        ((EXISTING_COUNT++))
    fi
done

echo -e "${GREEN}Already completed: $EXISTING_COUNT${NC}"
echo -e "${YELLOW}To process: $((${#SYMBOLS[@]} - EXISTING_COUNT))${NC}"
echo ""

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FAILED_SYMBOLS=()

# Process each symbol
for symbol in "${SYMBOLS[@]}"; do
    # Check if CSV already exists
    if [ -f "$OUTPUT_DIR/${symbol}_transformed_features.csv" ]; then
        echo -e "${BLUE}⏭️  $symbol${NC}"
        echo "Already exists, skipping..."
        ((SKIP_COUNT++))
        echo ""
        continue
    fi

    echo -e "${BLUE}Processing: $symbol${NC}"
    echo "----------------------------------------"

    python3 research/extract_training_data.py \
        --symbol "$symbol" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --data-dir "$DATA_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $symbol completed successfully${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ $symbol failed${NC}"
        ((FAIL_COUNT++))
        FAILED_SYMBOLS+=("$symbol")
    fi

    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Extraction Complete - Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Skipped (already done): $SKIP_COUNT${NC}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed symbols: ${FAILED_SYMBOLS[*]}${NC}"
fi

echo ""
echo "Training data saved to: $OUTPUT_DIR/"
echo ""

# List all completed files
TOTAL_FILES=$(ls -1 "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)
echo -e "${GREEN}Total CSV files: $TOTAL_FILES${NC}"
echo ""
echo "Recent files:"
ls -lht "$OUTPUT_DIR"/*.csv 2>/dev/null | head -10

echo ""
echo "To view this log later: tail -f extraction_missing.log"
echo ""
