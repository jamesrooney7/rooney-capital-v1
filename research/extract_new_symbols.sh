#!/bin/bash

# Extract training data for newly added execution symbols
# These were previously REFERENCE_SYMBOLS and now can be traded
# Usage: bash research/extract_new_symbols.sh

# Newly added symbols (excluding TLT which remains reference-only)
SYMBOLS=(
    "PL"       # Platinum (was reference)
    "6C"       # Canadian Dollar (was reference)
    "6J"       # Japanese Yen (was reference)
    "6M"       # Mexican Peso (was reference)
    "6N"       # Norwegian Krone (was reference)
    "6S"       # Swedish Krona (was reference)
    "ZC"       # Corn (grain)
    "ZS"       # Soybeans (grain)
    "ZW"       # Wheat (grain)
)

START_DATE="2010-01-01"
END_DATE="2024-12-31"
DATA_DIR="data/resampled"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Extract Training Data - New Symbols${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Start Date: $START_DATE"
echo "End Date: $END_DATE"
echo "Total Symbols: ${#SYMBOLS[@]}"
echo ""

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_SYMBOLS=()

# Process each symbol
for symbol in "${SYMBOLS[@]}"; do
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
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed symbols: ${FAILED_SYMBOLS[*]}${NC}"
fi

echo ""
echo "Training data saved to: data/training/"
echo ""
