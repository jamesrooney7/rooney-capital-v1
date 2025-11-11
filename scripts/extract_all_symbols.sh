#!/bin/bash
# Extract training data for all symbols using progressive year-by-year extraction
# Runs 4 symbols in parallel to maximize CPU usage while staying within memory limits
# Each year runs progressively longer backtest (2010, 2010-2011, 2010-2012, etc.)

set -e

# Configuration
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)
YEARS=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024)
START_DATE="2010-01-01"
CHUNKS_DIR="data/training_chunks"
OUTPUT_DIR="data/training"
BATCH_SIZE=4  # Number of symbols to process in parallel

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Extract Training Data (Progressive)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Symbols: ${#SYMBOLS[@]}"
echo "Years: ${#YEARS[@]} (2010-2024)"
echo "Parallel batch size: $BATCH_SIZE symbols at a time"
echo "Strategy: Progressive end dates (2010, 2010-2011, ..., 2010-2024)"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p logs

# Function to process a single symbol (all years progressively)
process_symbol() {
    local sym=$1
    local symbol_num=$2
    local total=$3

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}[$sym] Starting extraction ($symbol_num/$total)${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Extract each year progressively (expanding end date)
    local year_count=0
    local prev_trades=0

    for year in "${YEARS[@]}"; do
        year_count=$((year_count + 1))
        local end_date="${year}-12-31"
        echo -e "${YELLOW}[$sym] Extracting through $year ($year_count/${#YEARS[@]})...${NC}"

        python3 research/extract_training_data.py \
            --symbol "$sym" \
            --start "$START_DATE" \
            --end "$end_date" \
            --output "$CHUNKS_DIR/${sym}_${year}.csv" \
            >> "logs/${sym}_extraction.log" 2>&1

        if [ $? -eq 0 ]; then
            rows=$(wc -l < "$CHUNKS_DIR/${sym}_${year}.csv" 2>/dev/null || echo "0")
            if [ "$rows" -gt 1 ]; then
                new_trades=$((rows - 1 - prev_trades))
                echo -e "${GREEN}[$sym]   ✓ Through $year: $((rows-1)) total trades (+$new_trades new)${NC}"
                prev_trades=$((rows - 1))
            else
                echo -e "${YELLOW}[$sym]   ⊘ Through $year: 0 trades${NC}"
            fi
        else
            echo -e "${RED}[$sym]   ✗ Through $year: FAILED${NC}"
        fi
    done

    # Use the final year file (has all trades from 2010-2024)
    echo -e "${YELLOW}[$sym] Using final extraction (all years)...${NC}"

    if [ -f "$CHUNKS_DIR/${sym}_2024.csv" ]; then
        cp "$CHUNKS_DIR/${sym}_2024.csv" "$OUTPUT_DIR/${sym}_transformed_features.csv"

        local total_rows=$(wc -l < "$OUTPUT_DIR/${sym}_transformed_features.csv")
        local size=$(du -h "$OUTPUT_DIR/${sym}_transformed_features.csv" | cut -f1)
        echo -e "${GREEN}[$sym] ✓ COMPLETE: $((total_rows-1)) trades, $size${NC}"

        # Clean up chunks for this symbol to save space
        rm -f "$CHUNKS_DIR/${sym}_"*.csv
        return 0
    else
        echo -e "${RED}[$sym] ✗ Final file (2024) not created${NC}"
        return 1
    fi
}

# Process symbols in batches of 4
total_symbols=${#SYMBOLS[@]}
completed_symbols=0
batch_num=0

for ((i=0; i<${#SYMBOLS[@]}; i+=BATCH_SIZE)); do
    batch_num=$((batch_num + 1))
    batch_symbols=("${SYMBOLS[@]:i:BATCH_SIZE}")

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}BATCH $batch_num: Processing ${#batch_symbols[@]} symbols in parallel${NC}"
    echo -e "${BLUE}Symbols: ${batch_symbols[*]}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Start all symbols in this batch in parallel
    for sym in "${batch_symbols[@]}"; do
        completed_symbols=$((completed_symbols + 1))
        process_symbol "$sym" $completed_symbols $total_symbols &
    done

    # Wait for all symbols in this batch to complete
    wait

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}BATCH $batch_num COMPLETE${NC}"
    echo -e "${BLUE}========================================${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL EXTRACTIONS COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
echo "Extraction Summary:"
total=0
success=0
failed=0

for symbol in "${SYMBOLS[@]}"; do
    total=$((total + 1))
    if [ -f "$OUTPUT_DIR/${symbol}_transformed_features.csv" ]; then
        rows=$(wc -l < "$OUTPUT_DIR/${symbol}_transformed_features.csv")
        size=$(du -h "$OUTPUT_DIR/${symbol}_transformed_features.csv" | cut -f1)
        cols=$(head -1 "$OUTPUT_DIR/${symbol}_transformed_features.csv" | awk -F',' '{print NF}')
        echo -e "  ${GREEN}✓${NC} $symbol: $((rows-1)) trades, $cols columns, $size"
        success=$((success + 1))
    else
        echo -e "  ${RED}✗${NC} $symbol: FAILED"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Results: $success succeeded, $failed failed (out of $total total)"
echo ""

# Verify warmup fix is present
echo "Verifying warmup fix:"
grep "warmup period" logs/*_extraction.log 2>/dev/null | head -3 || echo "  (logs not yet available)"
echo ""

# Check for any errors
echo "Checking for errors:"
if grep -qi "error\|traceback" logs/*_extraction.log 2>/dev/null; then
    echo -e "${RED}⚠️  Errors found in logs. Check individual log files.${NC}"
else
    echo -e "${GREEN}✓ No errors found!${NC}"
fi
echo ""

if [ $success -gt 0 ]; then
    echo -e "${GREEN}Ready to start training!${NC}"
    echo "Next step:"
    echo "  for sym in ES NQ RTY YM; do"
    echo "    nohup python3 research/train_rf_three_way_split.py --symbol \$sym --bo-trials 50 > logs/\${sym}_training.log 2>&1 &"
    echo "  done"
fi
