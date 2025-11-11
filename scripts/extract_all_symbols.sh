#!/bin/bash
# Extract training data for all symbols year-by-year with proper warmup
# Runs 4 symbols in parallel to maximize CPU usage
# Each year only loads that year + warmup (not full 15 years!)
# Memory per symbol: ~2-3GB (very safe!)

set -e

# Configuration
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)
YEARS=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024)
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
echo -e "${GREEN}Extract Training Data (Year-by-Year)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Symbols: ${#SYMBOLS[@]}"
echo "Years: ${#YEARS[@]} (2010-2024)"
echo "Parallel batch size: $BATCH_SIZE symbols at a time"
echo "Strategy: Extract one year at a time (with warmup only)"
echo "Memory per symbol: ~2-3GB (year + warmup, not full 15 years!)"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p logs

# Function to calculate warmup start date (252 trading days ≈ 352 calendar days back)
calculate_warmup_start() {
    local year=$1
    # Go back ~352 days from Jan 1 of the year
    python3 -c "from datetime import datetime, timedelta; d = datetime($year, 1, 1) - timedelta(days=352); print(d.strftime('%Y-%m-%d'))"
}

# Function to process a single symbol (all years)
process_symbol() {
    local sym=$1
    local symbol_num=$2
    local total=$3

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}[$sym] Starting extraction ($symbol_num/$total)${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Extract each year separately (with warmup)
    local year_count=0

    for year in "${YEARS[@]}"; do
        year_count=$((year_count + 1))
        local warmup_start=$(calculate_warmup_start $year)
        local year_start="${year}-01-01"
        local year_end="${year}-12-31"

        echo -e "${YELLOW}[$sym] Year $year ($year_count/${#YEARS[@]}) [warmup from $warmup_start]...${NC}"

        python3 research/extract_training_data.py \
            --symbol "$sym" \
            --start "$year_start" \
            --end "$year_end" \
            --output "$CHUNKS_DIR/${sym}_${year}.csv" \
            >> "logs/${sym}_extraction.log" 2>&1

        if [ $? -eq 0 ]; then
            rows=$(wc -l < "$CHUNKS_DIR/${sym}_${year}.csv" 2>/dev/null || echo "0")
            if [ "$rows" -gt 1 ]; then
                echo -e "${GREEN}[$sym]   ✓ $year: $((rows-1)) trades${NC}"
            else
                echo -e "${YELLOW}[$sym]   ⊘ $year: 0 trades (no signals)${NC}"
            fi
        else
            echo -e "${RED}[$sym]   ✗ $year: FAILED${NC}"
        fi
    done

    # Combine all years into one file
    echo -e "${YELLOW}[$sym] Combining ${#YEARS[@]} years...${NC}"

    # Start with header from first year that has data
    local header_written=false
    for year in "${YEARS[@]}"; do
        if [ -f "$CHUNKS_DIR/${sym}_${year}.csv" ] && [ $(wc -l < "$CHUNKS_DIR/${sym}_${year}.csv") -gt 1 ]; then
            head -1 "$CHUNKS_DIR/${sym}_${year}.csv" > "$OUTPUT_DIR/${sym}_transformed_features.csv"
            header_written=true
            break
        fi
    done

    if [ "$header_written" = false ]; then
        echo -e "${RED}[$sym]   ✗ No data for any year!${NC}"
        return 1
    fi

    # Append data from all years (skip headers)
    for year in "${YEARS[@]}"; do
        if [ -f "$CHUNKS_DIR/${sym}_${year}.csv" ]; then
            tail -n +2 "$CHUNKS_DIR/${sym}_${year}.csv" >> "$OUTPUT_DIR/${sym}_transformed_features.csv" 2>/dev/null || true
        fi
    done

    # Verify final file
    if [ -f "$OUTPUT_DIR/${sym}_transformed_features.csv" ]; then
        local total_rows=$(wc -l < "$OUTPUT_DIR/${sym}_transformed_features.csv")
        local size=$(du -h "$OUTPUT_DIR/${sym}_transformed_features.csv" | cut -f1)
        echo -e "${GREEN}[$sym] ✓ COMPLETE: $((total_rows-1)) trades, $size${NC}"

        # Clean up chunks for this symbol to save space
        rm -f "$CHUNKS_DIR/${sym}_"*.csv
        return 0
    else
        echo -e "${RED}[$sym] ✗ Final file not created${NC}"
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
