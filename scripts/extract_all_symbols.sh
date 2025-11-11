#!/bin/bash
# Extract training data for all symbols using year-by-year chunked extraction
# This avoids memory exhaustion by processing one year at a time per symbol

set -e

# Configuration
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)
YEARS=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024)
START_DATE="2010-01-01"
END_DATE="2024-12-31"
OUTPUT_DIR="data/training"
CHUNKS_DIR="data/training_chunks"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Extract Training Data (Chunked by Year)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Symbols: ${#SYMBOLS[@]}"
echo "Years: ${#YEARS[@]} (2010-2024)"
echo "Strategy: Extract year-by-year to avoid memory exhaustion"
echo "Date range: $START_DATE to $END_DATE"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p logs

# Track progress
total_symbols=${#SYMBOLS[@]}
completed_symbols=0

for sym in "${SYMBOLS[@]}"; do
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Extracting $sym ($((completed_symbols+1))/$total_symbols)${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Extract each year separately
    year_count=0
    for year in "${YEARS[@]}"; do
        year_count=$((year_count + 1))
        echo -e "${YELLOW}[$sym] Year $year ($year_count/${#YEARS[@]})...${NC}"

        python3 research/extract_training_data.py \
            --symbol "$sym" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --filter-year $year \
            --output "$CHUNKS_DIR/${sym}_${year}.csv" \
            >> "logs/${sym}_extraction.log" 2>&1

        if [ $? -eq 0 ]; then
            rows=$(wc -l < "$CHUNKS_DIR/${sym}_${year}.csv" 2>/dev/null || echo "0")
            if [ "$rows" -gt 1 ]; then
                echo -e "${GREEN}  ✓ $year: $((rows-1)) trades${NC}"
            else
                echo -e "${YELLOW}  ⊘ $year: 0 trades (no signals)${NC}"
            fi
        else
            echo -e "${RED}  ✗ $year: FAILED${NC}"
        fi
    done

    # Combine all years into one file
    echo -e "${YELLOW}[$sym] Combining ${#YEARS[@]} years...${NC}"

    # Start with header from first year that has data
    header_written=false
    for year in "${YEARS[@]}"; do
        if [ -f "$CHUNKS_DIR/${sym}_${year}.csv" ] && [ $(wc -l < "$CHUNKS_DIR/${sym}_${year}.csv") -gt 1 ]; then
            head -1 "$CHUNKS_DIR/${sym}_${year}.csv" > "$OUTPUT_DIR/${sym}_transformed_features.csv"
            header_written=true
            break
        fi
    done

    if [ "$header_written" = false ]; then
        echo -e "${RED}  ✗ $sym: No data for any year!${NC}"
        continue
    fi

    # Append data from all years (skip headers)
    for year in "${YEARS[@]}"; do
        if [ -f "$CHUNKS_DIR/${sym}_${year}.csv" ]; then
            tail -n +2 "$CHUNKS_DIR/${sym}_${year}.csv" >> "$OUTPUT_DIR/${sym}_transformed_features.csv" 2>/dev/null || true
        fi
    done

    # Verify final file
    if [ -f "$OUTPUT_DIR/${sym}_transformed_features.csv" ]; then
        total_rows=$(wc -l < "$OUTPUT_DIR/${sym}_transformed_features.csv")
        size=$(du -h "$OUTPUT_DIR/${sym}_transformed_features.csv" | cut -f1)
        echo -e "${GREEN}✓ $sym complete: $((total_rows-1)) trades, $size${NC}"
        completed_symbols=$((completed_symbols + 1))

        # Clean up chunks for this symbol to save space
        rm -f "$CHUNKS_DIR/${sym}_"*.csv
    else
        echo -e "${RED}✗ $sym: Final file not created${NC}"
    fi
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
grep "warmup period" logs/*_extraction.log | head -3
echo ""

# Check for any errors
echo "Checking for errors:"
if grep -qi "error\|traceback" logs/*_extraction.log; then
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
