#!/bin/bash
# Extract training data for all symbols in parallel batches
# Optimized for 16 cores / 125GB RAM
# Runs 4 symbols at a time to avoid memory contention

set -e

# Configuration
SYMBOLS=(ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S)
START_DATE="2010-01-01"
END_DATE="2024-12-31"
OUTPUT_DIR="data/training"
BATCH_SIZE=4  # Run 4 symbols at a time (optimal for 16 cores)

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
echo "Batch size: $BATCH_SIZE (parallel per batch)"
echo ""

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Process symbols in batches
total=${#SYMBOLS[@]}
for ((i=0; i<$total; i+=BATCH_SIZE)); do
    batch_num=$((i/BATCH_SIZE + 1))
    batch_symbols=("${SYMBOLS[@]:i:BATCH_SIZE}")

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}BATCH $batch_num: ${batch_symbols[@]}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Start batch
    for sym in "${batch_symbols[@]}"; do
        echo -e "${YELLOW}[$(date '+%H:%M:%S')] Starting extraction for $sym${NC}"
        nohup python3 research/extract_training_data.py \
            --symbol "$sym" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            > logs/${sym}_extraction.log 2>&1 &
    done

    # Wait for this batch to complete
    echo -e "${YELLOW}Waiting for batch $batch_num to complete...${NC}"
    while pgrep -f "extract_training_data.py" > /dev/null; do
        sleep 10
        # Show progress
        running=$(pgrep -f "extract_training_data.py" | wc -l)
        completed=$(ls $OUTPUT_DIR/*.csv 2>/dev/null | wc -l)
        echo -e "${YELLOW}  Progress: $completed/18 complete, $running running${NC}"
    done

    echo -e "${GREEN}✅ Batch $batch_num complete!${NC}"

    # Show which symbols completed in this batch
    for sym in "${batch_symbols[@]}"; do
        if [ -f "$OUTPUT_DIR/${sym}_transformed_features.csv" ]; then
            size=$(du -h "$OUTPUT_DIR/${sym}_transformed_features.csv" | cut -f1)
            rows=$(wc -l < "$OUTPUT_DIR/${sym}_transformed_features.csv" | xargs)
            echo -e "  ${GREEN}✓${NC} $sym: $((rows-1)) trades, $size"
        else
            echo -e "  ${RED}✗${NC} $sym: FAILED (check logs/${sym}_extraction.log)"
        fi
    done
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

# Verify warmup fix is present
echo "Verifying warmup fix:"
grep "warmup period" logs/*_extraction.log | head -3
echo ""

# Check for any errors
echo "Checking for errors:"
if grep -qi error logs/*_extraction.log; then
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
