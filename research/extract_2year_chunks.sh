#!/bin/bash

# 2-Year Chunk Extraction Script
# Runs 16 symbols in parallel with 2-year rolling windows
# Each chunk uses 1 year warmup, keeps only year 2
#
# Usage: bash research/extract_2year_chunks.sh

set -e

# Configuration
MAX_PARALLEL=12
DATA_DIR="data/resampled"
OUTPUT_DIR="data/training_chunks"
LOG_DIR="logs/extraction_chunks"

# All symbols to extract
SYMBOLS=(
    "ES" "NQ" "RTY" "YM"
    "GC" "SI" "HG" "CL" "NG" "PL"
    "6A" "6B" "6C" "6E" "6J" "6M" "6N" "6S"
)

# Time chunks: [warmup_year, extraction_year]
# We extract 2 years but only keep the second year (after warmup)
CHUNKS=(
    "2010:2011"
    "2011:2012"
    "2012:2013"
    "2013:2014"
    "2014:2015"
    "2015:2016"
    "2016:2017"
    "2017:2018"
    "2018:2019"
    "2019:2020"
    "2020:2021"
    "2021:2022"
    "2022:2023"
    "2023:2024"
)

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "2-Year Chunk Extraction"
echo "========================================"
echo "Symbols: ${#SYMBOLS[@]}"
echo "Time chunks: ${#CHUNKS[@]}"
echo "Total extractions: $((${#SYMBOLS[@]} * ${#CHUNKS[@]}))"
echo "Max parallel: $MAX_PARALLEL symbols"
echo "Expected RAM: ~50GB per chunk"
echo "========================================"
echo ""

# Function to check if a chunk is already extracted
is_chunk_extracted() {
    local symbol=$1
    local year=$2
    local csv_file="${OUTPUT_DIR}/${symbol}_${year}.csv"

    if [ -f "$csv_file" ]; then
        # Check if file has content (more than just header)
        local line_count=$(wc -l < "$csv_file" 2>/dev/null || echo "0")
        if [ "$line_count" -gt 1 ]; then
            return 0  # Already extracted
        fi
    fi
    return 1  # Not extracted
}

# Function to count running extraction processes
count_running() {
    ps aux | grep "extract_training_data.py" | grep -v grep | wc -l
}

# Process each time chunk sequentially
for chunk in "${CHUNKS[@]}"; do
    warmup_year=$(echo $chunk | cut -d: -f1)
    extract_year=$(echo $chunk | cut -d: -f2)

    start_date="${warmup_year}-01-01"
    end_date="${extract_year}-12-31"

    echo ""
    echo "========================================"
    echo "Processing chunk: $start_date to $end_date"
    echo "Warmup year: $warmup_year (discarded)"
    echo "Output year: $extract_year (kept)"
    echo "========================================"
    echo ""

    # Track symbols for this chunk
    PENDING_SYMBOLS=()
    SKIPPED_SYMBOLS=()

    # Check which symbols need extraction for this chunk
    for symbol in "${SYMBOLS[@]}"; do
        if is_chunk_extracted "$symbol" "$extract_year"; then
            echo "✓ ${symbol}_${extract_year} already extracted, skipping"
            SKIPPED_SYMBOLS+=("$symbol")
        else
            echo "→ ${symbol}_${extract_year} queued for extraction"
            PENDING_SYMBOLS+=("$symbol")
        fi
    done

    if [ ${#PENDING_SYMBOLS[@]} -eq 0 ]; then
        echo "All symbols already extracted for $extract_year! Skipping chunk."
        continue
    fi

    echo ""
    echo "Extracting ${#PENDING_SYMBOLS[@]} symbols for year $extract_year..."
    echo ""

    # Extract symbols in parallel (up to MAX_PARALLEL at once)
    SYMBOL_INDEX=0
    TOTAL_SYMBOLS=${#PENDING_SYMBOLS[@]}

    while [ $SYMBOL_INDEX -lt $TOTAL_SYMBOLS ] || [ $(count_running) -gt 0 ]; do
        # Start new processes if we have capacity and pending symbols
        while [ $(count_running) -lt $MAX_PARALLEL ] && [ $SYMBOL_INDEX -lt $TOTAL_SYMBOLS ]; do
            SYMBOL="${PENDING_SYMBOLS[$SYMBOL_INDEX]}"
            LOG_FILE="$LOG_DIR/${SYMBOL}_${extract_year}.log"
            OUTPUT_FILE="${OUTPUT_DIR}/${SYMBOL}_${extract_year}.csv"

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${SYMBOL}_${extract_year} ($(($SYMBOL_INDEX + 1))/$TOTAL_SYMBOLS)"

            nohup python3 research/extract_training_data.py \
                --symbol "$SYMBOL" \
                --start "$start_date" \
                --end "$end_date" \
                --data-dir "$DATA_DIR" \
                --output "$OUTPUT_FILE" \
                --filter-year "$extract_year" \
                > "$LOG_FILE" 2>&1 &

            SYMBOL_INDEX=$((SYMBOL_INDEX + 1))
            sleep 2  # Small delay to avoid race conditions
        done

        # Wait a bit before checking again
        if [ $(count_running) -ge $MAX_PARALLEL ] || [ $SYMBOL_INDEX -lt $TOTAL_SYMBOLS ]; then
            sleep 10
        fi
    done

    echo ""
    echo "✅ Completed chunk $extract_year"
    echo ""

    # Show current RAM usage
    echo "Current RAM usage:"
    free -h | grep -E "Mem|Swap"
    echo ""
done

echo ""
echo "========================================"
echo "All chunks complete!"
echo "========================================"
echo ""
echo "Next step: Concatenate chunks into final CSV files"
echo "Run: python3 research/concatenate_chunks.py"
echo ""
