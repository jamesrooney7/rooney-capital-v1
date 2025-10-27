#!/bin/bash

# Parallel extraction script - runs N symbols concurrently
# Usage: bash research/extract_parallel.sh [max_parallel]
# Example: bash research/extract_parallel.sh 4

set -e

# Configuration
MAX_PARALLEL=${1:-4}  # Default to 4 parallel processes
START_DATE="2010-01-01"
END_DATE="2024-12-31"
DATA_DIR="data/resampled"
OUTPUT_DIR="data/training"
LOG_DIR="logs/extraction"

# All symbols to extract
SYMBOLS=(
    "ES" "NQ" "RTY" "YM"
    "GC" "SI" "HG" "CL" "NG" "PL"
    "6A" "6B" "6C" "6E" "6J" "6M" "6N" "6S"
)

# Create log directory
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Parallel Feature Extraction"
echo "========================================"
echo "Max parallel processes: $MAX_PARALLEL"
echo "Symbols to process: ${#SYMBOLS[@]}"
echo "Date range: $START_DATE to $END_DATE"
echo "========================================"
echo ""

# Function to check if a symbol is already extracted
is_extracted() {
    local symbol=$1
    local csv_file="${OUTPUT_DIR}/${symbol}_transformed_features.csv"

    if [ -f "$csv_file" ]; then
        # Check if file has content (more than just header)
        local line_count=$(wc -l < "$csv_file")
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

# Track symbols
PENDING_SYMBOLS=()
COMPLETED_SYMBOLS=()
SKIPPED_SYMBOLS=()

# Check which symbols need extraction
for symbol in "${SYMBOLS[@]}"; do
    if is_extracted "$symbol"; then
        echo "✓ $symbol already extracted, skipping"
        SKIPPED_SYMBOLS+=("$symbol")
    else
        echo "→ $symbol queued for extraction"
        PENDING_SYMBOLS+=("$symbol")
    fi
done

echo ""
echo "Summary:"
echo "  To extract: ${#PENDING_SYMBOLS[@]}"
echo "  Already done: ${#SKIPPED_SYMBOLS[@]}"
echo "========================================"
echo ""

if [ ${#PENDING_SYMBOLS[@]} -eq 0 ]; then
    echo "All symbols already extracted! Nothing to do."
    exit 0
fi

# Start extraction with parallelization
SYMBOL_INDEX=0
TOTAL_SYMBOLS=${#PENDING_SYMBOLS[@]}

echo "Starting extraction..."
echo ""

while [ $SYMBOL_INDEX -lt $TOTAL_SYMBOLS ] || [ $(count_running) -gt 0 ]; do
    # Start new processes if we have capacity and pending symbols
    while [ $(count_running) -lt $MAX_PARALLEL ] && [ $SYMBOL_INDEX -lt $TOTAL_SYMBOLS ]; do
        SYMBOL="${PENDING_SYMBOLS[$SYMBOL_INDEX]}"
        LOG_FILE="$LOG_DIR/${SYMBOL}_extraction.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $SYMBOL ($(($SYMBOL_INDEX + 1))/$TOTAL_SYMBOLS)"

        nohup python3 research/extract_training_data.py \
            --symbol "$SYMBOL" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --data-dir "$DATA_DIR" \
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
echo "========================================"
echo "All extractions complete!"
echo "========================================"
echo ""

# Final summary
echo "Extraction results:"
for symbol in "${PENDING_SYMBOLS[@]}"; do
    if is_extracted "$symbol"; then
        echo "  ✓ $symbol"
    else
        echo "  ✗ $symbol (FAILED - check logs/$symbol_extraction.log)"
    fi
done

echo ""
echo "Log files available in: $LOG_DIR"
echo "Output CSV files in: $OUTPUT_DIR"
