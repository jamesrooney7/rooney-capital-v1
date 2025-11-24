#!/bin/bash
#
# Run Strategy Factory optimization on all instruments
#
# Usage:
#   ./run_all_instruments.sh
#
# Runs sequentially through all instruments with the new optimized pipeline:
# - Top 10 to walk-forward
# - Top 3 final selection
# - Discretionary exit testing
#
# Expected runtime: 4-8 hours total (8 instruments × 30-60 min each)
# Expected output: 24 strategies (3 per instrument)

set -e  # Exit on error

# List of instruments to optimize
INSTRUMENTS=("ES" "NQ" "RTY" "YM" "CL" "GC" "SI" "HG")

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create it first: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Start timestamp
START_TIME=$(date +%s)
echo "=================================================="
echo "STRATEGY FACTORY - ALL INSTRUMENTS"
echo "=================================================="
echo "Started at: $(date)"
echo "Instruments: ${INSTRUMENTS[@]}"
echo "Total instruments: ${#INSTRUMENTS[@]}"
echo "=================================================="
echo ""

# Track success/failure
SUCCEEDED=()
FAILED=()

# Run each instrument sequentially
for symbol in "${INSTRUMENTS[@]}"; do
    INSTRUMENT_START=$(date +%s)

    echo ""
    echo "=================================================="
    echo "Starting optimization for $symbol"
    echo "Started at: $(date)"
    echo "=================================================="

    if python -m research.strategy_factory.main phase1 \
        --symbol "$symbol" \
        --start 2010-01-01 \
        --end 2024-12-31 \
        --workers 16; then

        INSTRUMENT_END=$(date +%s)
        INSTRUMENT_DURATION=$((INSTRUMENT_END - INSTRUMENT_START))
        SUCCEEDED+=("$symbol")

        echo ""
        echo "✓ Completed $symbol successfully"
        echo "  Runtime: $((INSTRUMENT_DURATION / 60)) minutes"
        echo "  Completed at: $(date)"
    else
        INSTRUMENT_END=$(date +%s)
        INSTRUMENT_DURATION=$((INSTRUMENT_END - INSTRUMENT_START))
        FAILED+=("$symbol")

        echo ""
        echo "✗ Failed $symbol"
        echo "  Runtime: $((INSTRUMENT_DURATION / 60)) minutes"
        echo "  Failed at: $(date)"
        echo "  Continuing with next instrument..."
    fi

    echo ""
done

# Summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=================================================="
echo "ALL INSTRUMENTS COMPLETE!"
echo "=================================================="
echo "Finished at: $(date)"
echo "Total runtime: $((TOTAL_DURATION / 60)) minutes ($((TOTAL_DURATION / 3600)) hours)"
echo ""
echo "Summary:"
echo "  Total instruments: ${#INSTRUMENTS[@]}"
echo "  Succeeded: ${#SUCCEEDED[@]} - ${SUCCEEDED[@]}"
echo "  Failed: ${#FAILED[@]} - ${FAILED[@]}"
echo ""
echo "Expected strategies: $((${#SUCCEEDED[@]} * 3))"
echo "=================================================="

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Some instruments failed. Check logs above for details."
    exit 1
fi

echo ""
echo "✓ All instruments completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review results in strategy_factory.log"
echo "  2. Check database: python -m research.strategy_factory.database.manager --list-runs"
echo "  3. Proceed to ML training for winning strategies"
