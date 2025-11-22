#!/bin/bash
################################################################################
# Multi-Instrument Strategy Factory Runner
# Runs Phase 1 on all 15 instruments sequentially
# Safe to run unattended - will continue even if individual instruments fail
################################################################################

set -e  # Exit on error in main script (but not in instrument loops)

# Configuration
SYMBOLS="ES NQ YM RTY GC SI HG CL NG 6A 6B 6E 6J 6N 6S"
START_DATE="2010-01-01"
END_DATE="2021-12-31"
TIMEFRAME="15min"
WORKERS=10
BASE_DIR="/opt/pine/rooney-capital-v1"
LOG_DIR="${BASE_DIR}/logs/strategy_factory"

# Create log directory
mkdir -p "${LOG_DIR}"

# Overall tracking
OVERALL_LOG="${LOG_DIR}/all_instruments_$(date +%Y%m%d_%H%M%S).log"
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_SYMBOLS=""

# Start time
START_TIME=$(date +%s)

echo "================================================================================"
echo "MULTI-INSTRUMENT STRATEGY FACTORY RUNNER"
echo "================================================================================"
echo "Start Time: $(date)"
echo "Instruments: ${SYMBOLS}"
echo "Workers: ${WORKERS}"
echo "Date Range: ${START_DATE} to ${END_DATE}"
echo "Log Directory: ${LOG_DIR}"
echo "================================================================================"
echo ""

# Log to both console and file
exec > >(tee -a "${OVERALL_LOG}")
exec 2>&1

# Function to run Phase 1 for a single instrument
run_instrument() {
    local symbol=$1
    local instrument_log="${LOG_DIR}/${symbol}_$(date +%Y%m%d_%H%M%S).log"

    echo "--------------------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] Starting ${symbol}"
    echo "--------------------------------------------------------------------------------"

    # Check if data file exists (optional check - adjust path as needed)
    # Uncomment if you want to verify data exists first:
    # if [ ! -f "${BASE_DIR}/data/${symbol}_${TIMEFRAME}.parquet" ]; then
    #     echo "⚠️  WARNING: Data file not found for ${symbol}"
    #     echo "   Expected: ${BASE_DIR}/data/${symbol}_${TIMEFRAME}.parquet"
    #     return 1
    # fi

    # Run Phase 1
    cd "${BASE_DIR}"

    if python3 -m research.strategy_factory.main phase1 \
        --symbol "${symbol}" \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        --timeframe "${TIMEFRAME}" \
        --workers "${WORKERS}" \
        > "${instrument_log}" 2>&1; then

        echo "✅ ${symbol} COMPLETED SUCCESSFULLY"
        echo "   Log: ${instrument_log}"
        return 0
    else
        echo "❌ ${symbol} FAILED"
        echo "   Log: ${instrument_log}"
        echo "   Check log for details"
        return 1
    fi
}

# Main loop - run each instrument
for symbol in ${SYMBOLS}; do
    echo ""
    echo "================================================================================"
    echo "PROCESSING: ${symbol} (${SUCCESS_COUNT} complete, ${FAILED_COUNT} failed so far)"
    echo "================================================================================"

    if run_instrument "${symbol}"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_SYMBOLS="${FAILED_SYMBOLS} ${symbol}"
    fi

    echo ""
    echo "Progress: ${SUCCESS_COUNT}/${#SYMBOLS} successful, ${FAILED_COUNT} failed"
    echo ""

    # Small delay between instruments
    sleep 5
done

# Final summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================================"
echo "ALL INSTRUMENTS COMPLETE"
echo "================================================================================"
echo "End Time: $(date)"
echo "Total Runtime: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo ""
echo "Results:"
echo "  ✅ Successful: ${SUCCESS_COUNT}"
echo "  ❌ Failed: ${FAILED_COUNT}"
if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "  Failed symbols:${FAILED_SYMBOLS}"
fi
echo ""
echo "Overall Log: ${OVERALL_LOG}"
echo "Individual Logs: ${LOG_DIR}/"
echo "================================================================================"

# Optional: Send email notification (uncomment and configure if needed)
# if command -v mail &> /dev/null; then
#     echo "Summary: ${SUCCESS_COUNT} successful, ${FAILED_COUNT} failed" | \
#         mail -s "Strategy Factory Complete" your-email@example.com
# fi

# Exit with success if at least one instrument succeeded
if [ ${SUCCESS_COUNT} -gt 0 ]; then
    exit 0
else
    exit 1
fi
