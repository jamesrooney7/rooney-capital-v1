#!/bin/bash
################################################################################
# Data Availability Checker
# Verifies all required data files exist before running multi-instrument job
################################################################################

SYMBOLS="ES NQ YM RTY GC SI HG CL NG 6A 6B 6E 6J 6N 6S"
TIMEFRAME="15min"
BASE_DIR="/opt/pine/rooney-capital-v1"

# Common data directory locations (adjust as needed)
DATA_DIRS=(
    "${BASE_DIR}/Data"
    "${BASE_DIR}/data"
    "${BASE_DIR}/Data/futures"
    "${BASE_DIR}/data/futures"
)

echo "================================================================================"
echo "DATA AVAILABILITY CHECK"
echo "================================================================================"
echo ""

FOUND_COUNT=0
MISSING_COUNT=0
MISSING_SYMBOLS=""

for symbol in ${SYMBOLS}; do
    FOUND=false

    # Check common file patterns
    for data_dir in "${DATA_DIRS[@]}"; do
        for pattern in "${symbol}_${TIMEFRAME}.parquet" "${symbol}_${TIMEFRAME}.csv" "${symbol}.parquet" "${symbol}.csv"; do
            if [ -f "${data_dir}/${pattern}" ]; then
                echo "✅ ${symbol}: Found at ${data_dir}/${pattern}"
                FOUND=true
                FOUND_COUNT=$((FOUND_COUNT + 1))
                break 2
            fi
        done
    done

    if [ "$FOUND" = false ]; then
        echo "❌ ${symbol}: NOT FOUND"
        MISSING_COUNT=$((MISSING_COUNT + 1))
        MISSING_SYMBOLS="${MISSING_SYMBOLS} ${symbol}"
    fi
done

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Found: ${FOUND_COUNT}/15"
echo "Missing: ${MISSING_COUNT}/15"

if [ ${MISSING_COUNT} -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: Missing data for:${MISSING_SYMBOLS}"
    echo ""
    echo "These instruments will likely fail during the run."
    echo "Consider downloading data or removing them from the symbol list."
    exit 1
else
    echo ""
    echo "✅ ALL DATA FILES FOUND - Ready to run!"
    exit 0
fi
