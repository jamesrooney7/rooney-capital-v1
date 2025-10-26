#!/bin/bash

# Train Random Forest models for all 22 symbols using CPCV + Bayesian Optimization
# Usage: bash research/train_all_symbols.sh

# All symbols
SYMBOLS=(
    "ES" "NQ" "RTY" "YM"           # Equity indices
    "GC" "SI" "HG" "CL" "NG" "PL"  # Commodities
    "6A" "6B" "6C" "6E" "6J" "6M" "6N" "6S"  # Currencies
    "ZC" "ZS" "ZW"                 # Grains
    "TLT"                          # Bonds
)

DATA_DIR="data/training"
OUTPUT_DIR="src/models"
N_TRIALS=30              # Hyperparameter optimization trials
N_FOLDS=5                # CPCV folds
SCORE_METRIC="sharpe"    # Optimization metric (sharpe, profit_factor, auc)
MIN_TRADES=100           # Minimum trades for threshold optimization
MIN_TOTAL_TRADES=1000    # Minimum total trades for robust training

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Model Training - All Symbols${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Optimization Trials: $N_TRIALS"
echo "CPCV Folds: $N_FOLDS"
echo "Score Metric: $SCORE_METRIC"
echo "Min Total Trades: $MIN_TOTAL_TRADES (robustness requirement)"
echo "Total Symbols: ${#SYMBOLS[@]}"
echo ""
echo -e "${YELLOW}⚠️  This will take several hours to complete!${NC}"
echo -e "${YELLOW}⚠️  Symbols with <$MIN_TOTAL_TRADES trades will be skipped.${NC}"
echo ""

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_SYMBOLS=()
START_TIME=$(date +%s)

# Process each symbol
for symbol in "${SYMBOLS[@]}"; do
    echo -e "${BLUE}Training: $symbol${NC}"
    echo "----------------------------------------"

    SYMBOL_START=$(date +%s)

    python3 research/train_rf_cpcv_bo.py \
        --symbol "$symbol" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --n-trials "$N_TRIALS" \
        --n-folds "$N_FOLDS" \
        --score-metric "$SCORE_METRIC" \
        --min-trades "$MIN_TRADES" \
        --min-total-trades "$MIN_TOTAL_TRADES"

    if [ $? -eq 0 ]; then
        SYMBOL_END=$(date +%s)
        SYMBOL_DURATION=$((SYMBOL_END - SYMBOL_START))
        echo -e "${GREEN}✓ $symbol completed successfully (${SYMBOL_DURATION}s)${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ $symbol failed${NC}"
        ((FAIL_COUNT++))
        FAILED_SYMBOLS+=("$symbol")
    fi

    echo ""
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Complete - Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo "Total Duration: ${HOURS}h ${MINUTES}m"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed symbols: ${FAILED_SYMBOLS[*]}${NC}"
fi

echo ""
echo "Model bundles saved to: $OUTPUT_DIR/"
echo ""

# List generated files
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.pkl "$OUTPUT_DIR"/*_best.json 2>/dev/null | tail -20

echo ""
