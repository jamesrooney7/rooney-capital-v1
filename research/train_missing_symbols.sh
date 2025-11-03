#!/bin/bash

# Train the 7 missing symbols that have training data but weren't included in initial batch
#
# Missing symbols:
#   HG - Copper (was in old models)
#   6C, 6J, 6M, 6N, 6S - Additional currencies with training data
#   PL - Platinum
#
# Usage: bash research/train_missing_symbols.sh

# Missing symbols that have training data
SYMBOLS=(
    "HG"                           # Copper (was in old models!)
    "6C" "6J" "6M" "6N" "6S"      # Additional currencies
    "PL"                           # Platinum
)

# Data and output directories
DATA_DIR="data/training"
OUTPUT_DIR="src/models"

# Temporal split dates (CRITICAL: DO NOT use 2022-2024 for training!)
TRAIN_END="2020-12-31"      # Phase 1: Hyperparameter tuning (2010-2020)
THRESHOLD_END="2021-12-31"  # Phase 2: Threshold optimization (2021)
                            # Phase 3: Test period (2022-2024) - NEVER SEEN DURING TRAINING!

# Optimization parameters
RS_TRIALS=100   # Random Search trials
BO_TRIALS=100   # Bayesian Optimization trials
N_FOLDS=5       # CPCV folds
K_TEST=2        # CPCV k_test parameter
EMBARGO_DAYS=2  # Embargo period (days)

# Feature screening
FEATURE_SCREEN_METHOD="importance"  # Valid: "importance", "permutation", "l1", "none"
N_FEATURES=30                        # Top N features to select

# Create log directory
mkdir -p results/logs

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training 7 Missing Symbols (FIXED!)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "🔧 FIXES APPLIED:"
echo "   ✅ CPCV future data bug fixed"
echo "   ✅ Fixed threshold (0.50) during hyperparameter tuning"
echo "   ✅ Proper three-way temporal split"
echo ""
echo "📅 TEMPORAL SPLIT:"
echo "   Training:   2010-2020 (hyperparameters)"
echo "   Threshold:  2021 (threshold optimization)"
echo "   Holdout:    2022-2024 (NEVER seen during training)"
echo ""
echo "📊 SYMBOLS TO TRAIN: ${#SYMBOLS[@]}"
echo "   ${SYMBOLS[*]}"
echo ""
echo "⚙️  PARAMETERS:"
echo "   Random Search Trials: $RS_TRIALS"
echo "   Bayesian Opt Trials: $BO_TRIALS"
echo "   CPCV Folds: $N_FOLDS"
echo "   Feature Selection: $FEATURE_SCREEN_METHOD (top $N_FEATURES)"
echo ""
echo -e "${YELLOW}⏱  Estimated time: 5-7 hours (45-60 min per symbol)${NC}"
echo ""

# Confirm before proceeding (skip if --yes flag or non-interactive)
if [[ "$1" != "--yes" ]] && [ -t 0 ]; then
    read -p "Continue with retraining 7 missing symbols? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo "Starting retraining (non-interactive mode or --yes flag)..."
fi

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_SYMBOLS=()
START_TIME=$(date +%s)

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Training...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Process each symbol
for symbol in "${SYMBOLS[@]}"; do
    SYMBOL_NUM=$((SUCCESS_COUNT + FAIL_COUNT + 1))
    TOTAL_SYMBOLS=${#SYMBOLS[@]}

    echo ""
    echo -e "${BLUE}[$SYMBOL_NUM/$TOTAL_SYMBOLS] Training: $symbol${NC}"
    echo "----------------------------------------"

    SYMBOL_START=$(date +%s)
    LOGFILE="results/logs/${symbol}_retrain_$(date +%Y%m%d_%H%M%S).log"

    python3 research/train_rf_three_way_split.py \
        --symbol "$symbol" \
        --data-dir "$DATA_DIR" \
        --train-end "$TRAIN_END" \
        --threshold-end "$THRESHOLD_END" \
        --rs-trials "$RS_TRIALS" \
        --bo-trials "$BO_TRIALS" \
        --folds "$N_FOLDS" \
        --k-test "$K_TEST" \
        --embargo-days "$EMBARGO_DAYS" \
        --screen-method "$FEATURE_SCREEN_METHOD" \
        --k-features "$N_FEATURES" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOGFILE"

    # Check exit code of python command (not tee)
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        SYMBOL_END=$(date +%s)
        SYMBOL_DURATION=$((SYMBOL_END - SYMBOL_START))
        SYMBOL_MINUTES=$((SYMBOL_DURATION / 60))
        SYMBOL_SECONDS=$((SYMBOL_DURATION % 60))

        echo -e "${GREEN}✓ $symbol completed successfully (${SYMBOL_MINUTES}m ${SYMBOL_SECONDS}s)${NC}"

        # Extract Sharpe from saved model
        if [ -f "$OUTPUT_DIR/${symbol}_test_results.json" ]; then
            SHARPE=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/${symbol}_test_results.json')); print(f\"{d['test_metrics']['sharpe']:.3f}\")" 2>/dev/null)
            echo -e "  Test Sharpe (2022-2024): $SHARPE"
        fi

        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ $symbol failed - check $LOGFILE${NC}"
        ((FAIL_COUNT++))
        FAILED_SYMBOLS+=("$symbol")
    fi

    # Show progress
    COMPLETED=$((SUCCESS_COUNT + FAIL_COUNT))
    REMAINING=$((TOTAL_SYMBOLS - COMPLETED))
    echo "Progress: $COMPLETED/$TOTAL_SYMBOLS symbols completed, $REMAINING remaining"
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Complete - Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}✗ Failed: $FAIL_COUNT${NC}"
echo "⏱  Total Duration: ${HOURS}h ${MINUTES}m"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed symbols: ${FAILED_SYMBOLS[*]}${NC}"
    echo "Check logs in: results/logs/"
fi

echo ""
echo "Model bundles saved to: $OUTPUT_DIR/"
echo ""

# List generated files with Sharpe ratios
echo "Generated models:"
echo ""
printf "%-8s %-12s %-12s %-10s\n" "Symbol" "Sharpe" "Threshold" "Status"
echo "--------------------------------------------------------"

for symbol in "${SYMBOLS[@]}"; do
    TEST_FILE="$OUTPUT_DIR/${symbol}_test_results.json"
    MODEL_FILE="$OUTPUT_DIR/${symbol}_rf_model.pkl"

    if [ -f "$TEST_FILE" ] && [ -f "$MODEL_FILE" ]; then
        SHARPE=$(python3 -c "import json; d=json.load(open('$TEST_FILE')); print(f\"{d['test_metrics']['sharpe']:.3f}\")" 2>/dev/null)
        THRESHOLD="0.50"
        STATUS="✅"
    else
        SHARPE="N/A"
        THRESHOLD="N/A"
        STATUS="❌"
    fi

    printf "%-8s %-12s %-12s %-10s\n" "$symbol" "$SHARPE" "$THRESHOLD" "$STATUS"
done

echo ""
echo -e "${YELLOW}📊 Next step: View all 18 symbol results${NC}"
echo "   python3 research/show_retrained_results.py src/models/"
echo ""
echo -e "${YELLOW}🎯 Then: Calculate portfolio-level metrics${NC}"
echo "   See: PROPER_TESTING_WORKFLOW.md"
echo ""
