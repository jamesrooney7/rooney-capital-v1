#!/bin/bash
# Parallel ML Optimization Script
# Optimized for 16 cores, 13 GB RAM

set -e

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
MAX_PARALLEL_JOBS=16  # Optimized for 125GB RAM / 16 cores (change to 12 if running other workloads)
DATA_DIR="$PROJECT_ROOT/data/training"
OUTPUT_DIR="$PROJECT_ROOT/src/models"
TRAIN_END="2018-12-31"        # End of hyperparameter tuning period
THRESHOLD_END="2020-12-31"     # End of threshold optimization period
RS_TRIALS=120                  # Random search trials
BO_TRIALS=300                  # Bayesian optimization trials
EMBARGO_DAYS=3                 # CPCV embargo days
K_FEATURES=30                  # Number of features to select

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Parallel ML Optimization${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "System Resources:"
echo "  CPUs: $(nproc)"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Warning: $DATA_DIR does not exist. Creating it...${NC}"
    mkdir -p "$DATA_DIR"
fi

# Find all transformed features CSV files
CSV_FILES=($(find "$DATA_DIR" -name "*_transformed_features.csv" -type f 2>/dev/null))

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No *_transformed_features.csv files found in $DATA_DIR${NC}"
    echo ""
    echo "Please run extract_training_data.py first:"
    echo "  python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31 --output $DATA_DIR/ES_transformed_features.csv"
    exit 1
fi

echo -e "${GREEN}Found ${#CSV_FILES[@]} symbol(s) to optimize:${NC}"
for csv in "${CSV_FILES[@]}"; do
    symbol=$(basename "$csv" | sed 's/_transformed_features.csv//')
    echo "  - $symbol"
done
echo ""

# Function to run optimization for one symbol
optimize_symbol() {
    local csv_path=$1
    local symbol=$(basename "$csv_path" | sed 's/_transformed_features.csv//')
    local log_dir="$PROJECT_ROOT/logs"

    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Starting three-way split optimization for $symbol${NC}"

    # Run optimization with logging
    # Set PYTHONPATH to include research and src directories
    cd "$PROJECT_ROOT" || exit 1

    # Create log directory
    mkdir -p "$log_dir"

    PYTHONPATH="$PROJECT_ROOT/research:$PROJECT_ROOT/src:$PYTHONPATH" \
    python3 research/train_rf_three_way_split.py \
        --symbol "$symbol" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --train-end "$TRAIN_END" \
        --threshold-end "$THRESHOLD_END" \
        --rs-trials "$RS_TRIALS" \
        --bo-trials "$BO_TRIALS" \
        --embargo-days "$EMBARGO_DAYS" \
        --k-features "$K_FEATURES" \
        2>&1 | tee "$log_dir/${symbol}_optimization.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed $symbol${NC}"

        # Print summary from test results
        if [ -f "$OUTPUT_DIR/${symbol}_test_results.json" ]; then
            echo -e "${YELLOW}Summary for $symbol:${NC}"
            python3 -c "
import json
with open('$OUTPUT_DIR/${symbol}_test_results.json') as f:
    data = json.load(f)
    print(f\"  Test Sharpe: {data.get('test_sharpe', 'N/A'):.3f}\")
    print(f\"  Test Sortino: {data.get('test_sortino', 'N/A'):.3f}\")
    print(f\"  Profit Factor: {data.get('test_profit_factor', 'N/A'):.2f}\")
    print(f\"  Win Rate: {data.get('test_win_rate', 'N/A'):.1f}%\")
    print(f\"  Trades: {data.get('test_trades', 'N/A')}\")
" 2>/dev/null || echo "  (Could not parse results)"
        fi
    else
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed $symbol (exit code: $exit_code)${NC}"
    fi

    return $exit_code
}

export -f optimize_symbol
export PROJECT_ROOT DATA_DIR OUTPUT_DIR TRAIN_END THRESHOLD_END RS_TRIALS BO_TRIALS EMBARGO_DAYS K_FEATURES GREEN RED YELLOW NC

# Run optimizations in parallel
echo -e "${GREEN}Starting parallel optimization (max $MAX_PARALLEL_JOBS jobs)...${NC}"
echo ""

printf '%s\n' "${CSV_FILES[@]}" | xargs -P "$MAX_PARALLEL_JOBS" -I {} bash -c 'optimize_symbol "$@"' _ {}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All optimizations complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
echo "Results Summary:"
for csv in "${CSV_FILES[@]}"; do
    symbol=$(basename "$csv" | sed 's/_transformed_features.csv//')

    if [ -f "$OUTPUT_DIR/${symbol}_rf_model.pkl" ] && [ -f "$OUTPUT_DIR/${symbol}_best.json" ]; then
        echo -e "  ${GREEN}✓${NC} $symbol: Model and metadata saved"
        if [ -f "$OUTPUT_DIR/${symbol}_test_results.json" ]; then
            # Show test Sharpe
            test_sharpe=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/${symbol}_test_results.json')).get('test_sharpe', 'N/A'))" 2>/dev/null)
            echo -e "       Test Sharpe: $test_sharpe"
        fi
    else
        echo -e "  ${RED}✗${NC} $symbol: FAILED"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Review test results: cat $OUTPUT_DIR/*_test_results.json"
echo "  2. Review model metadata: cat $OUTPUT_DIR/*_best.json"
echo "  3. Models are ready for production in: $OUTPUT_DIR/"
echo "  4. Individual logs available in: logs/"
