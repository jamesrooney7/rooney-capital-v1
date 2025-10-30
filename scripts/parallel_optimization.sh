#!/bin/bash
# Parallel ML Optimization Script
# Optimized for 16 cores, 13 GB RAM

set -e

# Configuration
MAX_PARALLEL_JOBS=16  # Optimized for 125GB RAM / 16 cores (change to 12 if running other workloads)
DATA_DIR="data/ml_training"
OUTPUT_BASE_DIR="results"
FEATURE_SELECTION_END="2020-12-31"
HOLDOUT_DATE="2024-01-01"

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
    local output_dir="$OUTPUT_BASE_DIR/${symbol}_optimization"

    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Starting optimization for $symbol${NC}"

    # Create output directory
    mkdir -p "$output_dir"

    # Run optimization with logging
    python research/rf_cpcv_random_then_bo.py \
        --csv "$csv_path" \
        --screen_method clustered \
        --n_clusters 15 \
        --features_per_cluster 2 \
        --feature_selection_end "$FEATURE_SELECTION_END" \
        --holdout "$HOLDOUT_DATE" \
        --n_random 25 \
        --n_bo 65 \
        --n_splits 5 \
        --n_test_folds 2 \
        --embargo_days 2 \
        --output_dir "$output_dir" \
        --export_all_trials \
        2>&1 | tee "$output_dir/optimization_${symbol}.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed $symbol${NC}"

        # Print summary
        if [ -f "$output_dir/best.json" ]; then
            echo -e "${YELLOW}Summary for $symbol:${NC}"
            python3 -c "
import json
with open('$output_dir/best.json') as f:
    data = json.load(f)
    perf = data.get('performance', {})
    print(f\"  Sharpe: {perf.get('sharpe_ratio', 'N/A'):.3f}\")
    print(f\"  DSR: {perf.get('deflated_sharpe_ratio', 'N/A'):.3f}\")
    print(f\"  Profit Factor: {perf.get('profit_factor', 'N/A'):.2f}\")
    print(f\"  Era Positive: {perf.get('era_positive', 'N/A'):.2f}\")
    print(f\"  Features: {len(data.get('ml_features', []))}\")
" 2>/dev/null || echo "  (Could not parse results)"
        fi
    else
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed $symbol (exit code: $exit_code)${NC}"
    fi

    return $exit_code
}

export -f optimize_symbol
export OUTPUT_BASE_DIR FEATURE_SELECTION_END HOLDOUT_DATE GREEN RED YELLOW NC

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
    output_dir="$OUTPUT_BASE_DIR/${symbol}_optimization"

    if [ -f "$output_dir/best.json" ]; then
        echo -e "  ${GREEN}✓${NC} $symbol: $output_dir/best.json"
    else
        echo -e "  ${RED}✗${NC} $symbol: FAILED"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Review results in $OUTPUT_BASE_DIR/*/best.json"
echo "  2. Deploy models: cp $OUTPUT_BASE_DIR/*/best.json src/models/"
echo "  3. Deploy pickles: cp $OUTPUT_BASE_DIR/*/*.pkl src/models/"
