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
OUTPUT_BASE_DIR="$PROJECT_ROOT/results"
FEATURE_SELECTION_END="2020-12-31"
HOLDOUT_DATE="2023-01-01"

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

    # Run optimization with logging
    # Set PYTHONPATH to include research and src directories
    cd "$PROJECT_ROOT" || exit 1

    # Create output directory (after cd to project root)
    mkdir -p "$output_dir"

    PYTHONPATH="$PROJECT_ROOT/research:$PROJECT_ROOT/src:$PYTHONPATH" \
    python3 research/rf_cpcv_random_then_bo.py \
        --input "$csv_path" \
        --outdir "$output_dir" \
        --symbol "$symbol" \
        --screen_method clustered \
        --n_clusters 15 \
        --features_per_cluster 2 \
        --feature_selection_end "$FEATURE_SELECTION_END" \
        --holdout_start "$HOLDOUT_DATE" \
        --rs_trials 25 \
        --bo_trials 65 \
        --folds 5 \
        --k_test 2 \
        --embargo_days 2 \
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
export PROJECT_ROOT DATA_DIR OUTPUT_BASE_DIR FEATURE_SELECTION_END HOLDOUT_DATE GREEN RED YELLOW NC

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
