#!/bin/bash
##
# Run multiple model configurations in parallel and compare Phase 2/3 results.
#
# This script tests different hyperparameter configurations to find the best model.
# Each configuration is run in parallel, then results are compared.
#
# Usage:
#   bash research/compare_model_configs_parallel.sh
#   bash research/compare_model_configs_parallel.sh --symbol NQ
#
# Output:
#   - Individual log files in outputs/model_comparison/logs/
#   - Comparison summary printed to console
#   - Best model JSON saved to outputs/model_comparison/
##

set -euo pipefail

# Default parameters
SYMBOL="${1:-ES}"
DATA_DIR="data/training"
OUTPUT_BASE="outputs/model_comparison"
LOG_DIR="${OUTPUT_BASE}/logs"
TRAIN_END="2018-12-31"
THRESHOLD_END="2020-12-31"

# Create output directories
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Model Comparison Pipeline"
echo "=========================================="
echo "Symbol: ${SYMBOL}"
echo "Log directory: ${LOG_DIR}"
echo "Training period: [start] to ${TRAIN_END}"
echo "Threshold period: ${TRAIN_END} to ${THRESHOLD_END}"
echo "Test period: ${THRESHOLD_END} to [end]"
echo "=========================================="
echo ""

# Model configurations
# Format: MODEL_NAME:RS_TRIALS:BO_TRIALS:EMBARGO_DAYS:K_FEATURES:SEED
declare -a CONFIGS=(
    # Baseline configurations with varying trial counts
    "MODEL_A:25:65:2:30:42"
    "MODEL_B:30:70:2:30:43"
    "MODEL_C:20:60:2:30:44"

    # Varying embargo days (leakage sensitivity)
    "MODEL_D:25:65:1:30:42"    # Less embargo (more data, but potential leakage)
    "MODEL_E:25:65:3:30:42"    # More embargo (less leakage, but less data)
    "MODEL_F:25:65:5:30:42"    # Even more embargo

    # Varying feature counts
    "MODEL_G:25:65:2:20:42"    # Fewer features (less overfitting risk)
    "MODEL_H:25:65:2:40:42"    # More features (more complexity)
    "MODEL_I:25:65:2:50:42"    # Many features

    # More thorough search
    "MODEL_J:40:100:2:30:42"   # More trials (better optimization, slower)
    "MODEL_K:50:120:2:30:42"   # Even more trials

    # Different random seeds (stability check)
    "MODEL_L:25:65:2:30:100"
    "MODEL_M:25:65:2:30:200"

    # Combined variations
    "MODEL_N:30:80:3:25:45"    # More trials + more embargo + fewer features
    "MODEL_O:35:90:1:35:46"    # More trials + less embargo + more features
)

echo "Testing ${#CONFIGS[@]} model configurations..."
echo ""

# Array to track background process IDs
PIDS=()

# Function to run a single model configuration
run_model() {
    local config="$1"
    IFS=':' read -r model rs_trials bo_trials embargo k_features seed <<< "$config"

    local log_file="${LOG_DIR}/${model}.log"

    echo "Starting ${model} (RS=${rs_trials}, BO=${bo_trials}, embargo=${embargo}, k=${k_features}, seed=${seed})..."

    # Run training script and save output to log
    python research/train_rf_three_way_split.py \
        --symbol "${SYMBOL}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_BASE}/${model}" \
        --train-end "${TRAIN_END}" \
        --threshold-end "${THRESHOLD_END}" \
        --rs-trials "${rs_trials}" \
        --bo-trials "${bo_trials}" \
        --embargo-days "${embargo}" \
        --k-features "${k_features}" \
        --seed "${seed}" \
        > "${log_file}" 2>&1

    echo "${model} complete"
}

# Export function so it's available to parallel processes
export -f run_model
export SYMBOL DATA_DIR OUTPUT_BASE LOG_DIR TRAIN_END THRESHOLD_END

# Run all configurations in parallel
for config in "${CONFIGS[@]}"; do
    run_model "$config" &
    PIDS+=($!)
done

# Wait for all background processes
echo ""
echo "Waiting for all models to complete..."
echo "Running ${#PIDS[@]} processes in parallel..."
echo ""

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "Process $pid completed"
done

echo ""
echo "=========================================="
echo "All models complete! Analyzing results..."
echo "=========================================="
echo ""

# Parse results and display comparison
echo "Comparing Phase 2 (threshold period) performance..."
echo ""

python research/parse_model_comparison.py \
    --log-dir "${LOG_DIR}" \
    --output-dir "${OUTPUT_BASE}" \
    --symbol "${SYMBOL}"

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review Phase 3 results above"
echo "  2. If performance is positive, deploy the winning model"
echo "  3. If still negative, consider:"
echo "     - Testing base IBS strategy without ML"
echo "     - Different feature engineering approaches"
echo "     - Alternative model architectures"
echo ""
