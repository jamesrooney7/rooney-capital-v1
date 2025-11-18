#!/bin/bash
# Compare model configurations for ES IN PARALLEL (Lightweight Version)
# Focus on fewer features + different screening methods
# Pick the best based on Phase 2 (threshold period 2019-2020)
# Then evaluate winner on Phase 3 (test period 2021-2024)

set -e  # Exit on error

SYMBOL="ES"
TRAIN_END="2018-12-31"
THRESHOLD_END="2020-12-31"
RS_TRIALS=30
BO_TRIALS=50

echo "=========================================="
echo "ES Model Configuration Comparison (PARALLEL - LIGHTWEIGHT)"
echo "=========================================="
echo ""
echo "Training 12 different configurations IN PARALLEL:"
echo ""
echo "Feature Count Sweep (importance screening):"
echo "  Model A:  5 features"
echo "  Model B: 10 features"
echo "  Model C: 15 features"
echo "  Model D: 20 features"
echo "  Model E: 25 features"
echo "  Model F: 30 features (baseline)"
echo ""
echo "Feature Count + Permutation Screening:"
echo "  Model G:  5 features, permutation"
echo "  Model H: 10 features, permutation"
echo "  Model I: 15 features, permutation"
echo "  Model J: 20 features, permutation"
echo ""
echo "Alternative Screening Methods (15 features):"
echo "  Model K: 15 features, l1"
echo "  Model L: 15 features, clustered"
echo ""
echo "Trials per model: $RS_TRIALS RS + $BO_TRIALS BO = 80 total"
echo "Estimated runtime per model: ~1 hour"
echo "Total parallel runtime: ~1 hour (with sufficient resources)"
echo ""
echo "Selection: Pick best based on Phase 2 (2019-2020)"
echo "Evaluation: Report winner's Phase 3 (2021-2024) performance"
echo ""
echo "⚠️  WARNING: Running 12 models in parallel is CPU/memory intensive!"
echo "    Recommended: 24+ CPU cores, 48+ GB RAM"
echo "=========================================="
echo ""

# Create output directory
mkdir -p outputs/model_comparison
OUTPUT_DIR="outputs/model_comparison"

# Array to store background PIDs
declare -a PIDS

echo "Launching all 12 models in parallel..."
echo ""

#######################################
# FEATURE COUNT SWEEP - IMPORTANCE
#######################################

# Model A: 5 features
echo "Launching Model A: 5 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 5 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_a_5feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_a_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_a_ES_test_results.json" 2>/dev/null || true
echo "Model A complete") &
PIDS+=($!)

# Model B: 10 features
echo "Launching Model B: 10 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 10 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_b_10feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_b_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_b_ES_test_results.json" 2>/dev/null || true
echo "Model B complete") &
PIDS+=($!)

# Model C: 15 features
echo "Launching Model C: 15 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_c_15feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_c_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_c_ES_test_results.json" 2>/dev/null || true
echo "Model C complete") &
PIDS+=($!)

# Model D: 20 features
echo "Launching Model D: 20 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 20 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_d_20feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_d_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_d_ES_test_results.json" 2>/dev/null || true
echo "Model D complete") &
PIDS+=($!)

# Model E: 25 features
echo "Launching Model E: 25 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 25 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_e_25feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_e_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_e_ES_test_results.json" 2>/dev/null || true
echo "Model E complete") &
PIDS+=($!)

# Model F: 30 features (baseline)
echo "Launching Model F: 30 features, importance (baseline)..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_f_30feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_f_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_f_ES_test_results.json" 2>/dev/null || true
echo "Model F complete") &
PIDS+=($!)

#######################################
# FEATURE COUNT SWEEP - PERMUTATION
#######################################

# Model G: 5 features, permutation
echo "Launching Model G: 5 features, permutation..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 5 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_g_5feat_permutation.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_g_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_g_ES_test_results.json" 2>/dev/null || true
echo "Model G complete") &
PIDS+=($!)

# Model H: 10 features, permutation
echo "Launching Model H: 10 features, permutation..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 10 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_h_10feat_permutation.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_h_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_h_ES_test_results.json" 2>/dev/null || true
echo "Model H complete") &
PIDS+=($!)

# Model I: 15 features, permutation
echo "Launching Model I: 15 features, permutation..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_i_15feat_permutation.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_i_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_i_ES_test_results.json" 2>/dev/null || true
echo "Model I complete") &
PIDS+=($!)

# Model J: 20 features, permutation
echo "Launching Model J: 20 features, permutation..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 20 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_j_20feat_permutation.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_j_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_j_ES_test_results.json" 2>/dev/null || true
echo "Model J complete") &
PIDS+=($!)

#######################################
# ALTERNATIVE SCREENING METHODS
#######################################

# Model K: 15 features, l1
echo "Launching Model K: 15 features, l1..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method l1 \
    2>&1 | tee "$OUTPUT_DIR/model_k_15feat_l1.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_k_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_k_ES_test_results.json" 2>/dev/null || true
echo "Model K complete") &
PIDS+=($!)

# Model L: 15 features, clustered
echo "Launching Model L: 15 features, clustered..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method clustered \
    2>&1 | tee "$OUTPUT_DIR/model_l_15feat_clustered.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_l_ES_best.json" 2>/dev/null || true
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_l_ES_test_results.json" 2>/dev/null || true
echo "Model L complete") &
PIDS+=($!)

#######################################
# WAIT FOR ALL MODELS TO COMPLETE
#######################################

echo ""
echo "All 12 models launched. Waiting for completion..."
echo "PIDs: ${PIDS[@]}"
echo ""

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo ""
echo "=========================================="
echo "All models complete! Analyzing results..."
echo "=========================================="
echo ""

#######################################
# COMPARE PHASE 2 PERFORMANCE
#######################################

echo "Comparing Phase 2 (threshold period) performance..."
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import glob
from pathlib import Path

results = []

# Read all test results
for result_file in glob.glob("outputs/model_comparison/model_*_ES_test_results.json"):
    try:
        with open(result_file) as f:
            data = json.load(f)

        # Extract model name from filename
        model_name = Path(result_file).stem.replace("_ES_test_results", "").upper()

        # Get Phase 2 metrics (threshold period)
        phase2 = data.get("phase2_threshold_metrics", {})

        results.append({
            "model": model_name,
            "sharpe": phase2.get("Sharpe Ratio", 0),
            "sortino": phase2.get("Sortino Ratio", 0),
            "pf": phase2.get("Profit Factor", 0),
            "trades": phase2.get("Trades", 0),
            "pnl": phase2.get("Total PnL", 0),
            "file": result_file
        })
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

# Sort by Sharpe ratio (descending)
results.sort(key=lambda x: x["sharpe"], reverse=True)

print("="*80)
print("PHASE 2 RESULTS (Threshold Period 2019-2020)")
print("="*80)
print(f"{'Model':<15} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'Trades':>7} {'PnL':>12}")
print("-"*80)

for r in results:
    print(f"{r['model']:<15} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
          f"{r['pf']:>6.2f} {r['trades']:>7} ${r['pnl']:>11,.2f}")

print("-"*80)
print("")

if results:
    winner = results[0]
    print(f"🏆 WINNER: {winner['model']}")
    print(f"   Phase 2 Sharpe: {winner['sharpe']:.3f}")
    print(f"   Phase 2 PnL: ${winner['pnl']:,.2f}")
    print("")

    # Read full test results for winner
    with open(winner['file']) as f:
        winner_data = json.load(f)

    phase3 = winner_data.get("phase3_test_metrics", {})

    print("="*80)
    print(f"PHASE 3 RESULTS (Test Period 2021-2024) - {winner['model']}")
    print("="*80)
    print(f"Sharpe Ratio:    {phase3.get('Sharpe Ratio', 0):.3f}")
    print(f"Sortino Ratio:   {phase3.get('Sortino Ratio', 0):.3f}")
    print(f"Profit Factor:   {phase3.get('Profit Factor', 0):.3f}")
    print(f"Win Rate:        {phase3.get('Win Rate', 0):.1f}%")
    print(f"Total PnL:       ${phase3.get('Total PnL', 0):,.2f}")
    print(f"CAGR:            {phase3.get('CAGR', 0):.1f}%")
    print(f"Max Drawdown:    {phase3.get('Max Drawdown', 0):.1f}%")
    print(f"Trades:          {phase3.get('Trades', 0)}")
    print("="*80)
    print("")
    print("✅ Model selection complete!")
    print(f"   Best model JSON: {winner['file'].replace('test_results', 'best')}")
    print("")
else:
    print("❌ No results found!")

PYTHON_SCRIPT

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
