#!/bin/bash
# Compare multiple model configurations for ES
# Pick the best based on Phase 2 (threshold period 2019-2020)
# Then evaluate winner on Phase 3 (test period 2021-2024)

set -e  # Exit on error

SYMBOL="ES"
TRAIN_END="2018-12-31"
THRESHOLD_END="2020-12-31"
RS_TRIALS=50
BO_TRIALS=150

echo "=========================================="
echo "ES Model Configuration Comparison"
echo "=========================================="
echo ""
echo "Training 8 different configurations:"
echo ""
echo "Feature Count Sweep (importance screening):"
echo "  Model A: 15 features  (simple - less overfitting risk)"
echo "  Model B: 30 features  (baseline)"
echo "  Model C: 50 features  (more complex)"
echo "  Model D: 75 features  (most complex)"
echo ""
echo "Screening Method Sweep (30 features):"
echo "  Model E: importance   (tree-based MDI)"
echo "  Model F: permutation  (permutation importance)"
echo "  Model G: l1           (linear sparse)"
echo ""
echo "CPCV Configuration:"
echo "  Model H: Conservative (fewer folds, more embargo)"
echo ""
echo "Selection: Pick best based on Phase 2 (2019-2020)"
echo "Evaluation: Report winner's Phase 3 (2021-2024) performance"
echo ""
echo "Estimated runtime: ~40 hours total (~5 hours per model)"
echo "=========================================="
echo ""

# Create output directory
mkdir -p outputs/model_comparison
OUTPUT_DIR="outputs/model_comparison"

#######################################
# FEATURE COUNT SWEEP
#######################################

# Model A: 15 features, importance
echo "=========================================="
echo "Training Model A: 15 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_a_15feat_importance.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_a_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_a_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_a_ES_test_results.json"
echo "Model A complete. Results saved."
echo ""
sleep 2

# Model B: 30 features, importance
echo "=========================================="
echo "Training Model B: 30 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_b_30feat_importance.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_b_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_b_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_b_ES_test_results.json"
echo "Model B complete. Results saved."
echo ""
sleep 2

# Model C: 50 features, importance
echo "=========================================="
echo "Training Model C: 50 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 50 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_c_50feat_importance.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_c_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_c_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_c_ES_test_results.json"
echo "Model C complete. Results saved."
echo ""
sleep 2

# Model D: 75 features, importance
echo "=========================================="
echo "Training Model D: 75 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 75 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_d_75feat_importance.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_d_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_d_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_d_ES_test_results.json"
echo "Model D complete. Results saved."
echo ""
sleep 2

#######################################
# SCREENING METHOD SWEEP
#######################################

# Model E: 30 features, permutation
echo "=========================================="
echo "Training Model E: 30 features, permutation"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_e_30feat_permutation.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_e_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_e_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_e_ES_test_results.json"
echo "Model E complete. Results saved."
echo ""
sleep 2

# Model F: 30 features, l1
echo "=========================================="
echo "Training Model F: 30 features, l1"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method l1 \
    2>&1 | tee "$OUTPUT_DIR/model_f_30feat_l1.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_f_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_f_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_f_ES_test_results.json"
echo "Model F complete. Results saved."
echo ""
sleep 2

#######################################
# CPCV CONFIGURATION VARIANT
#######################################

# Model G: 30 features, importance, conservative CPCV
echo "=========================================="
echo "Training Model G: 30 features, conservative CPCV"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    --folds 3 \
    --k-test 1 \
    --embargo-days 5 \
    2>&1 | tee "$OUTPUT_DIR/model_g_30feat_conservative.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_g_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_g_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_g_ES_test_results.json"
echo "Model G complete. Results saved."
echo ""
sleep 2

# Model H: 30 features, importance, aggressive CPCV
echo "=========================================="
echo "Training Model H: 30 features, aggressive CPCV"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    --folds 10 \
    --k-test 3 \
    --embargo-days 1 \
    2>&1 | tee "$OUTPUT_DIR/model_h_30feat_aggressive.log"

cp src/models/ES_best.json "$OUTPUT_DIR/model_h_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_h_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_h_ES_test_results.json"
echo "Model H complete. Results saved."
echo ""

#######################################
# COMPARISON & RESULTS
#######################################

echo ""
echo "=========================================="
echo "COMPARISON RESULTS"
echo "=========================================="
echo ""
echo "Comparing Phase 2 (Threshold Period 2019-2020) performance:"
echo ""

python3 << 'EOF'
import json
import os

output_dir = "outputs/model_comparison"

models = [
    ("Model A (15 feat, importance)", f"{output_dir}/model_a_ES_best.json"),
    ("Model B (30 feat, importance)", f"{output_dir}/model_b_ES_best.json"),
    ("Model C (50 feat, importance)", f"{output_dir}/model_c_ES_best.json"),
    ("Model D (75 feat, importance)", f"{output_dir}/model_d_ES_best.json"),
    ("Model E (30 feat, permutation)", f"{output_dir}/model_e_ES_best.json"),
    ("Model F (30 feat, l1)", f"{output_dir}/model_f_ES_best.json"),
    ("Model G (30 feat, conservative)", f"{output_dir}/model_g_ES_best.json"),
    ("Model H (30 feat, aggressive)", f"{output_dir}/model_h_ES_best.json"),
]

print("Phase 2 (Threshold Period 2019-2020) Results:")
print("-" * 90)
print(f"{'Model':<38} {'Sharpe':>10} {'PF':>10} {'Win%':>10} {'Trades':>10}")
print("-" * 90)

best_sharpe = -999
best_model = None
best_idx = None

for idx, (name, path) in enumerate(models):
    with open(path, 'r') as f:
        data = json.load(f)

    threshold_opt = data.get("threshold_optimization", {})
    sharpe = threshold_opt.get("sharpe", 0.0)
    pf = threshold_opt.get("profit_factor", 0.0)
    win_rate = threshold_opt.get("win_rate", 0.0) * 100
    trades = threshold_opt.get("trades", 0)

    marker = " 🏆" if sharpe > best_sharpe else ""
    print(f"{name:<38} {sharpe:>10.3f} {pf:>10.3f} {win_rate:>9.1f}% {trades:>10}{marker}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_model = name
        best_idx = idx

print("-" * 90)
print(f"\n🏆 WINNER: {best_model} (Sharpe {best_sharpe:.3f})")
print("\nThis is the model you should deploy based on unbiased threshold period validation.")
print("\nPhase 3 (Test Period 2021-2024) - TRUE OUT-OF-SAMPLE PERFORMANCE:")
print("-" * 90)

# Show test results for winner
test_path = f"{output_dir}/model_{chr(97 + best_idx)}_ES_test_results.json"
with open(test_path, 'r') as f:
    test_data = json.load(f)

test_metrics = test_data.get("test_metrics", {})
print(f"Trades:        {test_metrics.get('trades', 0)}")
print(f"Sharpe:        {test_metrics.get('sharpe', 0.0):.3f}")
print(f"Sortino:       {test_metrics.get('sortino', 0.0):.3f}")
print(f"Profit Factor: {test_metrics.get('profit_factor', 0.0):.3f}")
print(f"Win Rate:      {test_metrics.get('win_rate', 0.0) * 100:.1f}%")
print(f"Total PnL:     ${test_metrics.get('total_pnl_usd', 0.0):,.2f}")
print(f"CAGR:          {test_metrics.get('cagr', 0.0) * 100:.1f}%")
print(f"Max DD:        {test_metrics.get('max_drawdown_pct', 0.0) * 100:.1f}%")
print("-" * 90)

# Copy winner to main models directory
winner_json = f"{output_dir}/model_{chr(97 + best_idx)}_ES_best.json"
winner_pkl = f"{output_dir}/model_{chr(97 + best_idx)}_ES_rf_model.pkl"
print(f"\n✅ To deploy winner:")
print(f"   cp {winner_json} src/models/ES_best.json")
print(f"   cp {winner_pkl} src/models/ES_rf_model.pkl")
EOF

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "All models saved in: $OUTPUT_DIR"
echo "=========================================="
