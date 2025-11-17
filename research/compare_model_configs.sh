#!/bin/bash
# Compare 3 different model configurations for ES
# Pick the best based on Phase 2 (threshold period 2019-2020)
# Then evaluate winner on Phase 3 (test period 2021-2024)

set -e  # Exit on error

SYMBOL="ES"
TRAIN_END="2018-12-31"
THRESHOLD_END="2020-12-31"
RS_TRIALS=120
BO_TRIALS=300

echo "=========================================="
echo "ES Model Configuration Comparison"
echo "=========================================="
echo ""
echo "Training 3 different configurations:"
echo "  Model A: 30 features, importance screening"
echo "  Model B: 50 features, importance screening"
echo "  Model C: 30 features, permutation screening"
echo ""
echo "Selection: Pick best based on Phase 2 (2019-2020)"
echo "Evaluation: Report winner's Phase 3 (2021-2024) performance"
echo ""
echo "=========================================="
echo ""

# Create output directory
mkdir -p outputs/model_comparison
OUTPUT_DIR="outputs/model_comparison"

# Model A: 30 features, importance screening (current approach)
echo "=========================================="
echo "Training Model A: 30 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_a_30feat_importance.log"

# Save Model A results
cp src/models/ES_best.json "$OUTPUT_DIR/model_a_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_a_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_a_ES_test_results.json"

echo ""
echo "Model A complete. Results saved to $OUTPUT_DIR/model_a_*"
echo ""
sleep 2

# Model B: 50 features, importance screening
echo "=========================================="
echo "Training Model B: 50 features, importance"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 50 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_b_50feat_importance.log"

# Save Model B results
cp src/models/ES_best.json "$OUTPUT_DIR/model_b_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_b_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_b_ES_test_results.json"

echo ""
echo "Model B complete. Results saved to $OUTPUT_DIR/model_b_*"
echo ""
sleep 2

# Model C: 30 features, permutation screening
echo "=========================================="
echo "Training Model C: 30 features, permutation"
echo "=========================================="
python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_c_30feat_permutation.log"

# Save Model C results
cp src/models/ES_best.json "$OUTPUT_DIR/model_c_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_c_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_c_ES_test_results.json"

echo ""
echo "Model C complete. Results saved to $OUTPUT_DIR/model_c_*"
echo ""

# Compare results
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
    ("Model A (30 feat, importance)", f"{output_dir}/model_a_ES_best.json"),
    ("Model B (50 feat, importance)", f"{output_dir}/model_b_ES_best.json"),
    ("Model C (30 feat, permutation)", f"{output_dir}/model_c_ES_best.json"),
]

print("Phase 2 (Threshold Period 2019-2020) Results:")
print("-" * 80)
print(f"{'Model':<35} {'Sharpe':>10} {'PF':>10} {'Win%':>10} {'Trades':>10}")
print("-" * 80)

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
    print(f"{name:<35} {sharpe:>10.3f} {pf:>10.3f} {win_rate:>9.1f}% {trades:>10}{marker}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_model = name
        best_idx = idx

print("-" * 80)
print(f"\n🏆 WINNER: {best_model} (Sharpe {best_sharpe:.3f})")
print("\nThis is the model you should deploy based on unbiased threshold period validation.")
print("\nPhase 3 (Test Period 2021-2024) - TRUE OUT-OF-SAMPLE PERFORMANCE:")
print("-" * 80)

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
print("-" * 80)

# Copy winner to main models directory
winner_json = f"{output_dir}/model_{chr(97 + best_idx)}_ES_best.json"
winner_pkl = f"{output_dir}/model_{chr(97 + best_idx)}_ES_rf_model.pkl"
print(f"\n✅ To deploy winner: cp {winner_json} src/models/ES_best.json")
print(f"✅ To deploy winner: cp {winner_pkl} src/models/ES_rf_model.pkl")
EOF

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "All models saved in: $OUTPUT_DIR"
echo "=========================================="
