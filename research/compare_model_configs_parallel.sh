#!/bin/bash
# Compare up to 16 model configurations for ES IN PARALLEL
# Pick the best based on Phase 2 (threshold period 2019-2020)
# Then evaluate winner on Phase 3 (test period 2021-2024)

set -e  # Exit on error

SYMBOL="ES"
TRAIN_END="2018-12-31"
THRESHOLD_END="2020-12-31"
RS_TRIALS=50
BO_TRIALS=150

echo "=========================================="
echo "ES Model Configuration Comparison (PARALLEL)"
echo "=========================================="
echo ""
echo "Training 16 different configurations IN PARALLEL:"
echo ""
echo "Feature Count Sweep (importance):"
echo "  Model A: 10 features   |  Model I:  20 features"
echo "  Model B: 15 features   |  Model J:  40 features"
echo "  Model C: 30 features   |  Model K:  60 features"
echo "  Model D: 50 features   |  Model L: 100 features"
echo ""
echo "Screening Method Sweep (30 features):"
echo "  Model E: importance    |  Model M: l1"
echo "  Model F: permutation   |  Model N: clustered"
echo ""
echo "CPCV Configuration Variants (30 features, importance):"
echo "  Model G: Conservative (3 folds, k=1, emb=5)"
echo "  Model H: Aggressive   (10 folds, k=3, emb=1)"
echo "  Model O: Very Conservative (3 folds, k=1, emb=10)"
echo "  Model P: Very Aggressive (15 folds, k=5, emb=0)"
echo ""
echo "Selection: Pick best based on Phase 2 (2019-2020)"
echo "Evaluation: Report winner's Phase 3 (2021-2024) performance"
echo ""
echo "⚠️  WARNING: Running 16 models in parallel is CPU/memory intensive!"
echo "    Recommended: 32+ CPU cores, 64+ GB RAM"
echo "    Estimated runtime: ~5-6 hours (if resources available)"
echo "=========================================="
echo ""

# Create output directory
mkdir -p outputs/model_comparison
OUTPUT_DIR="outputs/model_comparison"

# Array to store background PIDs
declare -a PIDS

echo "Launching all 16 models in parallel..."
echo ""

#######################################
# FEATURE COUNT SWEEP (Models A-D, I-L)
#######################################

# Model A: 10 features
echo "Launching Model A: 10 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 10 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_a_10feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_a_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_a_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_a_ES_test_results.json"
echo "Model A complete") &
PIDS+=($!)

# Model B: 15 features
echo "Launching Model B: 15 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 15 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_b_15feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_b_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_b_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_b_ES_test_results.json"
echo "Model B complete") &
PIDS+=($!)

# Model C: 30 features
echo "Launching Model C: 30 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_c_30feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_c_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_c_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_c_ES_test_results.json"
echo "Model C complete") &
PIDS+=($!)

# Model D: 50 features
echo "Launching Model D: 50 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 50 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_d_50feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_d_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_d_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_d_ES_test_results.json"
echo "Model D complete") &
PIDS+=($!)

#######################################
# SCREENING METHOD SWEEP (Models E-F, M-N)
#######################################

# Model E: 30 features, importance (baseline)
echo "Launching Model E: 30 features, importance..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_e_30feat_importance2.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_e_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_e_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_e_ES_test_results.json"
echo "Model E complete") &
PIDS+=($!)

# Model F: 30 features, permutation
echo "Launching Model F: 30 features, permutation..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method permutation \
    2>&1 | tee "$OUTPUT_DIR/model_f_30feat_permutation.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_f_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_f_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_f_ES_test_results.json"
echo "Model F complete") &
PIDS+=($!)

#######################################
# CPCV CONFIGURATION VARIANTS (Models G-H, O-P)
#######################################

# Model G: Conservative CPCV
echo "Launching Model G: 30 features, conservative CPCV..."
(python3 research/train_rf_three_way_split.py \
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
echo "Model G complete") &
PIDS+=($!)

# Model H: Aggressive CPCV
echo "Launching Model H: 30 features, aggressive CPCV..."
(python3 research/train_rf_three_way_split.py \
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
echo "Model H complete") &
PIDS+=($!)

# Model I: 20 features
echo "Launching Model I: 20 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 20 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_i_20feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_i_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_i_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_i_ES_test_results.json"
echo "Model I complete") &
PIDS+=($!)

# Model J: 40 features
echo "Launching Model J: 40 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 40 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_j_40feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_j_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_j_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_j_ES_test_results.json"
echo "Model J complete") &
PIDS+=($!)

# Model K: 60 features
echo "Launching Model K: 60 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 60 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_k_60feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_k_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_k_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_k_ES_test_results.json"
echo "Model K complete") &
PIDS+=($!)

# Model L: 100 features
echo "Launching Model L: 100 features..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 100 \
    --screen-method importance \
    2>&1 | tee "$OUTPUT_DIR/model_l_100feat_importance.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_l_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_l_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_l_ES_test_results.json"
echo "Model L complete") &
PIDS+=($!)

# Model M: 30 features, l1
echo "Launching Model M: 30 features, l1..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method l1 \
    2>&1 | tee "$OUTPUT_DIR/model_m_30feat_l1.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_m_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_m_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_m_ES_test_results.json"
echo "Model M complete") &
PIDS+=($!)

# Model N: 30 features, clustered
echo "Launching Model N: 30 features, clustered..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method clustered \
    2>&1 | tee "$OUTPUT_DIR/model_n_30feat_clustered.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_n_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_n_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_n_ES_test_results.json"
echo "Model N complete") &
PIDS+=($!)

# Model O: Very Conservative CPCV
echo "Launching Model O: 30 features, very conservative CPCV..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    --folds 3 \
    --k-test 1 \
    --embargo-days 10 \
    2>&1 | tee "$OUTPUT_DIR/model_o_30feat_veryconservative.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_o_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_o_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_o_ES_test_results.json"
echo "Model O complete") &
PIDS+=($!)

# Model P: Very Aggressive CPCV
echo "Launching Model P: 30 features, very aggressive CPCV..."
(python3 research/train_rf_three_way_split.py \
    --symbol $SYMBOL \
    --train-end $TRAIN_END \
    --threshold-end $THRESHOLD_END \
    --rs-trials $RS_TRIALS \
    --bo-trials $BO_TRIALS \
    --k-features 30 \
    --screen-method importance \
    --folds 15 \
    --k-test 5 \
    --embargo-days 0 \
    2>&1 | tee "$OUTPUT_DIR/model_p_30feat_veryaggressive.log"
cp src/models/ES_best.json "$OUTPUT_DIR/model_p_ES_best.json"
cp src/models/ES_rf_model.pkl "$OUTPUT_DIR/model_p_ES_rf_model.pkl"
cp src/models/ES_test_results.json "$OUTPUT_DIR/model_p_ES_test_results.json"
echo "Model P complete") &
PIDS+=($!)

echo ""
echo "All 16 models launched! Waiting for completion..."
echo "PIDs: ${PIDS[@]}"
echo ""

# Wait for all background processes
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo ""
echo "All models finished training!"
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
    ("Model A (10 feat, importance)", f"{output_dir}/model_a_ES_best.json"),
    ("Model B (15 feat, importance)", f"{output_dir}/model_b_ES_best.json"),
    ("Model C (30 feat, importance)", f"{output_dir}/model_c_ES_best.json"),
    ("Model D (50 feat, importance)", f"{output_dir}/model_d_ES_best.json"),
    ("Model E (30 feat, importance)", f"{output_dir}/model_e_ES_best.json"),
    ("Model F (30 feat, permutation)", f"{output_dir}/model_f_ES_best.json"),
    ("Model G (30 feat, conservative)", f"{output_dir}/model_g_ES_best.json"),
    ("Model H (30 feat, aggressive)", f"{output_dir}/model_h_ES_best.json"),
    ("Model I (20 feat, importance)", f"{output_dir}/model_i_ES_best.json"),
    ("Model J (40 feat, importance)", f"{output_dir}/model_j_ES_best.json"),
    ("Model K (60 feat, importance)", f"{output_dir}/model_k_ES_best.json"),
    ("Model L (100 feat, importance)", f"{output_dir}/model_l_ES_best.json"),
    ("Model M (30 feat, l1)", f"{output_dir}/model_m_ES_best.json"),
    ("Model N (30 feat, clustered)", f"{output_dir}/model_n_ES_best.json"),
    ("Model O (30 feat, very conservative)", f"{output_dir}/model_o_ES_best.json"),
    ("Model P (30 feat, very aggressive)", f"{output_dir}/model_p_ES_best.json"),
]

print("Phase 2 (Threshold Period 2019-2020) Results:")
print("-" * 95)
print(f"{'Model':<40} {'Sharpe':>10} {'PF':>10} {'Win%':>10} {'Trades':>10}")
print("-" * 95)

best_sharpe = -999
best_model = None
best_idx = None

for idx, (name, path) in enumerate(models):
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        threshold_opt = data.get("threshold_optimization", {})
        sharpe = threshold_opt.get("sharpe", 0.0)
        pf = threshold_opt.get("profit_factor", 0.0)
        win_rate = threshold_opt.get("win_rate", 0.0) * 100
        trades = threshold_opt.get("trades", 0)

        marker = " 🏆" if sharpe > best_sharpe else ""
        print(f"{name:<40} {sharpe:>10.3f} {pf:>10.3f} {win_rate:>9.1f}% {trades:>10}{marker}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_model = name
            best_idx = idx
    except FileNotFoundError:
        print(f"{name:<40} {'ERROR: File not found':>42}")

print("-" * 95)
print(f"\n🏆 WINNER: {best_model} (Sharpe {best_sharpe:.3f})")
print("\nThis is the model you should deploy based on unbiased threshold period validation.")
print("\nPhase 3 (Test Period 2021-2024) - TRUE OUT-OF-SAMPLE PERFORMANCE:")
print("-" * 95)

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
print("-" * 95)

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
