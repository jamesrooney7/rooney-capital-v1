#!/bin/bash
#
# Feature Stability Analysis - Run Before Model Comparison
#
# This script analyzes feature stability across different time periods
# to identify regime-dependent features that should be reviewed before
# running the full model comparison.
#
# Usage:
#   bash research/run_stability_analysis.sh
#

set -e

SYMBOL="ES"
OUTPUT_DIR="outputs/feature_stability"

echo "========================================="
echo "FEATURE STABILITY ANALYSIS"
echo "========================================="
echo ""
echo "This analysis will:"
echo "  1. Split training period (2010-2018) into rolling 4-year windows"
echo "  2. Screen features and train simple models on each window"
echo "  3. Track feature importance and selection consistency"
echo "  4. Flag high-variance and regime-dependent features"
echo ""
echo "This uses ONLY training data - no lookahead bias."
echo ""
echo "Output will be saved to: $OUTPUT_DIR"
echo ""
read -p "Press Enter to continue..."
echo ""

# Run stability analysis
python3 research/analyze_feature_stability.py \
    --symbol "$SYMBOL" \
    --k-features 30 \
    --screen-method importance \
    --window-years 4 \
    --step-years 1 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "========================================="
echo "ANALYSIS COMPLETE"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the stability report: $OUTPUT_DIR/${SYMBOL}_stability_report.txt"
echo "  2. Identify features to filter out based on:"
echo "     - High coefficient of variation (CV > 0.7)"
echo "     - Low selection rate (< 0.5)"
echo "     - Currency features with inconsistent performance"
echo "  3. Update train_rf_three_way_split.py to filter those features"
echo "  4. Run model comparison: bash research/compare_model_configs_parallel.sh"
echo ""
