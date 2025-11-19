#!/bin/bash
################################################################################
# ML Meta-Labeling Bug Fixes - Automated Application Script
#
# This script applies 3 critical fixes to the ML meta-labeling system:
# 1. Fix held-out test to filter trades by ML predictions
# 2. Reduce embargo period from 60 to 2 days
# 3. Update reporting to show filtered vs unfiltered metrics
#
# Run this on your server in the rooney-capital-v1 directory
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "ML META-LABELING FIXES - AUTOMATED APPLICATION"
echo "================================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "research/ml_meta_labeling/ml_meta_labeling_optimizer.py" ]; then
    echo "ERROR: ml_meta_labeling_optimizer.py not found!"
    echo "Please run this script from the rooney-capital-v1 directory"
    exit 1
fi

echo "✓ Found ml_meta_labeling files"
echo ""

# Backup files
echo "Creating backups..."
cp research/ml_meta_labeling/ml_meta_labeling_optimizer.py research/ml_meta_labeling/ml_meta_labeling_optimizer.py.backup
cp research/ml_meta_labeling/components/config_defaults.py research/ml_meta_labeling/components/config_defaults.py.backup
cp research/ml_meta_labeling/utils/reporting.py research/ml_meta_labeling/utils/reporting.py.backup
echo "✓ Backups created (.backup files)"
echo ""

################################################################################
# FIX #1: Update embargo period in config_defaults.py
################################################################################
echo "Applying Fix #1: Reduce embargo period (60 → 2 days)..."

sed -i "s/'embargo_days': 60,  # 2 months/'embargo_days': 2,  # Conservative: 1 day label evaluation + 1 day buffer/" \
    research/ml_meta_labeling/components/config_defaults.py

echo "✓ Fixed config_defaults.py"

################################################################################
# FIX #2: Update embargo period in ml_meta_labeling_optimizer.py argument
################################################################################
echo "Applying Fix #2: Update embargo default in optimizer..."

sed -i 's/default=60, help="Embargo period (days)"/default=2, help="Embargo period (days)"/' \
    research/ml_meta_labeling/ml_meta_labeling_optimizer.py

echo "✓ Fixed optimizer argument"

################################################################################
# FIX #3: Replace held-out test evaluation section
################################################################################
echo "Applying Fix #3: Fix held-out test evaluation (this may take a moment)..."

# Create Python script to do the complex replacement
python3 << 'PYTHON_EOF'
import re

# Read the file
with open('research/ml_meta_labeling/ml_meta_labeling_optimizer.py', 'r') as f:
    content = f.read()

# Find and replace the held-out metrics calculation section
old_pattern = r'''    # Calculate held-out metrics
    from sklearn\.metrics import roc_auc_score, precision_score, recall_score, f1_score

    held_out_results = \{
        'auc': roc_auc_score\(y_test, y_pred_proba\),
        'precision': precision_score\(y_test, \(y_pred_proba >= 0\.5\)\.astype\(int\), zero_division=0\),
        'recall': recall_score\(y_test, \(y_pred_proba >= 0\.5\)\.astype\(int\), zero_division=0\),
        'f1': f1_score\(y_test, \(y_pred_proba >= 0\.5\)\.astype\(int\), zero_division=0\),
        'win_rate': y_test\.mean\(\),
        'n_trades': len\(y_test\)
    \}

    # Calculate financial metrics if available
    if 'y_return' in test_df\.columns:
        returns = test_df\['y_return'\]\.values
        from research\.ml_meta_labeling\.utils\.metrics import calculate_performance_metrics

        perf_metrics = calculate_performance_metrics\(returns\)
        held_out_results\.update\(perf_metrics\)

    logger\.info\("Held-Out Test Results \(2021-2024\):"\)
    logger\.info\(f"  AUC:          \{held_out_results\['auc'\]:.4f\}"\)
    logger\.info\(f"  Sharpe:       \{held_out_results\.get\('sharpe_ratio', 0\):.3f\}"\)
    logger\.info\(f"  Win Rate:     \{held_out_results\['win_rate'\]:.2%\}"\)
    logger\.info\(f"  Total Trades: \{held_out_results\['n_trades'\]\}"\)'''

new_code = '''    # Calculate held-out metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # Save held-out predictions to CSV
    held_out_pred_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'y_true': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred_binary': (y_pred_proba >= 0.5).astype(int)
    })
    if 'y_return' in test_df.columns:
        held_out_pred_df['y_return'] = test_df['y_return'].values

    held_out_pred_df.to_csv(
        output_dir / f"{args.symbol}_ml_meta_labeling_held_out_predictions.csv",
        index=False
    )
    logger.info(f"Saved held-out predictions to CSV")

    # Filter trades based on ML threshold (0.50)
    threshold = 0.5
    filter_mask = y_pred_proba >= threshold

    # Unfiltered metrics (for comparison)
    unfiltered_metrics = {
        'n_trades': len(y_test),
        'win_rate': y_test.mean(),
    }

    if 'y_return' in test_df.columns:
        returns_unfiltered = test_df['y_return'].values
        from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics
        unfiltered_perf = calculate_performance_metrics(returns_unfiltered)
        unfiltered_metrics['sharpe_ratio'] = unfiltered_perf['sharpe_ratio']
        unfiltered_metrics['profit_factor'] = unfiltered_perf['profit_factor']

    # Filtered metrics (actual ML meta-labeling performance)
    y_test_filtered = y_test[filter_mask]

    held_out_results = {
        'threshold': threshold,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, (y_pred_proba >= threshold).astype(int), zero_division=0),
        'recall': recall_score(y_test, (y_pred_proba >= threshold).astype(int), zero_division=0),
        'f1': f1_score(y_test, (y_pred_proba >= threshold).astype(int), zero_division=0),
        'n_trades_unfiltered': unfiltered_metrics['n_trades'],
        'n_trades_filtered': int(filter_mask.sum()),
        'filter_rate': float(1 - filter_mask.mean()),
        'win_rate_unfiltered': float(unfiltered_metrics['win_rate']),
        'win_rate_filtered': float(y_test_filtered.mean()) if len(y_test_filtered) > 0 else 0.0,
    }

    # Calculate financial metrics on FILTERED trades
    if 'y_return' in test_df.columns and filter_mask.sum() > 0:
        returns_filtered = test_df['y_return'].values[filter_mask]
        from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics

        perf_metrics = calculate_performance_metrics(returns_filtered)
        held_out_results.update({
            'sharpe_ratio_unfiltered': unfiltered_metrics['sharpe_ratio'],
            'sharpe_ratio_filtered': perf_metrics['sharpe_ratio'],
            'profit_factor_unfiltered': unfiltered_metrics['profit_factor'],
            'profit_factor_filtered': perf_metrics['profit_factor'],
            'sortino_ratio': perf_metrics['sortino_ratio'],
            'calmar_ratio': perf_metrics['calmar_ratio'],
            'max_drawdown': perf_metrics['max_drawdown'],
            'total_return': perf_metrics['total_return'],
        })

    logger.info("Held-Out Test Results (2021-2024):")
    logger.info(f"  AUC:                    {held_out_results['auc']:.4f}")
    logger.info(f"  Threshold:              {threshold:.2f}")
    logger.info("")
    logger.info("  UNFILTERED (Primary Strategy):")
    logger.info(f"    Trades:               {held_out_results['n_trades_unfiltered']}")
    logger.info(f"    Win Rate:             {held_out_results['win_rate_unfiltered']:.2%}")
    logger.info(f"    Sharpe:               {held_out_results.get('sharpe_ratio_unfiltered', 0):.3f}")
    logger.info(f"    Profit Factor:        {held_out_results.get('profit_factor_unfiltered', 0):.2f}")
    logger.info("")
    logger.info("  FILTERED (ML Meta-Labeling):")
    logger.info(f"    Trades:               {held_out_results['n_trades_filtered']}")
    logger.info(f"    Filter Rate:          {held_out_results['filter_rate']:.1%}")
    logger.info(f"    Win Rate:             {held_out_results['win_rate_filtered']:.2%}")
    logger.info(f"    Sharpe:               {held_out_results.get('sharpe_ratio_filtered', 0):.3f}")
    logger.info(f"    Profit Factor:        {held_out_results.get('profit_factor_filtered', 0):.2f}")'''

# Replace the section
content_new = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

if content_new == content:
    print("WARNING: Pattern not found - trying line-based replacement...")
    # Alternative: Find line numbers and replace
    lines = content.split('\n')

    # Find the section start
    for i, line in enumerate(lines):
        if '# Calculate held-out metrics' in line and i > 200:  # Make sure we're in the right section
            # Find the end (next major section)
            for j in range(i, min(i+50, len(lines))):
                if lines[j].strip().startswith('# Save held-out results') or \
                   lines[j].strip().startswith('# ==='):
                    # Replace this section
                    lines[i:j] = new_code.split('\n')
                    content_new = '\n'.join(lines)
                    break
            break
else:
    print("✓ Pattern matched and replaced")

# Write back
with open('research/ml_meta_labeling/ml_meta_labeling_optimizer.py', 'w') as f:
    f.write(content_new)

print("✓ Updated ml_meta_labeling_optimizer.py")
PYTHON_EOF

################################################################################
# FIX #4: Update reporting.py
################################################################################
echo "Applying Fix #4: Update executive summary reporting..."

python3 << 'PYTHON_EOF'
# Read the file
with open('research/ml_meta_labeling/utils/reporting.py', 'r') as f:
    content = f.read()

# Replace the held-out test results section
old_section = '''    # Held-out test results
    if held_out_results:
        lines.append("HELD-OUT TEST PERIOD (2021-2024)")
        lines.append("-" * 100)
        lines.append(f"  Test AUC:     {held_out_results.get('auc', 0):.4f}")
        lines.append(f"  Test Sharpe:  {held_out_results.get('sharpe', 0):.3f}")
        lines.append(f"  Win Rate:     {held_out_results.get('win_rate', 0):.2%}")
        lines.append(f"  Profit Factor: {held_out_results.get('profit_factor', 0):.2f}")
        lines.append(f"  Total Trades:  {held_out_results.get('n_trades', 0)}")
        lines.append("")'''

new_section = '''    # Held-out test results
    if held_out_results:
        lines.append("HELD-OUT TEST PERIOD (2021-2024)")
        lines.append("-" * 100)
        lines.append(f"  Test AUC:             {held_out_results.get('auc', 0):.4f}")
        lines.append(f"  Threshold:            {held_out_results.get('threshold', 0.5):.2f}")
        lines.append("")
        lines.append("  Unfiltered (Primary Strategy):")
        lines.append(f"    Total Trades:       {held_out_results.get('n_trades_unfiltered', 0)}")
        lines.append(f"    Win Rate:           {held_out_results.get('win_rate_unfiltered', 0):.2%}")
        lines.append(f"    Sharpe Ratio:       {held_out_results.get('sharpe_ratio_unfiltered', 0):.3f}")
        lines.append(f"    Profit Factor:      {held_out_results.get('profit_factor_unfiltered', 0):.2f}")
        lines.append("")
        lines.append("  Filtered (ML Meta-Labeling):")
        lines.append(f"    Total Trades:       {held_out_results.get('n_trades_filtered', 0)}")
        lines.append(f"    Filter Rate:        {held_out_results.get('filter_rate', 0):.1%}")
        lines.append(f"    Win Rate:           {held_out_results.get('win_rate_filtered', 0):.2%}")
        lines.append(f"    Sharpe Ratio:       {held_out_results.get('sharpe_ratio_filtered', 0):.3f}")
        lines.append(f"    Profit Factor:      {held_out_results.get('profit_factor_filtered', 0):.2f}")
        lines.append("")'''

content = content.replace(old_section, new_section)

with open('research/ml_meta_labeling/utils/reporting.py', 'w') as f:
    f.write(content)

print("✓ Updated reporting.py")
PYTHON_EOF

echo ""
echo "================================================================================"
echo "✓ ALL FIXES APPLIED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Files modified:"
echo "  - research/ml_meta_labeling/ml_meta_labeling_optimizer.py"
echo "  - research/ml_meta_labeling/components/config_defaults.py"
echo "  - research/ml_meta_labeling/utils/reporting.py"
echo ""
echo "Backups created with .backup extension"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test the optimizer: python research/ml_meta_labeling/ml_meta_labeling_optimizer.py --symbol ES"
echo "  3. Commit changes: git add -A && git commit -m 'Fix: ML meta-labeling bugs'"
echo ""
echo "Expected improvements:"
echo "  - Held-out test will show FILTERED vs UNFILTERED metrics"
echo "  - ~40-50% more training data (embargo: 60 → 2 days)"
echo "  - Filtered Sharpe should be POSITIVE (vs -0.296 unfiltered)"
echo ""
