"""
Reporting utilities for ML meta-labeling system.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List


def generate_executive_summary(
    symbol: str,
    walk_forward_results: List[Dict],
    held_out_results: Dict,
    selected_features: List[str],
    output_path: Path
):
    """
    Generate executive summary report.

    Args:
        symbol: Trading symbol
        walk_forward_results: List of walk-forward window results
        held_out_results: Held-out test results
        selected_features: List of selected feature names
        output_path: Path to save report
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"ML META-LABELING SYSTEM - EXECUTIVE SUMMARY")
    lines.append(f"Symbol: {symbol}")
    lines.append("=" * 100)
    lines.append("")

    # Feature selection summary
    lines.append("FEATURE SELECTION")
    lines.append("-" * 100)
    lines.append(f"  Selected Features: {len(selected_features)}")
    lines.append("")

    # Detect task type from first window
    is_classification = 'auc' in walk_forward_results[0]['test_metrics'] if walk_forward_results else True

    # Walk-forward summary
    lines.append("WALK-FORWARD VALIDATION (2011-2020)")
    lines.append("-" * 100)
    lines.append(f"  Number of Windows: {len(walk_forward_results)}")
    lines.append(f"  Task Type: {'Classification' if is_classification else 'Regression'}")
    lines.append("")

    # Window-by-window results
    lines.append("  Window Results:")
    for window in walk_forward_results:
        lines.append(f"    {window['window_name']}:")
        lines.append(f"      Test Period: {window['test_start']} to {window['test_end']}")

        if is_classification:
            lines.append(f"      Test AUC:    {window['test_metrics']['auc']:.4f}")
        else:
            lines.append(f"      Test R²:     {window['test_metrics'].get('r2', 0):.4f}")
            lines.append(f"      Test MAE:    ${window['test_metrics'].get('mae', 0):.2f}")

        lines.append(f"      Test Sharpe: {window['test_metrics'].get('sharpe', 0):.3f}")
        lines.append(f"      WFE:         {window['wfe']:.2%}")
        lines.append("")

    # Aggregate statistics
    test_sharpes = [w['test_metrics'].get('sharpe', 0) for w in walk_forward_results]
    wfes = [w['wfe'] for w in walk_forward_results]

    lines.append("  Aggregate Statistics:")
    lines.append(f"    Mean Test Sharpe: {pd.Series(test_sharpes).mean():.3f} ± {pd.Series(test_sharpes).std():.3f}")

    if is_classification:
        test_aucs = [w['test_metrics']['auc'] for w in walk_forward_results]
        lines.append(f"    Mean Test AUC:    {pd.Series(test_aucs).mean():.4f} ± {pd.Series(test_aucs).std():.4f}")
    else:
        test_r2s = [w['test_metrics'].get('r2', 0) for w in walk_forward_results]
        test_maes = [w['test_metrics'].get('mae', 0) for w in walk_forward_results]
        lines.append(f"    Mean Test R²:     {pd.Series(test_r2s).mean():.4f} ± {pd.Series(test_r2s).std():.4f}")
        lines.append(f"    Mean Test MAE:    ${pd.Series(test_maes).mean():.2f} ± ${pd.Series(test_maes).std():.2f}")

    lines.append(f"    Mean WFE:         {pd.Series(wfes).mean():.2%} ± {pd.Series(wfes).std():.2%}")
    lines.append(f"    Positive Windows: {sum(s > 0 for s in test_sharpes)}/{len(test_sharpes)}")
    lines.append("")

    # Held-out test results
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
        lines.append("")

    lines.append("=" * 100)

    # Write to file
    output_path.write_text("\n".join(lines))


def save_walk_forward_results_csv(
    walk_forward_results: List[Dict],
    output_path: Path
):
    """
    Save walk-forward results to CSV.

    Args:
        walk_forward_results: List of window results
        output_path: Path to save CSV
    """
    data = []
    for window in walk_forward_results:
        # Detect task type by checking which metrics are present
        test_metrics = window['test_metrics']
        is_classification = 'auc' in test_metrics

        row = {
            'Window': window['window_name'],
            'Train_Start': window['train_start'],
            'Train_End': window['train_end'],
            'Test_Start': window['test_start'],
            'Test_End': window['test_end'],
            'Train_Samples': window['train_samples'],
            'Test_Samples': window['test_samples'],
            'CV_Score': window['cv_score'],
            'Test_Sharpe': test_metrics.get('sharpe', 0),
            'WFE': window['wfe']
        }

        # Add task-specific metrics
        if is_classification:
            row.update({
                'Test_AUC': test_metrics['auc'],
                'Test_Win_Rate': test_metrics['win_rate'],
                'Test_Precision': test_metrics.get('precision', 0),
                'Test_Recall': test_metrics.get('recall', 0),
            })
        else:  # regression
            row.update({
                'Test_R2': test_metrics.get('r2', 0),
                'Test_MAE': test_metrics.get('mae', 0),
                'Test_RMSE': test_metrics.get('rmse', 0),
                'Mean_True_PnL': test_metrics.get('mean_true_pnl', 0),
                'Mean_Pred_PnL': test_metrics.get('mean_pred_pnl', 0),
            })

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def save_hyperparameter_stability_csv(
    walk_forward_results: List[Dict],
    output_path: Path
):
    """
    Save hyperparameter stability analysis to CSV.

    Args:
        walk_forward_results: List of window results
        output_path: Path to save CSV
    """
    data = []
    for window in walk_forward_results:
        row = {
            'Window': window['window_name'],
            **window['best_hyperparameters']
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
