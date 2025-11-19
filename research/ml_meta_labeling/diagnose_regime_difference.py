#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Regime Differences Between Periods

Investigates why held-out period (2021-2024) performed spectacularly (Sharpe 5.7)
while walk-forward periods (2016-2020) were mostly negative (mean Sharpe -0.61).

Usage:
    python research/ml_meta_labeling/diagnose_regime_difference.py --symbol ES
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics

def load_data(symbol: str):
    """Load OOS predictions and training data."""
    results_dir = Path(f"research/ml_meta_labeling/results/{symbol}")

    # Load OOS predictions
    oos_pred_file = results_dir / f"{symbol}_ml_meta_labeling_oos_predictions.csv"
    oos_df = pd.read_csv(oos_pred_file)
    oos_df['Date'] = pd.to_datetime(oos_df['Date'])

    # Load original training data to get P&L
    training_file = Path(f"data/training/{symbol}_transformed_features.csv")
    training_df = pd.read_csv(training_file)
    training_df['Entry_Date'] = pd.to_datetime(training_df['Entry_Date'])

    # Merge to get P&L information
    merged = oos_df.merge(
        training_df[['Entry_Date', 'Trade_PnL_Points', 'Trade_PnL_Pct']],
        left_on='Date',
        right_on='Entry_Date',
        how='left'
    )

    return merged

def analyze_period(df: pd.DataFrame, period_name: str, start_date: str, end_date: str, threshold: float = 0.5):
    """Analyze a specific time period."""
    print(f"\n{'='*100}")
    print(f"{period_name}: {start_date} to {end_date}")
    print(f"{'='*100}")

    # Filter to period
    period_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    if len(period_df) == 0:
        print(f"⚠️  No data for this period!")
        return None

    # Apply ML filter
    period_df['ml_prediction'] = (period_df['y_pred_proba'] >= threshold).astype(int)

    # Unfiltered metrics
    unfiltered_trades = len(period_df)
    unfiltered_wins = period_df['y_true'].sum()
    unfiltered_win_rate = unfiltered_wins / unfiltered_trades if unfiltered_trades > 0 else 0

    # Filtered metrics
    filtered_df = period_df[period_df['ml_prediction'] == 1].copy()
    filtered_trades = len(filtered_df)
    filtered_wins = filtered_df['y_true'].sum()
    filtered_win_rate = filtered_wins / filtered_trades if filtered_trades > 0 else 0
    filter_rate = (unfiltered_trades - filtered_trades) / unfiltered_trades if unfiltered_trades > 0 else 0

    print(f"\n📊 TRADE VOLUME:")
    print(f"  Unfiltered trades: {unfiltered_trades:,}")
    print(f"  Filtered trades:   {filtered_trades:,}")
    print(f"  Filter rate:       {filter_rate:.1%} (skipped {unfiltered_trades - filtered_trades:,} trades)")

    print(f"\n🎯 WIN RATE:")
    print(f"  Unfiltered: {unfiltered_win_rate:.2%}")
    print(f"  Filtered:   {filtered_win_rate:.2%}")
    print(f"  Improvement: {(filtered_win_rate - unfiltered_win_rate):.2%}")

    # P&L Analysis
    if 'Trade_PnL_Points' in period_df.columns:
        unfiltered_pnl = period_df['Trade_PnL_Points'].dropna()
        filtered_pnl = filtered_df['Trade_PnL_Points'].dropna()

        print(f"\n💰 P&L DISTRIBUTION (Points):")
        print(f"  Unfiltered:")
        print(f"    Mean:   {unfiltered_pnl.mean():.3f}")
        print(f"    Median: {unfiltered_pnl.median():.3f}")
        print(f"    Std:    {unfiltered_pnl.std():.3f}")
        print(f"  Filtered:")
        print(f"    Mean:   {filtered_pnl.mean():.3f}")
        print(f"    Median: {filtered_pnl.median():.3f}")
        print(f"    Std:    {filtered_pnl.std():.3f}")

        # Check for outliers
        if len(filtered_pnl) > 0:
            top_10_winners = filtered_pnl.nlargest(10)
            top_10_losers = filtered_pnl.nsmallest(10)

            top_10_pnl = top_10_winners.sum()
            total_pnl = filtered_pnl.sum()
            top_10_contribution = top_10_pnl / total_pnl if total_pnl != 0 else 0

            print(f"\n🎰 CONCENTRATION RISK:")
            print(f"  Top 10 winners contribute: {top_10_contribution:.1%} of total P&L")
            print(f"  Top 10 winners: {top_10_winners.mean():.2f} pts avg")
            print(f"  Top 10 losers:  {top_10_losers.mean():.2f} pts avg")

            if top_10_contribution > 0.5:
                print(f"  ⚠️  WARNING: Results highly concentrated in few trades!")

    # Model prediction distribution
    print(f"\n🤖 MODEL BEHAVIOR:")
    print(f"  Prediction probabilities:")
    print(f"    Mean:   {period_df['y_pred_proba'].mean():.3f}")
    print(f"    Median: {period_df['y_pred_proba'].median():.3f}")
    print(f"    Std:    {period_df['y_pred_proba'].std():.3f}")

    # How many trades at different confidence levels?
    high_conf = (period_df['y_pred_proba'] >= 0.7).sum()
    med_conf = ((period_df['y_pred_proba'] >= 0.5) & (period_df['y_pred_proba'] < 0.7)).sum()
    low_conf = (period_df['y_pred_proba'] < 0.5).sum()

    print(f"  Confidence distribution:")
    print(f"    High (≥0.70): {high_conf:,} ({high_conf/unfiltered_trades:.1%})")
    print(f"    Med (0.50-0.70): {med_conf:,} ({med_conf/unfiltered_trades:.1%})")
    print(f"    Low (<0.50): {low_conf:,} ({low_conf/unfiltered_trades:.1%})")

    # Calibration: does high confidence = high win rate?
    print(f"\n📈 MODEL CALIBRATION:")
    for threshold_check in [0.6, 0.7, 0.8]:
        high_conf_df = period_df[period_df['y_pred_proba'] >= threshold_check]
        if len(high_conf_df) > 0:
            actual_wr = high_conf_df['y_true'].mean()
            print(f"  Pred ≥{threshold_check:.1f}: {len(high_conf_df):4,} trades, {actual_wr:.1%} actual win rate")

    return {
        'period_name': period_name,
        'unfiltered_trades': unfiltered_trades,
        'filtered_trades': filtered_trades,
        'filter_rate': filter_rate,
        'unfiltered_win_rate': unfiltered_win_rate,
        'filtered_win_rate': filtered_win_rate,
        'mean_pred_proba': period_df['y_pred_proba'].mean(),
        'high_conf_pct': high_conf / unfiltered_trades if unfiltered_trades > 0 else 0
    }

def compare_periods(results: list):
    """Compare all periods side by side."""
    if not results or all(r is None for r in results):
        return

    results = [r for r in results if r is not None]

    print(f"\n{'='*100}")
    print("PERIOD COMPARISON")
    print(f"{'='*100}")

    df = pd.DataFrame(results)

    print(f"\n{'Period':<20} {'Trades':<10} {'Filter':<10} {'Unfilt WR':<12} {'Filt WR':<12} {'Avg Conf':<10}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{row['period_name']:<20} "
              f"{row['unfiltered_trades']:<10,} "
              f"{row['filter_rate']:<10.1%} "
              f"{row['unfiltered_win_rate']:<12.1%} "
              f"{row['filtered_win_rate']:<12.1%} "
              f"{row['mean_pred_proba']:<10.3f}")

    # Look for patterns
    print(f"\n🔍 KEY OBSERVATIONS:")

    # Check if held-out has different characteristics
    held_out = df[df['period_name'].str.contains('Held-Out')]
    walk_forward = df[~df['period_name'].str.contains('Held-Out')]

    if len(held_out) > 0 and len(walk_forward) > 0:
        print(f"\n  Walk-Forward avg filtered win rate: {walk_forward['filtered_win_rate'].mean():.1%}")
        print(f"  Held-Out filtered win rate:         {held_out['filtered_win_rate'].iloc[0]:.1%}")
        print(f"  Difference:                          {(held_out['filtered_win_rate'].iloc[0] - walk_forward['filtered_win_rate'].mean()):.1%}")

        if held_out['filtered_win_rate'].iloc[0] - walk_forward['filtered_win_rate'].mean() > 0.10:
            print(f"\n  ⚠️  REGIME CHANGE: Held-out period has significantly different characteristics!")

def main():
    parser = argparse.ArgumentParser(description="Diagnose regime differences")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"REGIME DIFFERENCE DIAGNOSTIC")
    print(f"Symbol: {args.symbol}")
    print(f"{'='*100}")

    # Load data
    print("\nLoading data...")
    df = load_data(args.symbol)
    print(f"Loaded {len(df):,} predictions")

    # Analyze each period
    results = []

    # Walk-forward windows
    results.append(analyze_period(df, "Window 1 (2016)", "2016-01-01", "2016-12-31"))
    results.append(analyze_period(df, "Window 2 (2017)", "2017-01-01", "2017-12-31"))
    results.append(analyze_period(df, "Window 3 (2018)", "2018-01-01", "2018-12-31"))
    results.append(analyze_period(df, "Window 4 (2019) ✓", "2019-01-01", "2019-12-31"))  # Only winner
    results.append(analyze_period(df, "Window 5 (2020)", "2020-01-01", "2020-12-31"))

    # Held-out period
    results.append(analyze_period(df, "Held-Out (2021-2024) 🔥", "2021-01-01", "2024-12-31"))

    # Compare
    compare_periods(results)

    print(f"\n{'='*100}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
