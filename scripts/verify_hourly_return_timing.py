#!/usr/bin/env python3
"""
Verify that hourly returns are NOT showing lookahead bias.

If hourly returns had lookahead bias (using Bar N instead of Bar N-1),
we would see:
1. Strong correlation between ES hourly return and trade outcome
2. Direction agreement >70% (feature predicts outcome direction)
3. Large difference in mean hourly return between winners and losers

We previously tested this and found 35.9% direction agreement (below 50% random),
but let's verify again with the actual features used by the trained models.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def verify_timing(symbol):
    """Verify hourly return timing for a symbol."""
    data_path = Path('data/training') / f'{symbol}_transformed_features.csv'

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)

    # Get the feature column names for hourly returns
    hourly_return_cols = [col for col in df.columns if 'Hourly Return' in col]

    print(f"\n{'='*100}")
    print(f"{symbol} - HOURLY RETURN TIMING VERIFICATION")
    print(f"{'='*100}\n")

    print(f"Found {len(hourly_return_cols)} hourly return features:")
    for col in hourly_return_cols:
        print(f"  - {col}")

    print(f"\nTotal trades: {len(df):,}")

    # For each hourly return feature, check for lookahead bias
    for col in hourly_return_cols:
        if col not in df.columns:
            continue

        # Skip if all NaN
        if df[col].isna().all():
            continue

        # Get trade outcomes
        y_return = df['y_return'].values
        feature_values = df[col].values

        # Remove NaN values
        mask = ~np.isnan(feature_values) & ~np.isnan(y_return)
        feature_clean = feature_values[mask]
        y_clean = y_return[mask]

        if len(feature_clean) == 0:
            continue

        # TEST 1: Direction agreement
        # If lookahead: feature direction would match outcome direction >70%
        # If no lookahead: feature direction weakly correlates or anti-correlates
        feature_positive = feature_clean > 0
        outcome_positive = y_clean > 0
        direction_agreement = np.mean(feature_positive == outcome_positive)

        # TEST 2: Mean difference between winners and losers
        # If lookahead: winners would have different feature values than losers
        # If no lookahead: similar feature values for winners and losers
        winners = feature_clean[y_clean > 0]
        losers = feature_clean[y_clean < 0]

        mean_winners = np.mean(winners) if len(winners) > 0 else 0
        mean_losers = np.mean(losers) if len(losers) > 0 else 0
        mean_diff = abs(mean_winners - mean_losers)

        # TEST 3: Correlation
        # If lookahead: strong positive correlation
        # If no lookahead: weak or negative correlation
        correlation = np.corrcoef(feature_clean, y_clean)[0, 1]

        print(f"\n{'-'*100}")
        print(f"Feature: {col}")
        print(f"{'-'*100}")
        print(f"Valid samples: {len(feature_clean):,} / {len(df):,} ({len(feature_clean)/len(df)*100:.1f}%)")
        print(f"\nStatistics:")
        print(f"  Min:    {np.min(feature_clean):>10.4f}%")
        print(f"  Max:    {np.max(feature_clean):>10.4f}%")
        print(f"  Mean:   {np.mean(feature_clean):>10.4f}%")
        print(f"  Median: {np.median(feature_clean):>10.4f}%")
        print(f"  Std:    {np.std(feature_clean):>10.4f}%")

        print(f"\n🔍 LOOKAHEAD TESTS:")
        print(f"\n  1. Direction Agreement: {direction_agreement*100:.2f}%")
        if direction_agreement > 0.70:
            print(f"     ⚠️  SUSPICIOUS: >70% suggests lookahead bias")
        elif direction_agreement < 0.40:
            print(f"     ✅ GOOD: <40% indicates no lookahead (anti-correlation for mean reversion)")
        else:
            print(f"     ⚠️  BORDERLINE: 40-70% - unclear")

        print(f"\n  2. Mean Difference (Winners vs Losers): {mean_diff:.4f}%")
        print(f"     Winners mean:  {mean_winners:>10.4f}%")
        print(f"     Losers mean:   {mean_losers:>10.4f}%")
        if mean_diff > 0.5:
            print(f"     ⚠️  SUSPICIOUS: >0.5% difference suggests predictive power")
        else:
            print(f"     ✅ GOOD: <0.5% difference suggests no lookahead")

        print(f"\n  3. Correlation with outcome: {correlation:.4f}")
        if abs(correlation) > 0.3:
            print(f"     ⚠️  SUSPICIOUS: |correlation| >0.3 suggests strong relationship")
        else:
            print(f"     ✅ GOOD: Weak correlation suggests no perfect prediction")

    return True


def main():
    symbols = ['ES', 'NQ', 'RTY', 'YM']

    print("="*100)
    print("HOURLY RETURN TIMING VERIFICATION")
    print("="*100)
    print("\nThis script verifies that hourly return features do NOT have lookahead bias.")
    print("We check if the feature values can predict trade outcomes (which would indicate timing issues).")
    print()

    for symbol in symbols:
        result = verify_timing(symbol)
        if not result:
            print(f"\n❌ {symbol}: Data not found")

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("\nIf hourly returns had lookahead bias, we would see:")
    print("  - Direction agreement >70% (feature predicts outcome direction)")
    print("  - Large mean difference between winners/losers (>0.5%)")
    print("  - Strong correlation (>0.3)")
    print("\nIf NO lookahead (correct timing):")
    print("  - Direction agreement 30-50% (weak or anti-correlation for mean reversion)")
    print("  - Small mean difference (<0.2%)")
    print("  - Weak correlation (<0.2)")
    print()


if __name__ == '__main__':
    main()