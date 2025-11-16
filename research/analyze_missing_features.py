#!/usr/bin/env python3
"""
Analyze which ML features are missing from the backtest CSV.
Compares the 30 features required by the ES model against what's in the CSV.
"""

import pandas as pd
import sys
from pathlib import Path

# Model's 30 required features
MODEL_FEATURES = [
    "volume_z_percentile", "volz_pct", "rsixvolz", "ibsxvolz",
    "tlt_daily_z_score", "6b_daily_return", "ma_spread_ribbon_tightness",
    "ma_spread_daily_ribbon_tightness", "distance_z_entry_daily",
    "6c_daily_return", "ibs_pct", "6a_hourly_return", "rsi_len_14",
    "atr_z_percentile", "6s_hourly_return", "price_usd", "atrz_pct",
    "nq_daily_z_score", "ma_slope_fast", "6a_daily_return",
    "daily_directional_drift", "hourly_atr_percentile", "prev_day_pctxvalue",
    "vix_hourly_z_score", "cl_hourly_z_score", "tlt_hourly_z_score",
    "adx_value", "daily_rsi_len_14", "nq_daily_return", "parabolic_sar_distance"
]

def main():
    # Find CSV file
    csv_path = Path('results/ES_optimization/ES_rf_best_trades.csv')

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        print("\nSearching for CSV files...")
        results_dir = Path('results')
        if results_dir.exists():
            csv_files = list(results_dir.rglob('*trades.csv'))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files:")
                for f in csv_files:
                    print(f"  - {f}")
                csv_path = csv_files[0]
                print(f"\nUsing: {csv_path}")
            else:
                print("No trade CSV files found in results/")
                sys.exit(1)
        else:
            print("No results/ directory found")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"ANALYZING: {csv_path}")
    print(f"{'='*80}\n")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} trades with {len(df.columns)} columns\n")

    # Check each required feature
    missing_features = []
    zero_features = []
    nan_features = []
    ok_features = []

    for feat in MODEL_FEATURES:
        if feat not in df.columns:
            missing_features.append(feat)
            print(f"❌ {feat:35s} - NOT IN CSV")
        elif df[feat].isna().all():
            nan_features.append(feat)
            print(f"⚠️  {feat:35s} - ALL NaN (100%)")
        elif (df[feat] == 0.0).all():
            zero_features.append(feat)
            print(f"⚠️  {feat:35s} - ALL zeros (mean=0.000)")
        else:
            ok_features.append(feat)
            mean_val = df[feat].mean()
            nan_pct = (df[feat].isna().sum() / len(df)) * 100
            print(f"✅ {feat:35s} - OK (mean={mean_val:7.3f}, NaN={nan_pct:.1f}%)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total required features: {len(MODEL_FEATURES)}")
    print(f"✅ OK features:          {len(ok_features)} ({len(ok_features)/len(MODEL_FEATURES)*100:.1f}%)")
    print(f"⚠️  ALL zeros:            {len(zero_features)} ({len(zero_features)/len(MODEL_FEATURES)*100:.1f}%)")
    print(f"⚠️  ALL NaN:              {len(nan_features)} ({len(nan_features)/len(MODEL_FEATURES)*100:.1f}%)")
    print(f"❌ Missing from CSV:     {len(missing_features)} ({len(missing_features)/len(MODEL_FEATURES)*100:.1f}%)")

    if zero_features:
        print(f"\nFeatures with ALL ZEROS (likely missing data sources):")
        for feat in zero_features:
            print(f"  - {feat}")

    if nan_features:
        print(f"\nFeatures with ALL NaN:")
        for feat in nan_features:
            print(f"  - {feat}")

    if missing_features:
        print(f"\nFeatures NOT in CSV:")
        for feat in missing_features:
            print(f"  - {feat}")

    print(f"\n{'='*80}")

    # Identify affected symbols
    problematic = zero_features + nan_features + missing_features
    if problematic:
        print(f"\n🔍 DIAGNOSIS:")
        print(f"The model requires {len(problematic)} features that have no valid values.")
        print(f"These features will be treated as 0.0, causing ML predictions to be WRONG.\n")

        # Group by likely data source
        vix_features = [f for f in problematic if 'vix' in f.lower()]
        if vix_features:
            print(f"VIX-related features ({len(vix_features)}):")
            for f in vix_features:
                print(f"  - {f}")

        # Check for other symbols
        symbols = ['6a', '6b', '6c', '6e', '6j', '6m', '6n', '6s',
                   'tlt', 'cl', 'ng', 'gc', 'si', 'nq', 'ym', 'rty']
        for sym in symbols:
            sym_features = [f for f in problematic if sym in f.lower() and f not in vix_features]
            if sym_features:
                print(f"\n{sym.upper()}-related features ({len(sym_features)}):")
                for f in sym_features:
                    print(f"  - {f}")

if __name__ == '__main__':
    main()
