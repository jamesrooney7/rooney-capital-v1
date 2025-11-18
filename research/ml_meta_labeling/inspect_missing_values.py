#!/usr/bin/env python3
"""
Inspect missing value percentages in transformed features CSV.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def inspect_missing_values(csv_path, threshold=25.0):
    """
    Analyze missing value percentages in CSV file.

    Args:
        csv_path: Path to CSV file
        threshold: Missing value threshold percentage (default: 25%)
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print()

    # Calculate missing value percentage for each column
    missing_stats = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        if missing_pct > 0:
            missing_stats.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct
            })

    # Sort by missing percentage
    missing_stats.sort(key=lambda x: x['missing_pct'], reverse=True)

    print("=" * 100)
    print("COLUMNS WITH MISSING VALUES (sorted by % missing)")
    print("=" * 100)
    print()

    # Show all columns with >threshold% missing
    print(f"Columns with >{threshold}% missing values (WILL BE REMOVED):")
    high_missing = [s for s in missing_stats if s['missing_pct'] > threshold]
    print(f"Count: {len(high_missing)}")
    print()

    for i, stat in enumerate(high_missing[:30], 1):  # Show first 30
        print(f"  {i:3d}. {stat['column']:55s} {stat['missing_pct']:6.2f}% ({stat['missing_count']:,} rows)")

    if len(high_missing) > 30:
        print(f"  ... and {len(high_missing) - 30} more")

    print()
    print("=" * 100)
    print(f"Columns with {threshold/2:.1f}-{threshold}% missing values (WILL BE KEPT):")
    medium_missing = [s for s in missing_stats if threshold/2 < s['missing_pct'] <= threshold]
    print(f"Count: {len(medium_missing)}")
    print()

    for i, stat in enumerate(medium_missing[:20], 1):  # Show first 20
        print(f"  {i:3d}. {stat['column']:55s} {stat['missing_pct']:6.2f}% ({stat['missing_count']:,} rows)")

    print()
    print("=" * 100)
    print(f"Columns with 1-{threshold/2:.1f}% missing values (WILL BE KEPT):")
    low_missing = [s for s in missing_stats if 1 < s['missing_pct'] <= threshold/2]
    print(f"Count: {len(low_missing)}")
    print()

    for i, stat in enumerate(low_missing[:20], 1):  # Show first 20
        print(f"  {i:3d}. {stat['column']:55s} {stat['missing_pct']:6.2f}% ({stat['missing_count']:,} rows)")

    print()
    print("=" * 100)
    print("SUMMARY:")
    print("=" * 100)
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns with NO missing values: {len(df.columns) - len(missing_stats)}")
    print(f"Columns with <1% missing: {len([s for s in missing_stats if s['missing_pct'] <= 1])}")
    print(f"Columns with 1-{threshold/2:.1f}% missing: {len(low_missing)}")
    print(f"Columns with {threshold/2:.1f}-{threshold}% missing: {len(medium_missing)}")
    print(f"Columns with >{threshold}% missing (REMOVED): {len(high_missing)}")
    print()

    # Check date range
    if 'Date/Time' in df.columns:
        date_col = 'Date/Time'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        date_col = None

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        print()

    # Show a sample of high-missing columns to understand pattern
    if high_missing:
        print("=" * 100)
        print("PATTERN ANALYSIS OF REMOVED COLUMNS:")
        print("=" * 100)

        # Count by pattern
        patterns = {}
        for stat in high_missing:
            col = stat['column']
            # Extract base pattern
            if '_' in col:
                base = col.split('_')[0]
            else:
                base = col[:10] if len(col) > 10 else col

            if base not in patterns:
                patterns[base] = []
            patterns[base].append(col)

        print(f"Number of unique patterns: {len(patterns)}")
        print()
        print("Top 10 patterns (by column count):")
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (pattern, cols) in enumerate(sorted_patterns[:10], 1):
            print(f"  {i:2d}. {pattern:20s}: {len(cols):3d} columns")
            # Show first 3 examples
            for col in cols[:3]:
                pct = next(s['missing_pct'] for s in high_missing if s['column'] == col)
                print(f"      - {col:50s} ({pct:.1f}% missing)")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect missing values in transformed features")
    parser.add_argument("--symbol", default="ES", help="Trading symbol")
    parser.add_argument("--data-dir", default="data/training", help="Data directory")
    parser.add_argument("--threshold", type=float, default=25.0, help="Missing value threshold %")

    args = parser.parse_args()

    csv_path = Path(args.data_dir) / f"{args.symbol}_transformed_features.csv"

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        exit(1)

    inspect_missing_values(csv_path, args.threshold)
