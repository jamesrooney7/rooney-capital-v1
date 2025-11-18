#!/usr/bin/env python3
"""
Verify missing value calculation is correct by showing actual data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

# Load ES data
csv_path = Path('data/training/ES_transformed_features.csv')
print(f"Loading: {csv_path}")
print()

df = pd.read_csv(csv_path)

print("=" * 100)
print("BASIC INFO")
print("=" * 100)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")
print()

# Get date range
if 'Date/Time' in df.columns:
    df['Date'] = pd.to_datetime(df['Date/Time'])
elif 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print()

# Simulate the filtering that happens in data_preparation.py
from research.ml_meta_labeling.components.config_defaults import (
    should_remove_column, METADATA_COLUMNS, TARGET_COLUMNS
)

# Filter columns like the actual system does
print("=" * 100)
print("STEP 1: FEATURE FILTERING (title case, enable*, VIX)")
print("=" * 100)

kept_columns = []
removed_columns = []

for col in df.columns:
    if should_remove_column(col, True, True, True):
        removed_columns.append(col)
    else:
        kept_columns.append(col)

filtered_df = df[kept_columns].copy()

print(f"Removed {len(removed_columns)} columns")
print(f"Kept {len(kept_columns)} columns")
print()

# Identify feature columns (exclude metadata and targets)
feature_columns = [
    col for col in kept_columns
    if col not in METADATA_COLUMNS and col not in TARGET_COLUMNS
    and pd.api.types.is_numeric_dtype(filtered_df[col])
]

print(f"Feature columns (numeric, non-metadata): {len(feature_columns)}")
print()

# Now check missing values on feature columns only
print("=" * 100)
print("STEP 2: MISSING VALUE ANALYSIS (on feature columns only)")
print("=" * 100)

missing_analysis = []
for col in feature_columns:
    missing_count = filtered_df[col].isna().sum()
    missing_fraction = missing_count / len(filtered_df)
    missing_pct = missing_fraction * 100

    missing_analysis.append({
        'column': col,
        'missing_count': missing_count,
        'missing_fraction': missing_fraction,
        'missing_pct': missing_pct
    })

# Sort by missing percentage
missing_analysis.sort(key=lambda x: x['missing_pct'], reverse=True)

# Apply 25% threshold
threshold = 0.25
high_missing = [m for m in missing_analysis if m['missing_fraction'] > threshold]
kept_features = [m for m in missing_analysis if m['missing_fraction'] <= threshold]

print(f"Threshold: {threshold*100}%")
print(f"Features with >{threshold*100}% missing (REMOVED): {len(high_missing)}")
print(f"Features with <={threshold*100}% missing (KEPT): {len(kept_features)}")
print()

# Show top 30 removed features
print("=" * 100)
print(f"TOP 30 FEATURES REMOVED (>{threshold*100}% missing):")
print("=" * 100)
print(f"{'#':<4} {'Column':<60} {'Missing %':<12} {'Missing Count':<15} {'Non-Missing'}")
print("-" * 100)

for i, m in enumerate(high_missing[:30], 1):
    non_missing = len(filtered_df) - m['missing_count']
    print(f"{i:<4} {m['column']:<60} {m['missing_pct']:>10.2f}% {m['missing_count']:>14,} {non_missing:>12,}")

print()
if len(high_missing) > 30:
    print(f"... and {len(high_missing) - 30} more columns removed")
    print()

# Show some examples of kept features
print("=" * 100)
print(f"SAMPLE OF KEPT FEATURES (<={threshold*100}% missing):")
print("=" * 100)
print(f"{'#':<4} {'Column':<60} {'Missing %':<12} {'Missing Count':<15} {'Non-Missing'}")
print("-" * 100)

for i, m in enumerate(kept_features[:20], 1):
    non_missing = len(filtered_df) - m['missing_count']
    print(f"{i:<4} {m['column']:<60} {m['missing_pct']:>10.2f}% {m['missing_count']:>14,} {non_missing:>12,}")

print()

# Verify the calculation with a specific example
if high_missing:
    print("=" * 100)
    print("VERIFICATION: Let's check one removed column manually")
    print("=" * 100)

    example = high_missing[0]
    col_name = example['column']

    print(f"Column: {col_name}")
    print(f"Total rows in filtered_df: {len(filtered_df):,}")
    print()

    # Manual count
    col_data = filtered_df[col_name]
    manual_missing = col_data.isna().sum()
    manual_non_missing = (~col_data.isna()).sum()
    manual_pct = manual_missing / len(filtered_df) * 100

    print(f"Manual calculation:")
    print(f"  Missing values: {manual_missing:,}")
    print(f"  Non-missing values: {manual_non_missing:,}")
    print(f"  Missing percentage: {manual_pct:.2f}%")
    print()

    print(f"Our calculation:")
    print(f"  Missing values: {example['missing_count']:,}")
    print(f"  Missing percentage: {example['missing_pct']:.2f}%")
    print()

    print(f"Match: {manual_missing == example['missing_count']}")
    print()

    # Show first few values
    print(f"First 20 values (showing NaN patterns):")
    for i, val in enumerate(col_data.head(20), 1):
        if pd.isna(val):
            print(f"  Row {i}: NaN")
        else:
            print(f"  Row {i}: {val}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total columns in CSV: {len(df.columns):,}")
print(f"After filtering (title case, enable*, VIX): {len(kept_columns):,}")
print(f"Numeric feature columns: {len(feature_columns):,}")
print(f"Features kept (≤{threshold*100}% missing): {len(kept_features):,}")
print(f"Features removed (>{threshold*100}% missing): {len(high_missing):,}")
print()
