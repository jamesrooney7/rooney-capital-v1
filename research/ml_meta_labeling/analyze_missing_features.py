#!/usr/bin/env python3
"""
Compare requested features vs actual CSV columns to find all missing features.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Import the feature key generator
# Need to import from the file directly since it's a script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_training_data",
    project_root / "research" / "extract_training_data.py"
)
extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_module)

FeatureLoggingStrategy = extract_module.FeatureLoggingStrategy

# Get all requested features
requested_features = FeatureLoggingStrategy._get_all_filter_param_keys()
print(f"Total requested features: {len(requested_features)}")
print()

# Load CSV to see what was actually calculated
csv_path = Path('data/training/ES_transformed_features.csv')
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found")
    sys.exit(1)

df = pd.read_csv(csv_path, nrows=1)
actual_columns = set(df.columns)

print(f"Total CSV columns: {len(actual_columns)}")
print()

# Core columns (not features)
core_columns = {
    'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',
    'Entry_Price', 'Exit_Price', 'Symbol', 'Trade_ID',
    'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross'
}

# Find missing features
# Note: Some requested features may have different names in output
# (e.g., enableXXX is requested to trigger creation, but actual value is XXX)
missing_features = []
for feature in requested_features:
    # Skip enable* features - they're just triggers
    if feature.lower().startswith('enable'):
        # Check if the corresponding value column exists
        # enableXXX should create a column like XXX or xxx_value
        base_name = feature[6:]  # Remove 'enable' prefix

        # Check various possible output names
        possible_names = [
            base_name,
            base_name.lower(),
            f"{base_name}_value",
            f"{base_name.lower()}_value",
        ]

        if not any(name in actual_columns for name in possible_names):
            missing_features.append(feature)
    else:
        # Direct feature check
        if feature not in actual_columns:
            missing_features.append(feature)

# Remove features that were intentionally filtered (VIX, title case, etc.)
filtered_missing = []
for feature in missing_features:
    # Skip VIX features (we filter these)
    if 'vix' in feature.lower():
        continue
    filtered_missing.append(feature)

print("=" * 100)
print(f"MISSING FEATURES ({len(filtered_missing)} features not calculated)")
print("=" * 100)
print()

# Group by category
z_scores = [f for f in filtered_missing if 'z_score' in f.lower()]
returns = [f for f in filtered_missing if 'return' in f.lower()]
enable_params = [f for f in filtered_missing if f.lower().startswith('enable')]
other = [f for f in filtered_missing if f not in z_scores and f not in returns and f not in enable_params]

if z_scores:
    print(f"Z-Score Features Missing ({len(z_scores)}):")
    for f in sorted(z_scores):
        print(f"  {f}")
    print()

if returns:
    print(f"Return Features Missing ({len(returns)}):")
    for f in sorted(returns):
        print(f"  {f}")
    print()

if enable_params:
    print(f"Enable Parameters Missing ({len(enable_params)}):")
    for f in sorted(enable_params):
        print(f"  {f}")
    print()

if other:
    print(f"Other Features Missing ({len(other)}):")
    for f in sorted(other):
        print(f"  {f}")
    print()

print("=" * 100)
print("ANALYSIS")
print("=" * 100)

# Check which symbols' z-scores are missing
missing_symbols = set()
for f in z_scores:
    # Extract symbol from patterns like 'cl_z_score_day' or 'enableCLZScoreHour'
    if '_z_score_' in f:
        symbol = f.split('_z_score_')[0].upper()
        missing_symbols.add(symbol)
    elif 'zscore' in f.lower():
        # Handle enableXXZScoreYY pattern
        import re
        match = re.search(r'enable([A-Z0-9]+)ZScore', f)
        if match:
            missing_symbols.add(match.group(1))

if missing_symbols:
    print(f"\nSymbols with missing z-scores: {', '.join(sorted(missing_symbols))}")
    print()

# Show what IS in the CSV
print("=" * 100)
print("WHAT'S IN THE CSV (sample of calculated features)")
print("=" * 100)
feature_cols = [c for c in actual_columns if c not in core_columns]
z_score_cols = [c for c in feature_cols if 'z_score' in c.lower()]
return_cols = [c for c in feature_cols if 'return' in c.lower()]

print(f"\nZ-scores calculated ({len(z_score_cols)}):")
for c in sorted(z_score_cols)[:20]:
    print(f"  {c}")
if len(z_score_cols) > 20:
    print(f"  ... and {len(z_score_cols)-20} more")

print(f"\nReturns calculated ({len(return_cols)}):")
for c in sorted(return_cols)[:20]:
    print(f"  {c}")
if len(return_cols) > 20:
    print(f"  ... and {len(return_cols)-20} more")
