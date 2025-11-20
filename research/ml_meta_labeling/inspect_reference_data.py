#!/usr/bin/env python3
"""
Inspect resampled reference data quality for cross-asset features.

This script checks:
1. Which reference symbols have resampled data
2. Date range coverage for each symbol
3. Missing value percentages in OHLCV data
4. Data quality issues (gaps, duplicates, etc.)
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Reference symbols used in extract_training_data.py
REFERENCE_SYMBOLS = [
    'ES', 'NQ', 'RTY', 'YM',           # US Equities Futures
    'GC', 'SI', 'HG',                  # Metals
    'CL', 'NG', 'PL',                  # Energy/Precious
    '6A', '6B', '6C', '6E', '6J',      # Currencies
    '6M', '6N', '6S',                  # More Currencies
    'TLT', 'VIX'                       # Reference/Regime
]

def inspect_resampled_data(data_dir='data/resampled', target_start='2010-01-01', target_end='2024-12-31'):
    """
    Inspect quality of resampled reference data.

    Args:
        data_dir: Directory containing resampled hourly/daily CSVs
        target_start: Desired start date for training data
        target_end: Desired end date for training data
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_path}")
        print(f"Expected resampled data in: {data_path.absolute()}")
        return

    target_start_dt = pd.to_datetime(target_start)
    target_end_dt = pd.to_datetime(target_end)

    print("=" * 120)
    print("RESAMPLED REFERENCE DATA INSPECTION")
    print("=" * 120)
    print(f"Data Directory: {data_path.absolute()}")
    print(f"Target Date Range: {target_start} to {target_end}")
    print()

    # Track results
    results = []

    for symbol in REFERENCE_SYMBOLS:
        hourly_file = data_path / f"{symbol}_hourly.csv"
        daily_file = data_path / f"{symbol}_daily.csv"

        symbol_result = {
            'symbol': symbol,
            'hourly_exists': hourly_file.exists(),
            'daily_exists': daily_file.exists(),
            'hourly_rows': 0,
            'daily_rows': 0,
            'hourly_start': None,
            'hourly_end': None,
            'daily_start': None,
            'daily_end': None,
            'hourly_coverage': 0.0,
            'daily_coverage': 0.0,
            'hourly_missing_pct': 0.0,
            'daily_missing_pct': 0.0,
            'status': 'MISSING',
        }

        # Inspect hourly data
        if hourly_file.exists():
            try:
                df = pd.read_csv(hourly_file)

                # Parse datetime column
                date_col = None
                for col in ['datetime', 'Datetime', 'Date', 'date', 'timestamp']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    symbol_result['hourly_rows'] = len(df)
                    symbol_result['hourly_start'] = df[date_col].min()
                    symbol_result['hourly_end'] = df[date_col].max()

                    # Calculate coverage
                    if symbol_result['hourly_start'] <= target_start_dt and symbol_result['hourly_end'] >= target_end_dt:
                        symbol_result['hourly_coverage'] = 100.0
                    elif symbol_result['hourly_start'] > target_end_dt or symbol_result['hourly_end'] < target_start_dt:
                        symbol_result['hourly_coverage'] = 0.0
                    else:
                        # Partial overlap
                        overlap_start = max(symbol_result['hourly_start'], target_start_dt)
                        overlap_end = min(symbol_result['hourly_end'], target_end_dt)
                        overlap_days = (overlap_end - overlap_start).days
                        target_days = (target_end_dt - target_start_dt).days
                        symbol_result['hourly_coverage'] = (overlap_days / target_days) * 100

                    # Check for missing OHLCV values
                    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
                    if ohlcv_cols:
                        missing = df[ohlcv_cols].isna().sum().sum()
                        total = len(df) * len(ohlcv_cols)
                        symbol_result['hourly_missing_pct'] = (missing / total) * 100

            except Exception as e:
                print(f"  ⚠️  Error reading {hourly_file.name}: {e}")

        # Inspect daily data
        if daily_file.exists():
            try:
                df = pd.read_csv(daily_file)

                # Parse datetime column
                date_col = None
                for col in ['datetime', 'Datetime', 'Date', 'date', 'timestamp']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    symbol_result['daily_rows'] = len(df)
                    symbol_result['daily_start'] = df[date_col].min()
                    symbol_result['daily_end'] = df[date_col].max()

                    # Calculate coverage
                    if symbol_result['daily_start'] <= target_start_dt and symbol_result['daily_end'] >= target_end_dt:
                        symbol_result['daily_coverage'] = 100.0
                    elif symbol_result['daily_start'] > target_end_dt or symbol_result['daily_end'] < target_start_dt:
                        symbol_result['daily_coverage'] = 0.0
                    else:
                        # Partial overlap
                        overlap_start = max(symbol_result['daily_start'], target_start_dt)
                        overlap_end = min(symbol_result['daily_end'], target_end_dt)
                        overlap_days = (overlap_end - overlap_start).days
                        target_days = (target_end_dt - target_start_dt).days
                        symbol_result['daily_coverage'] = (overlap_days / target_days) * 100

                    # Check for missing OHLCV values
                    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
                    if ohlcv_cols:
                        missing = df[ohlcv_cols].isna().sum().sum()
                        total = len(df) * len(ohlcv_cols)
                        symbol_result['daily_missing_pct'] = (missing / total) * 100

            except Exception as e:
                print(f"  ⚠️  Error reading {daily_file.name}: {e}")

        # Determine status
        if symbol_result['hourly_exists'] and symbol_result['daily_exists']:
            # Check if data meets minimum requirements (from extract_training_data.py line 434)
            if symbol_result['hourly_rows'] >= 252 * 6 and symbol_result['daily_rows'] >= 252:
                if symbol_result['hourly_coverage'] >= 90 and symbol_result['daily_coverage'] >= 90:
                    symbol_result['status'] = 'GOOD'
                else:
                    symbol_result['status'] = 'PARTIAL'
            else:
                symbol_result['status'] = 'INSUFFICIENT'
        elif symbol_result['hourly_exists'] or symbol_result['daily_exists']:
            symbol_result['status'] = 'INCOMPLETE'
        else:
            symbol_result['status'] = 'MISSING'

        results.append(symbol_result)

    # Print results
    print("=" * 120)
    print("SYMBOL DATA QUALITY SUMMARY")
    print("=" * 120)
    print(f"{'Symbol':<8} {'Status':<12} {'Hourly Rows':>12} {'Daily Rows':>11} {'H Coverage':>11} {'D Coverage':>11} {'Date Range'}")
    print("-" * 120)

    for r in results:
        hourly_range = f"{r['hourly_start'].date() if r['hourly_start'] else 'N/A'} to {r['hourly_end'].date() if r['hourly_end'] else 'N/A'}"
        daily_range = f"{r['daily_start'].date() if r['daily_start'] else 'N/A'} to {r['daily_end'].date() if r['daily_end'] else 'N/A'}"
        date_range = hourly_range if r['hourly_exists'] else daily_range

        status_icon = {
            'GOOD': '✓',
            'PARTIAL': '⚠',
            'INSUFFICIENT': '⚠',
            'INCOMPLETE': '✗',
            'MISSING': '✗',
        }.get(r['status'], '?')

        print(
            f"{r['symbol']:<8} "
            f"{status_icon} {r['status']:<10} "
            f"{r['hourly_rows']:>12,} "
            f"{r['daily_rows']:>11,} "
            f"{r['hourly_coverage']:>10.1f}% "
            f"{r['daily_coverage']:>10.1f}% "
            f"{date_range}"
        )

    print()
    print("=" * 120)
    print("STATUS BREAKDOWN")
    print("=" * 120)

    status_counts = {}
    for r in results:
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1

    for status, count in sorted(status_counts.items()):
        symbols = [r['symbol'] for r in results if r['status'] == status]
        print(f"{status:<15} {count:>3} symbols: {', '.join(symbols)}")

    print()

    # Show detailed info for problematic symbols
    problematic = [r for r in results if r['status'] not in ['GOOD']]

    if problematic:
        print("=" * 120)
        print("PROBLEMATIC SYMBOLS (need attention)")
        print("=" * 120)
        print()

        for r in problematic:
            print(f"{r['symbol']} - {r['status']}")
            print(f"  Hourly: {'EXISTS' if r['hourly_exists'] else 'MISSING'} "
                  f"({r['hourly_rows']:,} rows, {r['hourly_coverage']:.1f}% coverage, "
                  f"{r['hourly_missing_pct']:.2f}% missing OHLCV)")
            print(f"  Daily:  {'EXISTS' if r['daily_exists'] else 'MISSING'} "
                  f"({r['daily_rows']:,} rows, {r['daily_coverage']:.1f}% coverage, "
                  f"{r['daily_missing_pct']:.2f}% missing OHLCV)")
            if r['hourly_start']:
                print(f"  Hourly Date Range: {r['hourly_start'].date()} to {r['hourly_end'].date()}")
            if r['daily_start']:
                print(f"  Daily Date Range:  {r['daily_start'].date()} to {r['daily_end'].date()}")
            print()

    # Recommendations
    print("=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    good_symbols = [r['symbol'] for r in results if r['status'] == 'GOOD']
    bad_symbols = [r['symbol'] for r in results if r['status'] not in ['GOOD']]

    print()
    print(f"✓ {len(good_symbols)} symbols have good data: {', '.join(good_symbols)}")
    print()

    if bad_symbols:
        print(f"✗ {len(bad_symbols)} symbols have issues: {', '.join(bad_symbols)}")
        print()
        print("Options:")
        print()
        print("1. RESAMPLE MISSING SYMBOLS:")
        print("   Run: python research/utils/resample_data.py --symbols", ' '.join(bad_symbols))
        print()
        print("2. EXCLUDE PROBLEMATIC SYMBOLS FROM EXTRACTION:")
        print("   Modify extract_training_data.py line 411-413 to only use:")
        print(f"   reference_symbols = {good_symbols}")
        print()
        print("3. REGENERATE ES_transformed_features.csv with only good symbols:")
        print("   python research/extract_training_data.py --symbol ES --start 2011-01-01 --end 2024-12-31")
        print("   (After modifying reference_symbols list)")
        print()
    else:
        print("✓ All reference symbols have good data!")
        print()
        print("You can regenerate ES_transformed_features.csv with full cross-asset features:")
        print("  python research/extract_training_data.py --symbol ES --start 2011-01-01 --end 2024-12-31")
        print()

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inspect resampled reference data quality')
    parser.add_argument('--data-dir', default='data/resampled', help='Resampled data directory')
    parser.add_argument('--start', default='2010-01-01', help='Target start date')
    parser.add_argument('--end', default='2024-12-31', help='Target end date')

    args = parser.parse_args()

    inspect_resampled_data(args.data_dir, args.start, args.end)
