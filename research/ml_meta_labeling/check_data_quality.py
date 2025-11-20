#!/usr/bin/env python3
"""
Check for data quality issues in resampled CSVs that could cause feature calculation problems.

Checks for:
1. Rows with zero/null OHLCV values
2. Rows where high < low (impossible)
3. Rows where close outside high/low range
4. Duplicate timestamps
5. Time gaps (missing bars)
6. Constant price bars (open=high=low=close)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

REFERENCE_SYMBOLS = [
    'ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
    '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S', 'TLT', 'VIX'
]

def check_symbol_quality(symbol, data_dir='data/resampled'):
    """Check data quality for a single symbol."""

    data_path = Path(data_dir)
    issues = {
        'symbol': symbol,
        'hourly_issues': [],
        'daily_issues': [],
    }

    for timeframe in ['hourly', 'daily']:
        file_path = data_path / f"{symbol}_{timeframe}.csv"

        if not file_path.exists():
            issues[f'{timeframe}_issues'].append(f"File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            # Find datetime column
            date_col = None
            for col in ['datetime', 'Datetime', 'Date', 'date', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break

            if not date_col:
                issues[f'{timeframe}_issues'].append("No datetime column found")
                continue

            df[date_col] = pd.to_datetime(df[date_col])

            # Find OHLCV columns (case-insensitive)
            ohlcv_map = {}
            for target in ['open', 'high', 'low', 'close', 'volume']:
                for col in df.columns:
                    if col.lower() == target:
                        ohlcv_map[target] = col
                        break

            if len(ohlcv_map) < 4:  # Need at least OHLC
                issues[f'{timeframe}_issues'].append(f"Missing OHLC columns. Found: {list(ohlcv_map.keys())}")
                continue

            o_col = ohlcv_map.get('open')
            h_col = ohlcv_map.get('high')
            l_col = ohlcv_map.get('low')
            c_col = ohlcv_map.get('close')
            v_col = ohlcv_map.get('volume')

            # Check 1: Rows with zero values in OHLC
            zero_mask = (df[o_col] == 0) | (df[h_col] == 0) | (df[l_col] == 0) | (df[c_col] == 0)
            zero_count = zero_mask.sum()
            if zero_count > 0:
                zero_pct = (zero_count / len(df)) * 100
                issues[f'{timeframe}_issues'].append(
                    f"{zero_count:,} rows ({zero_pct:.2f}%) with zero OHLC values"
                )
                # Show first few examples
                zero_dates = df[zero_mask][date_col].head(5).tolist()
                issues[f'{timeframe}_issues'].append(f"  Examples: {[d.date() for d in zero_dates]}")

            # Check 2: Rows with NaN values in OHLC
            nan_mask = df[o_col].isna() | df[h_col].isna() | df[l_col].isna() | df[c_col].isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                nan_pct = (nan_count / len(df)) * 100
                issues[f'{timeframe}_issues'].append(
                    f"{nan_count:,} rows ({nan_pct:.2f}%) with NaN OHLC values"
                )
                nan_dates = df[nan_mask][date_col].head(5).tolist()
                issues[f'{timeframe}_issues'].append(f"  Examples: {[d.date() for d in nan_dates]}")

            # Check 3: Rows where high < low (impossible)
            invalid_hl = df[h_col] < df[l_col]
            invalid_hl_count = invalid_hl.sum()
            if invalid_hl_count > 0:
                issues[f'{timeframe}_issues'].append(
                    f"{invalid_hl_count:,} rows where high < low (INVALID)"
                )
                invalid_dates = df[invalid_hl][date_col].head(5).tolist()
                issues[f'{timeframe}_issues'].append(f"  Examples: {[d.date() for d in invalid_dates]}")

            # Check 4: Rows where close outside [low, high]
            close_outside = (df[c_col] < df[l_col]) | (df[c_col] > df[h_col])
            close_outside_count = close_outside.sum()
            if close_outside_count > 0:
                issues[f'{timeframe}_issues'].append(
                    f"{close_outside_count:,} rows where close outside [low, high] range"
                )
                outside_dates = df[close_outside][date_col].head(5).tolist()
                issues[f'{timeframe}_issues'].append(f"  Examples: {[d.date() for d in outside_dates]}")

            # Check 5: Duplicate timestamps
            duplicates = df[date_col].duplicated()
            dup_count = duplicates.sum()
            if dup_count > 0:
                issues[f'{timeframe}_issues'].append(
                    f"{dup_count:,} duplicate timestamps"
                )
                dup_dates = df[duplicates][date_col].head(5).tolist()
                issues[f'{timeframe}_issues'].append(f"  Examples: {[d.date() for d in dup_dates]}")

            # Check 6: Constant price bars (open=high=low=close)
            # This isn't always bad (halted trading), but worth noting
            constant_bars = (df[o_col] == df[h_col]) & (df[h_col] == df[l_col]) & (df[l_col] == df[c_col])
            constant_count = constant_bars.sum()
            if constant_count > 100:  # Only flag if many constant bars
                constant_pct = (constant_count / len(df)) * 100
                issues[f'{timeframe}_issues'].append(
                    f"{constant_count:,} rows ({constant_pct:.2f}%) with constant price (OHLC all equal)"
                )

            # Check 7: Time gaps (for hourly only, check for missing bars)
            if timeframe == 'hourly':
                df_sorted = df.sort_values(date_col)
                time_diffs = df_sorted[date_col].diff()

                # Expected max gap: 3 hours (weekend gap + 1 hour bar)
                # Futures trade nearly 24/7, so gaps >24 hours are suspicious
                large_gaps = time_diffs > timedelta(hours=24)
                gap_count = large_gaps.sum()

                if gap_count > 50:  # More than ~weekly gaps
                    issues[f'{timeframe}_issues'].append(
                        f"{gap_count:,} time gaps >24 hours"
                    )
                    # Show largest gaps
                    gap_df = df_sorted[large_gaps][[date_col]].copy()
                    gap_df['gap_hours'] = time_diffs[large_gaps].dt.total_seconds() / 3600
                    largest_gaps = gap_df.nlargest(3, 'gap_hours')
                    for _, row in largest_gaps.iterrows():
                        issues[f'{timeframe}_issues'].append(
                            f"  {row[date_col].date()}: {row['gap_hours']:.1f} hour gap"
                        )

        except Exception as e:
            issues[f'{timeframe}_issues'].append(f"Error reading file: {e}")

    return issues


def main(data_dir='data/resampled'):
    print("=" * 100)
    print("DATA QUALITY CHECK - Zero Values, NaN, Invalid OHLC")
    print("=" * 100)
    print()

    all_issues = []

    for symbol in REFERENCE_SYMBOLS:
        issues = check_symbol_quality(symbol, data_dir)

        has_issues = len(issues['hourly_issues']) > 0 or len(issues['daily_issues']) > 0

        if has_issues:
            all_issues.append(issues)

    # Print results
    if not all_issues:
        print("✓ NO DATA QUALITY ISSUES FOUND")
        print()
        print("All symbols have clean data:")
        print("  - No zero OHLC values")
        print("  - No NaN values")
        print("  - No invalid high/low relationships")
        print("  - No close values outside [low, high]")
        print("  - No duplicate timestamps")
        print()
        print("Your resampled data is ready for feature extraction!")
        print()
        return 0

    print(f"⚠ FOUND ISSUES IN {len(all_issues)} SYMBOLS")
    print()

    for issue_set in all_issues:
        symbol = issue_set['symbol']
        hourly = issue_set['hourly_issues']
        daily = issue_set['daily_issues']

        if hourly or daily:
            print("=" * 100)
            print(f"SYMBOL: {symbol}")
            print("=" * 100)

            if hourly:
                print()
                print("Hourly Issues:")
                for issue in hourly:
                    print(f"  {issue}")

            if daily:
                print()
                print("Daily Issues:")
                for issue in daily:
                    print(f"  {issue}")

            print()

    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()
    print("If symbols have zero/NaN values:")
    print("  1. Re-run resampling: python research/utils/resample_data.py --symbols <SYMBOL>")
    print("  2. Check source data quality in data/historical/")
    print()
    print("If symbols have invalid OHLC (high<low, close outside range):")
    print("  1. This indicates corrupt source data")
    print("  2. Exclude these date ranges or re-download source data")
    print()
    print("If symbols have many constant price bars:")
    print("  1. This may be normal (halted trading, illiquid periods)")
    print("  2. Or could indicate stale/missing data")
    print()

    return 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check data quality in resampled CSVs')
    parser.add_argument('--data-dir', default='data/resampled', help='Resampled data directory')

    args = parser.parse_args()

    sys.exit(main(args.data_dir))
