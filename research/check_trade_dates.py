#!/usr/bin/env python3
"""Check date ranges in trade data files"""
import pandas as pd
from pathlib import Path
import sys

results_dir = Path('results') if len(sys.argv) < 2 else Path(sys.argv[1])

if not results_dir.exists():
    print(f"ERROR: {results_dir} not found")
    sys.exit(1)

print(f"Checking trade data date ranges in {results_dir}\n")
print(f"{'Symbol':<8} {'First Trade':<20} {'Last Trade':<20} {'# Trades':<10} {'Years'}")
print("-" * 80)

all_symbols = []

for opt_dir in sorted(results_dir.glob('*_optimization')):
    symbol = opt_dir.name.replace('_optimization', '')
    trades_file = opt_dir / f"{symbol}_rf_best_trades.csv"

    if not trades_file.exists():
        continue

    try:
        df = pd.read_csv(trades_file)

        # Filter to selected trades only
        if 'Model_Selected' in df.columns:
            df = df[df['Model_Selected'] == 1]
        elif 'selected' in df.columns:
            df = df[df['selected'] == 1]

        if len(df) == 0:
            continue

        # Parse entry dates
        df['entry_time'] = pd.to_datetime(df['Date/Time'])

        first_date = df['entry_time'].min()
        last_date = df['entry_time'].max()
        n_trades = len(df)

        first_str = first_date.strftime('%Y-%m-%d %H:%M')
        last_str = last_date.strftime('%Y-%m-%d %H:%M')

        year_span = f"{first_date.year}-{last_date.year}"

        print(f"{symbol:<8} {first_str:<20} {last_str:<20} {n_trades:<10} {year_span}")

        all_symbols.append({
            'symbol': symbol,
            'first': first_date,
            'last': last_date,
            'n_trades': n_trades
        })

    except Exception as e:
        print(f"{symbol:<8} ERROR: {e}")

if all_symbols:
    print("\n" + "="*80)
    overall_first = min(s['first'] for s in all_symbols)
    overall_last = max(s['last'] for s in all_symbols)
    total_trades = sum(s['n_trades'] for s in all_symbols)

    print(f"OVERALL DATE RANGE: {overall_first.strftime('%Y-%m-%d')} to {overall_last.strftime('%Y-%m-%d')}")
    print(f"TOTAL TRADES (ML-filtered): {total_trades:,}")
    print(f"TIME SPAN: {(overall_last - overall_first).days / 365.25:.2f} years")
else:
    print("\nNo trade data found!")
