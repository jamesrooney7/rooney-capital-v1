#!/usr/bin/env python3
"""Extract Sharpe ratios from results directory (actual backtest results)"""
import json
from pathlib import Path
import sys

results_dir = Path('results') if len(sys.argv) < 2 else Path(sys.argv[1])

if not results_dir.exists():
    print(f"ERROR: Directory '{results_dir}' not found")
    print("Usage: python3 research/extract_symbol_sharpes.py [results_dir]")
    sys.exit(1)

sharpe_data = []

# Look for optimization directories
for opt_dir in sorted(results_dir.glob('*_optimization')):
    if not opt_dir.is_dir():
        continue

    symbol = opt_dir.name.replace('_optimization', '')

    # Try multiple possible file locations
    sharpe = None
    profit_factor = None
    trades = None

    # Option 1: best.json in the optimization directory
    best_file = opt_dir / 'best.json'
    if best_file.exists():
        try:
            with open(best_file, 'r') as f:
                data = json.load(f)
            sharpe = data.get('Sharpe') or data.get('Sharpe_OOS_CPCV') or data.get('sharpe')
            profit_factor = data.get('Profit_Factor') or data.get('profit_factor')
            trades = data.get('Trades') or data.get('trades')
        except:
            pass

    # Option 2: {SYMBOL}_rf_best_summary.txt
    summary_file = opt_dir / f'{symbol}_rf_best_summary.txt'
    if summary_file.exists() and sharpe is None:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            sharpe = data.get('Sharpe') or data.get('Sharpe_OOS_CPCV') or data.get('sharpe')
            profit_factor = data.get('Profit_Factor') or data.get('profit_factor')
            trades = data.get('Trades') or data.get('trades')
        except:
            pass

    if sharpe is not None:
        sharpe_data.append({
            'symbol': symbol,
            'sharpe': float(sharpe),
            'profit_factor': float(profit_factor) if profit_factor else None,
            'trades': int(trades) if trades else None
        })

if not sharpe_data:
    print(f"ERROR: No optimization results found in {results_dir}")
    print(f"Looking for directories matching: {results_dir}/*_optimization/")
    sys.exit(1)

# Print table
print(f"{'Symbol':<8} {'Sharpe':<10} {'Profit Factor':<15} {'Trades':<10}")
print("-" * 50)
for item in sorted(sharpe_data, key=lambda x: x['symbol']):
    pf_str = f"{item['profit_factor']:.2f}" if item['profit_factor'] else "N/A"
    trades_str = str(item['trades']) if item['trades'] else "N/A"
    print(f"{item['symbol']:<8} {item['sharpe']:<10.4f} {pf_str:<15} {trades_str:<10}")

print("\n" + "="*50)
print(f"Average Sharpe: {sum(d['sharpe'] for d in sharpe_data) / len(sharpe_data):.4f}")
print(f"Number of symbols: {len(sharpe_data)}")

# Calculate theoretical portfolio Sharpe with zero correlation
import math
avg_sharpe = sum(d['sharpe'] for d in sharpe_data) / len(sharpe_data)
n_symbols = len(sharpe_data)
theoretical_sharpe_zero_corr = avg_sharpe * math.sqrt(n_symbols)

print(f"\n" + "="*50)
print("THEORETICAL PORTFOLIO SHARPE (Zero Correlation):")
print(f"  Formula: Avg Sharpe × √N")
print(f"  = {avg_sharpe:.4f} × √{n_symbols}")
print(f"  = {theoretical_sharpe_zero_corr:.4f}")
