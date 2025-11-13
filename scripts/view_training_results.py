#!/usr/bin/env python3
"""
View and compare training results for all symbols.
Shows test performance vs baseline performance.
"""

import json
from pathlib import Path
import sys

# Baseline performance (unfiltered, from earlier analysis)
BASELINE = {
    'ES': {'total_pnl_usd': -309665.50, 'trades': 16939, 'win_rate': 54.3, 'sharpe': -0.501, 'pf': 0.907},
    'NQ': {'total_pnl_usd': -183838.00, 'trades': 16814, 'win_rate': 57.3, 'sharpe': -0.152, 'pf': 0.960},
    'RTY': {'total_pnl_usd': -32693.00, 'trades': 17334, 'win_rate': 56.5, 'sharpe': -0.035, 'pf': 0.986},
    'YM': {'total_pnl_usd': -68597.00, 'trades': 16956, 'win_rate': 56.6, 'sharpe': -0.076, 'pf': 0.974},
}

def load_test_results(symbol):
    """Load test results for a symbol."""
    test_path = Path('src/models') / f'{symbol}_test_results.json'

    if not test_path.exists():
        return None

    with open(test_path) as f:
        data = json.load(f)

    return data.get('test_metrics', {})

def load_metadata(symbol):
    """Load model metadata for a symbol."""
    meta_path = Path('src/models') / f'{symbol}_best.json'

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        return json.load(f)

def main():
    symbols = ['ES', 'NQ', 'RTY', 'YM']

    print("=" * 100)
    print("TRAINING RESULTS SUMMARY (Test Period: 2021-2024)")
    print("=" * 100)
    print()

    # Check which symbols have completed
    completed = []
    for symbol in symbols:
        results = load_test_results(symbol)
        if results:
            completed.append(symbol)

    if not completed:
        print("❌ No training results found yet. Training may still be in progress.")
        print("\nCheck training status with:")
        print("  ps aux | grep train_rf_three_way_split.py | grep -v grep")
        return 1

    print(f"✅ Completed: {', '.join(completed)}")
    print(f"⏳ In Progress: {', '.join([s for s in symbols if s not in completed])}")
    print()

    # Detailed results for each completed symbol
    for symbol in completed:
        results = load_test_results(symbol)
        meta = load_metadata(symbol)
        baseline = BASELINE.get(symbol, {})

        print("=" * 100)
        print(f"{symbol} - RF FILTERED vs BASELINE (Unfiltered)")
        print("=" * 100)

        # RF Filtered Performance
        pnl = results.get('total_pnl_usd', 0)
        trades = results.get('trades', 0)
        sharpe = results.get('sharpe', 0)
        sortino = results.get('sortino', 0)
        pf = results.get('profit_factor', 0)
        win_rate = results.get('win_rate', 0) * 100
        cagr = results.get('cagr', 0) * 100
        max_dd_pct = results.get('max_drawdown_pct', 0) * 100
        max_dd_usd = results.get('max_drawdown_usd', 0)

        print(f"\n📊 RF FILTERED (Test Period 2021-2024):")
        print(f"  Total P&L:       ${pnl:>15,.2f}")
        print(f"  Trades:          {trades:>15,}")
        print(f"  Win Rate:        {win_rate:>15.1f}%")
        print(f"  Sharpe Ratio:    {sharpe:>15.2f}")
        print(f"  Sortino Ratio:   {sortino:>15.2f}")
        print(f"  Profit Factor:   {pf:>15.2f}")
        print(f"  CAGR:            {cagr:>15.1f}%")
        print(f"  Max Drawdown:    {max_dd_pct:>14.1f}% (${max_dd_usd:,.0f})")

        # Baseline Performance
        if baseline:
            print(f"\n📉 BASELINE (All trades 2011-2024, unfiltered):")
            print(f"  Total P&L:       ${baseline['total_pnl_usd']:>15,.2f}")
            print(f"  Trades:          {baseline['trades']:>15,}")
            print(f"  Win Rate:        {baseline['win_rate']:>15.1f}%")
            print(f"  Sharpe Ratio:    {baseline['sharpe']:>15.2f}")
            print(f"  Profit Factor:   {baseline['pf']:>15.2f}")

            # Calculate improvement
            pnl_improvement = pnl - baseline['total_pnl_usd']
            sharpe_improvement = sharpe - baseline['sharpe']
            pf_improvement = pf - baseline['pf']

            print(f"\n💡 IMPROVEMENT (RF Filter Impact):")
            print(f"  P&L Improvement: ${pnl_improvement:>15,.2f}")
            print(f"  Sharpe Δ:        {sharpe_improvement:>15.2f}")
            print(f"  PF Δ:            {pf_improvement:>15.2f}")
            print(f"  Trade Reduction: {baseline['trades'] - trades:>15,} trades filtered out ({100*(baseline['trades']-trades)/baseline['trades']:.1f}%)")

        # Year-by-year breakdown
        yearly = results.get('yearly_breakdown', {})
        if yearly:
            print(f"\n📅 YEAR-BY-YEAR PERFORMANCE:")
            print(f"  {'Year':<8} {'Trades':<10} {'P&L':<20} {'Win Rate':<12}")
            print(f"  {'-'*8} {'-'*10} {'-'*20} {'-'*12}")
            for year in sorted(yearly.keys()):
                y = yearly[year]
                print(f"  {year:<8} {y['trades']:<10,} ${y['pnl_usd']:<18,.2f} {y['win_rate']*100:<11.1f}%")

        # Model info
        if meta:
            threshold = meta.get('threshold', 0.5)
            print(f"\n🔧 MODEL CONFIGURATION:")
            print(f"  Threshold:       {threshold:.2f}")
            print(f"  Features:        {len(meta.get('features', []))}")
            print(f"  Training Period: {meta['train_period']['start']} to {meta['train_period']['end']}")
            print(f"  Training Trades: {meta['train_period']['trades']:,}")

        print()

    # Summary table
    if len(completed) > 1:
        print("=" * 100)
        print("COMPARISON TABLE")
        print("=" * 100)
        print(f"{'Symbol':<8} {'Test P&L':<18} {'Baseline P&L':<18} {'Improvement':<18} {'Sharpe':<8} {'PF':<8}")
        print("-" * 100)

        for symbol in completed:
            results = load_test_results(symbol)
            baseline = BASELINE.get(symbol, {})

            pnl = results.get('total_pnl_usd', 0)
            sharpe = results.get('sharpe', 0)
            pf = results.get('profit_factor', 0)
            baseline_pnl = baseline.get('total_pnl_usd', 0)
            improvement = pnl - baseline_pnl

            print(f"{symbol:<8} ${pnl:<16,.0f} ${baseline_pnl:<16,.0f} ${improvement:<16,.0f} {sharpe:<8.2f} {pf:<8.2f}")

        print()

if __name__ == '__main__':
    sys.exit(main())
