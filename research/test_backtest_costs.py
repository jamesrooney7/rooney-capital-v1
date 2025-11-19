#!/usr/bin/env python3
"""
Test script to diagnose backtest costs and metrics.

Tests the same parameters with different cost configurations to see impact.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.vectorized_backtest import run_backtest, load_data

# Test parameters (using reasonable defaults)
TEST_PARAMS = {
    'ibs_entry_low': 0.0,
    'ibs_entry_high': 0.25,
    'ibs_exit_low': 0.75,
    'ibs_exit_high': 1.0,
    'stop_atr_mult': 3.0,
    'target_atr_mult': 2.0,
    'max_holding_bars': 10,
    'atr_period': 14,
    'auto_close_hour': 15,
}

def run_cost_comparison():
    """Compare results with different cost configurations."""

    print("\n" + "="*80)
    print("BACKTEST COST DIAGNOSIS")
    print("="*80)

    # Load 2016 data for testing
    print("\nLoading data (2016)...")
    data = load_data('ES', start_date='2016-01-01', end_date='2016-12-31')
    print(f"Loaded {len(data):,} bars")

    # Test 1: No costs (baseline)
    print("\n" + "="*80)
    print("TEST 1: NO COSTS (Baseline)")
    print("="*80)
    results_no_costs = run_backtest(
        data,
        TEST_PARAMS,
        symbol='ES',
        commission_per_side=0.0,
        slippage_entry=0.0,
        slippage_exit=0.0
    )
    print_results(results_no_costs)

    # Test 2: Commission only
    print("\n" + "="*80)
    print("TEST 2: COMMISSION ONLY ($9/round trip)")
    print("="*80)
    results_commission = run_backtest(
        data,
        TEST_PARAMS,
        symbol='ES',
        commission_per_side=4.50,
        slippage_entry=0.0,
        slippage_exit=0.0
    )
    print_results(results_commission)

    # Test 3: Realistic execution (limit entry, market exit)
    print("\n" + "="*80)
    print("TEST 3: REALISTIC EXECUTION (Entry: limit 0 pts, Exit: market 0.50 pts)")
    print("="*80)
    results_realistic = run_backtest(
        data,
        TEST_PARAMS,
        symbol='ES',
        commission_per_side=4.50,
        slippage_entry=0.0,
        slippage_exit=0.50
    )
    print_results(results_realistic)

    # Test 4: Conservative market orders both ways
    print("\n" + "="*80)
    print("TEST 4: CONSERVATIVE (0.25 pts each way = 0.50 pts total)")
    print("="*80)
    results_conservative = run_backtest(
        data,
        TEST_PARAMS,
        symbol='ES',
        commission_per_side=4.50,
        slippage_entry=0.25,
        slippage_exit=0.25
    )
    print_results(results_conservative)

    # Test 5: Optimistic execution (better exit fills)
    print("\n" + "="*80)
    print("TEST 5: OPTIMISTIC (Entry: limit 0 pts, Exit: market 0.25 pts)")
    print("="*80)
    results_optimistic = run_backtest(
        data,
        TEST_PARAMS,
        symbol='ES',
        commission_per_side=4.50,
        slippage_entry=0.0,
        slippage_exit=0.25
    )
    print_results(results_optimistic)

    # Summary comparison
    print("\n" + "="*80)
    print("COST IMPACT SUMMARY")
    print("="*80)

    baseline_sharpe = results_no_costs['sharpe_ratio']
    baseline_avg_pnl = results_no_costs['avg_pnl_per_trade']

    configs = [
        ("No costs", results_no_costs),
        ("Commission only", results_commission),
        ("Realistic (0/0.50)", results_realistic),
        ("Conservative (0.25/0.25)", results_conservative),
        ("Optimistic (0/0.25)", results_optimistic),
    ]

    print(f"\n{'Config':<25} {'Sharpe':>8} {'Δ Sharpe':>10} {'Avg P&L':>10} {'Δ P&L':>10}")
    print("-" * 80)
    for name, results in configs:
        sharpe = results['sharpe_ratio']
        avg_pnl = results['avg_pnl_per_trade']
        delta_sharpe = sharpe - baseline_sharpe
        delta_pnl = avg_pnl - baseline_avg_pnl
        print(f"{name:<25} {sharpe:>8.3f} {delta_sharpe:>10.3f} {avg_pnl:>10.3f} {delta_pnl:>10.3f}")

    # Check constraints
    print("\n" + "="*80)
    print("OPTIMIZATION CONSTRAINT CHECK (Realistic Execution)")
    print("="*80)

    r = results_realistic
    print(f"\nTrades:    {r['num_trades']:>6,}  (need ≥ 1,500) {'✓' if r['num_trades'] >= 1500 else '✗ FAIL'}")
    print(f"Win Rate:  {r['win_rate']:>6.1%}  (need ≥ 48%)   {'✓' if r['win_rate'] >= 0.48 else '✗ FAIL'}")
    print(f"Sharpe:    {r['sharpe_ratio']:>6.3f}  (need ≥ 0.20)  {'✓' if r['sharpe_ratio'] >= 0.20 else '✗ FAIL'}")

    if r['num_trades'] < 1500 or r['win_rate'] < 0.48 or r['sharpe_ratio'] < 0.20:
        print("\n⚠️  CONSTRAINTS FAILED - This explains the -999999 penalty values!")
        print("\nRecommendations:")
        if r['num_trades'] < 1500:
            print("  - Widen entry threshold (increase ibs_entry_high)")
        if r['win_rate'] < 0.48:
            print("  - Adjust stops/targets ratio")
        if r['sharpe_ratio'] < 0.20:
            print("  - Reduce transaction costs or widen parameter ranges")
    else:
        print("\n✓ All constraints passed!")


def print_results(results):
    """Print formatted backtest results."""
    print(f"\nTrades:        {results['num_trades']:>6,}")
    print(f"Win Rate:      {results['win_rate']:>6.1%}")
    print(f"Sharpe:        {results['sharpe_ratio']:>6.3f}")
    print(f"Avg P&L/Trade: {results['avg_pnl_per_trade']:>6.3f} pts (${results['avg_pnl_per_trade']*50:>7.2f})")
    print(f"Total P&L:     {results['total_pnl']:>6.1f} pts (${results['total_pnl']*50:>8.2f})")
    print(f"Profit Factor: {results['profit_factor']:.2f}" if results['profit_factor'] else "Profit Factor: N/A")
    print(f"Max DD:        {results['max_drawdown_pct']:>6.1%}")


if __name__ == '__main__':
    run_cost_comparison()
