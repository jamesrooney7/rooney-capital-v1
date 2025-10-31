#!/usr/bin/env python3
"""
Simple Portfolio Optimizer - Uses Pre-Computed Results

This script uses the optimization results already stored in src/models/
and estimates portfolio performance without re-running backtests.

This is MUCH faster than the full optimizer and suitable for quick analysis.

Usage:
    # Optimize using existing results
    python research/portfolio_optimizer_simple.py \
        --min-positions 1 --max-positions 10

    # Specify custom symbols
    python research/portfolio_optimizer_simple.py \
        --symbols ES NQ YM RTY \
        --output results/portfolio_quick.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def load_optimization_results(models_dir: Path, symbols: List[str] = None) -> Dict[str, dict]:
    """
    Load pre-computed optimization results from src/models/.

    Returns:
        Dict of {symbol: metadata}
    """
    results = {}

    # Auto-discover if no symbols specified
    if symbols is None:
        symbols = []
        for json_file in models_dir.glob('*_best.json'):
            symbol = json_file.stem.replace('_best', '')
            pkl_file = models_dir / f"{symbol}_rf_model.pkl"
            if pkl_file.exists():
                symbols.append(symbol)
        symbols = sorted(symbols)

    print(f"Loading optimization results for {len(symbols)} symbols...")
    print("=" * 80)

    for symbol in symbols:
        json_file = models_dir / f"{symbol}_best.json"

        if not json_file.exists():
            print(f"⚠️  No results found for {symbol}")
            continue

        with open(json_file, 'r') as f:
            metadata = json.load(f)

        results[symbol] = metadata

        sharpe = metadata.get('Sharpe', 0)
        pf = metadata.get('Profit_Factor', 0)
        trades = metadata.get('Trades', 0)

        print(f"{symbol:6s} | Sharpe: {sharpe:6.3f} | PF: {pf:6.2f} | Trades: {trades:4d}")

    print("=" * 80)
    print(f"✅ Loaded {len(results)} optimization results\n")

    return results


def estimate_portfolio_performance(
    results: Dict[str, dict],
    max_positions: int,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Estimate portfolio performance based on individual symbol metrics.

    This is an approximation that assumes:
    - Equal allocation across max_positions
    - Returns scale linearly with capital allocation
    - Correlation between symbols reduces aggregate volatility
    """
    # Filter to top symbols by Sharpe ratio
    sorted_symbols = sorted(
        results.items(),
        key=lambda x: x[1].get('Sharpe', 0),
        reverse=True
    )

    selected_symbols = sorted_symbols[:max_positions]

    if not selected_symbols:
        return {
            'max_positions': max_positions,
            'n_symbols': 0,
            'sharpe_ratio': 0,
            'cagr': 0,
            'profit_factor': 0,
            'total_return_dollars': 0,
            'max_drawdown_dollars': 0
        }

    # Calculate portfolio metrics
    n_symbols = len(selected_symbols)

    # Average metrics (weighted by Sharpe)
    total_sharpe = sum(meta.get('Sharpe', 0) for _, meta in selected_symbols)
    avg_sharpe = total_sharpe / n_symbols if n_symbols > 0 else 0

    avg_pf = np.mean([meta.get('Profit_Factor', 1.0) for _, meta in selected_symbols])

    # Portfolio Sharpe increases with diversification (sqrt rule approximation)
    # Assume average correlation of 0.3 between symbols
    avg_correlation = 0.3
    diversification_benefit = np.sqrt(n_symbols * (1 - avg_correlation) + n_symbols * avg_correlation)
    portfolio_sharpe = avg_sharpe * np.sqrt(diversification_benefit) / np.sqrt(n_symbols)

    # Estimate CAGR from Sharpe
    # Typical vol for futures strategies: 15-20%
    estimated_vol = 0.15 / np.sqrt(n_symbols)  # Reduces with diversification
    estimated_cagr = portfolio_sharpe * estimated_vol

    # Estimate total return in dollars
    # This is rough - assumes equal allocation and average performance
    capital_per_position = initial_capital / n_symbols
    total_return_dollars = initial_capital * estimated_cagr

    # Estimate max drawdown
    # Rule of thumb: MaxDD ≈ 2 * annual_vol for mean-reverting strategies
    estimated_max_dd_pct = 2 * estimated_vol
    estimated_max_dd_dollars = initial_capital * estimated_max_dd_pct

    # Estimate stop-outs
    # More positions = higher chance of hitting daily stop
    # Rough approximation: 1 stop per 100 trading days per position
    estimated_stops = int(n_symbols * periods_per_year / 100)

    return {
        'max_positions': max_positions,
        'n_symbols': n_symbols,
        'sharpe_ratio': portfolio_sharpe,
        'cagr': estimated_cagr,
        'profit_factor': avg_pf,
        'total_return_dollars': total_return_dollars,
        'max_drawdown_pct': estimated_max_dd_pct,
        'max_drawdown_dollars': estimated_max_dd_dollars,
        'daily_stops_hit': estimated_stops,
        'symbols': [sym for sym, _ in selected_symbols]
    }


def optimize_portfolio(
    results: Dict[str, dict],
    min_positions: int = 1,
    max_positions: int = None,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> pd.DataFrame:
    """
    Optimize max_positions parameter using pre-computed results.

    Returns:
        DataFrame with estimated performance for each configuration
    """
    n_symbols = len(results)
    max_pos = min(max_positions or n_symbols, n_symbols)

    print(f"\n{'='*90}")
    print(f"PORTFOLIO OPTIMIZATION USING PRE-COMPUTED RESULTS")
    print(f"Initial Capital: ${initial_capital:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    print(f"{'='*90}\n")

    print("⚠️  NOTE: These are ESTIMATES based on individual symbol metrics.")
    print("    For accurate results, run full backtests with actual trade data.\n")

    optimization_results = []

    for max_pos_val in range(min_positions, max_pos + 1):
        print(f"Estimating max_positions = {max_pos_val}...", end=' ')

        metrics = estimate_portfolio_performance(
            results,
            max_pos_val,
            initial_capital=initial_capital,
            daily_stop_loss=daily_stop_loss
        )

        optimization_results.append(metrics)

        print(f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"CAGR: {metrics['cagr']*100:.2f}% | "
              f"PF: {metrics['profit_factor']:.2f}")

    # Create DataFrame
    results_df = pd.DataFrame(optimization_results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    # Print summary
    print(f"\n{'='*90}")
    print("OPTIMIZATION RESULTS (sorted by Estimated Sharpe)")
    print(f"{'='*90}\n")

    print(f"{'MaxPos':<10}{'Sharpe':<10}{'CAGR%':<10}{'Return $':<15}{'MaxDD $':<15}{'PF':<10}{'Stops':<10}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        print(f"{row['max_positions']:<10.0f}"
              f"{row['sharpe_ratio']:<10.3f}"
              f"{row['cagr']*100:<10.2f}"
              f"${row['total_return_dollars']:>13,.0f}"
              f"${row['max_drawdown_dollars']:>13,.0f}"
              f"{row['profit_factor']:<10.2f}"
              f"{row['daily_stops_hit']:<10.0f}")

    print("=" * 90)

    # Best result
    best = results_df.iloc[0]
    print(f"\n🏆 RECOMMENDED CONFIGURATION (Estimated):")
    print(f"   Max Positions: {int(best['max_positions'])}")
    print(f"   Selected Symbols: {', '.join(best['symbols'])}")
    print(f"   Estimated Sharpe: {best['sharpe_ratio']:.3f}")
    print(f"   Estimated CAGR: {best['cagr']*100:.2f}%")
    print(f"   Estimated Return: ${best['total_return_dollars']:,.2f}")
    print(f"   Estimated MaxDD: ${best['max_drawdown_dollars']:,.2f} ({best['max_drawdown_pct']*100:.2f}%)")
    print(f"   Avg Profit Factor: {best['profit_factor']:.2f}")
    print(f"   Est. Daily Stops: {best['daily_stops_hit']:.0f}\n")

    print("💡 TIP: For accurate results, consider running actual backtests")
    print("   with the selected symbols using portfolio_optimizer_full.py\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Quick portfolio optimizer using pre-computed results'
    )

    parser.add_argument('--symbols', nargs='+', help='Symbols to include (default: auto-discover)')
    parser.add_argument('--models-dir', type=str, default='src/models', help='Models directory')
    parser.add_argument('--min-positions', type=int, default=1, help='Min positions to test')
    parser.add_argument('--max-positions', type=int, default=None, help='Max positions to test')
    parser.add_argument('--initial-cash', type=float, default=250000.0, help='Initial capital')
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0, help='Daily stop loss')
    parser.add_argument('--output', type=str, help='Output CSV file')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print(f"   Make sure you've run model optimization first.")
        return

    # Load optimization results
    results = load_optimization_results(models_dir, args.symbols)

    if not results:
        print("❌ No optimization results found.")
        print(f"   Check that {models_dir} contains *_best.json files")
        return

    # Run optimization
    results_df = optimize_portfolio(
        results,
        min_positions=args.min_positions,
        max_positions=args.max_positions,
        initial_capital=args.initial_cash,
        daily_stop_loss=args.daily_stop_loss
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save without the 'symbols' column (list of symbols)
        save_df = results_df.drop(columns=['symbols'])
        save_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
