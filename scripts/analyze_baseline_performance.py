#!/usr/bin/env python3
"""
Analyze baseline (unfiltered) performance of extracted training data.
Shows metrics comparable to RF training results.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_metrics(df: pd.DataFrame, symbol: str) -> dict:
    """Calculate trading metrics matching RF training output format."""

    # Ensure we have the required columns
    required = ['y_return', 'y_pnl_usd']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop any NaN returns
    df_clean = df.dropna(subset=['y_return', 'y_pnl_usd'])

    total_trades = len(df_clean)
    if total_trades == 0:
        return None

    # Win/loss metrics
    winners = (df_clean['y_return'] > 0).sum()
    losers = (df_clean['y_return'] < 0).sum()
    breakeven = (df_clean['y_return'] == 0).sum()
    win_rate = (winners / total_trades) * 100 if total_trades > 0 else 0

    # P&L metrics
    total_pnl = df_clean['y_pnl_usd'].sum()
    avg_pnl = df_clean['y_pnl_usd'].mean()

    # Profit factor
    gross_profit = df_clean[df_clean['y_pnl_usd'] > 0]['y_pnl_usd'].sum()
    gross_loss = abs(df_clean[df_clean['y_pnl_usd'] < 0]['y_pnl_usd'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Sharpe ratio (annualized, assuming ~252 trading days)
    returns = df_clean['y_return'].values
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Drawdown calculation (simplified - based on cumulative P&L)
    cum_pnl = df_clean['y_pnl_usd'].cumsum()
    running_max = cum_pnl.expanding().max()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'winners': winners,
        'losers': losers,
        'breakeven': breakeven,
        'win_rate': win_rate,
        'total_pnl_usd': total_pnl,
        'avg_pnl_usd': avg_pnl,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }


def main():
    data_dir = Path('data/training')
    symbols = ['ES', 'NQ', 'RTY', 'YM']

    print("=" * 80)
    print("BASELINE PERFORMANCE (Unfiltered Training Data)")
    print("=" * 80)
    print()

    results = []

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_transformed_features.csv"

        if not csv_path.exists():
            print(f"❌ {symbol}: File not found at {csv_path}")
            continue

        # Load data
        df = pd.read_csv(csv_path)

        # Calculate metrics
        metrics = calculate_metrics(df, symbol)

        if metrics is None:
            print(f"❌ {symbol}: No valid trades found")
            continue

        results.append(metrics)

        # Print results
        print(f"{symbol} (2011-2024 unfiltered):")
        print(f"  Trades:        {metrics['total_trades']:,}")
        print(f"  Winners:       {metrics['winners']:,} ({metrics['win_rate']:.1f}%)")
        print(f"  Losers:        {metrics['losers']:,}")
        print(f"  Breakeven:     {metrics['breakeven']:,}")
        print(f"  Total P&L:     ${metrics['total_pnl_usd']:,.2f}")
        print(f"  Avg P&L:       ${metrics['avg_pnl_usd']:.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Sharpe Ratio:  {metrics['sharpe']:.3f}")
        print(f"  Max Drawdown:  ${metrics['max_drawdown']:,.2f}")
        print()

    # Summary comparison table
    if results:
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Symbol':<8} {'Trades':<8} {'Win%':<8} {'Total P&L':<15} {'Sharpe':<8} {'PF':<8}")
        print("-" * 80)
        for r in results:
            print(f"{r['symbol']:<8} {r['total_trades']:<8,} "
                  f"{r['win_rate']:<8.1f} ${r['total_pnl_usd']:<14,.0f} "
                  f"{r['sharpe']:<8.2f} {r['profit_factor']:<8.2f}")
        print()


if __name__ == '__main__':
    main()
