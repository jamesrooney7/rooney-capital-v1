#!/usr/bin/env python3
"""
Portfolio Simulator with Train/Test Split and Greedy Optimization

Loads pre-existing trade data from results/ directory and:
1. Filters trades by date to create train/test periods
2. Tests max_positions on train period (finds optimal)
3. Validates on test period with optimal configuration
4. Shows detailed per-instrument statistics

Much faster than re-running backtests - uses existing trade data!

Usage:
    python research/portfolio_simulator_train_test.py \
        --results-dir results \
        --train-start 2019-01-01 --train-end 2021-12-31 \
        --test-start 2022-01-01 --test-end 2024-12-31 \
        --min-positions 1 --max-positions 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import from portfolio_simulator
from portfolio_simulator import (
    discover_available_symbols,
    load_symbol_trades,
    simulate_portfolio_intraday,
    optimize_max_positions
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_trades_by_date(
    trades_df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Filter trades to only include those within date range."""
    # Convert date columns to datetime
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    # Filter by entry time
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    filtered = trades_df[
        (trades_df['entry_time'] >= start) &
        (trades_df['entry_time'] <= end)
    ].copy()

    return filtered


def calculate_symbol_statistics(
    symbol_trades: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Calculate per-symbol statistics."""
    stats = []

    for symbol, trades_df in symbol_trades.items():
        n_trades = len(trades_df)

        if n_trades == 0:
            continue

        winning_trades = trades_df[trades_df['pnl_usd'] > 0]
        losing_trades = trades_df[trades_df['pnl_usd'] < 0]

        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        avg_win = winning_trades['pnl_usd'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_usd'].mean() if len(losing_trades) > 0 else 0

        total_pnl = trades_df['pnl_usd'].sum()
        gross_profit = winning_trades['pnl_usd'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl_usd'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate Sharpe (approximate)
        returns = trades_df['pnl_usd'] / 250000.0  # Assuming 250k capital
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        stats.append({
            'symbol': symbol,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'sharpe': sharpe
        })

    return pd.DataFrame(stats).sort_values('sharpe', ascending=False)


def print_symbol_statistics(stats_df: pd.DataFrame, period_name: str):
    """Print formatted symbol statistics table."""
    logger.info(f"\n{'='*100}")
    logger.info(f"PER-INSTRUMENT STATISTICS - {period_name}")
    logger.info(f"{'='*100}\n")

    print(f"{'Symbol':<8}{'Trades':<10}{'Win%':<10}{'Avg Win':<12}{'Avg Loss':<12}"
          f"{'Total P&L':<15}{'PF':<10}{'Sharpe':<10}")
    print("-" * 100)

    for _, row in stats_df.iterrows():
        print(f"{row['symbol']:<8}"
              f"{row['n_trades']:<10.0f}"
              f"{row['win_rate']*100:<10.1f}"
              f"${row['avg_win']:<11,.0f}"
              f"${row['avg_loss']:<11,.0f}"
              f"${row['total_pnl']:<14,.0f}"
              f"{row['profit_factor']:<10.2f}"
              f"{row['sharpe']:<10.2f}")

    print("=" * 100)

    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total Instruments: {len(stats_df)}")
    logger.info(f"  Total Trades: {stats_df['n_trades'].sum():.0f}")
    logger.info(f"  Avg Win Rate: {stats_df['win_rate'].mean()*100:.1f}%")
    logger.info(f"  Avg Sharpe: {stats_df['sharpe'].mean():.2f}")
    logger.info(f"  Total P&L: ${stats_df['total_pnl'].sum():,.0f}")
    logger.info(f"  Best Instrument: {stats_df.iloc[0]['symbol']} (Sharpe: {stats_df.iloc[0]['sharpe']:.2f})")
    logger.info(f"  Worst Instrument: {stats_df.iloc[-1]['symbol']} (Sharpe: {stats_df.iloc[-1]['sharpe']:.2f})\n")


def print_comparison_summary(
    train_results_df: pd.DataFrame,
    test_metrics: Dict,
    optimal_max_positions: int,
    train_period: str,
    test_period: str
):
    """Print side-by-side comparison."""
    logger.info(f"\n{'='*100}")
    logger.info(f"TRAIN vs TEST COMPARISON")
    logger.info(f"{'='*100}\n")

    # Get optimal train metrics
    train_optimal = train_results_df[train_results_df['max_positions'] == optimal_max_positions].iloc[0]

    comparison_data = {
        'Metric': [
            'Period',
            'Max Positions',
            'Sharpe Ratio',
            'CAGR',
            'Total Return $',
            'Max Drawdown $',
            'Max Drawdown %',
            'Profit Factor',
            'Daily Stops Hit',
            'Avg Positions'
        ],
        'Train (In-Sample)': [
            train_period,
            f"{int(train_optimal['max_positions'])}",
            f"{train_optimal['sharpe_ratio']:.3f}",
            f"{train_optimal['cagr']*100:.2f}%",
            f"${train_optimal['total_return_dollars']:,.0f}",
            f"${train_optimal['max_drawdown_dollars']:,.0f}",
            f"{train_optimal['max_drawdown_pct']*100:.1f}%",
            f"{train_optimal['profit_factor']:.2f}",
            f"{train_optimal['daily_stops_hit']:.0f}",
            f"{train_optimal['avg_positions']:.2f}"
        ],
        'Test (Out-of-Sample)': [
            test_period,
            f"{optimal_max_positions}",
            f"{test_metrics['sharpe_ratio']:.3f}",
            f"{test_metrics['cagr']*100:.2f}%",
            f"${test_metrics['total_return_dollars']:,.0f}",
            f"${test_metrics['max_drawdown_dollars']:,.0f}",
            f"{test_metrics['max_drawdown_pct']*100:.1f}%",
            f"{test_metrics['profit_factor']:.2f}",
            f"{test_metrics['daily_stops_hit']:.0f}",
            f"{test_metrics['avg_positions']:.2f}"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("=" * 100)

    # Performance assessment
    sharpe_ratio = test_metrics['sharpe_ratio'] / train_optimal['sharpe_ratio']

    logger.info(f"\n📊 PERFORMANCE ASSESSMENT:")
    if sharpe_ratio > 0.9:
        logger.info(f"   ✅ EXCELLENT: Test Sharpe is {sharpe_ratio*100:.1f}% of train Sharpe")
        logger.info(f"   → Strategy generalizes well to out-of-sample data")
    elif sharpe_ratio > 0.7:
        logger.info(f"   ⚠️  MODERATE: Test Sharpe is {sharpe_ratio*100:.1f}% of train Sharpe")
        logger.info(f"   → Some degradation, but strategy remains viable")
    else:
        logger.info(f"   ❌ POOR: Test Sharpe is {sharpe_ratio*100:.1f}% of train Sharpe")
        logger.info(f"   → Significant overfitting, reconsider strategy")

    logger.info(f"\n")


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio simulator with train/test split using existing trade data'
    )

    # Date ranges
    parser.add_argument('--train-start', type=str, required=True, help='Train start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, required=True, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, required=True, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, required=True, help='Test end date (YYYY-MM-DD)')

    # Other parameters
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory with optimization results')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--initial-capital', type=float, default=250000.0)
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)
    parser.add_argument('--ranking-method', type=str, default='sharpe',
                       choices=['sharpe', 'profit_factor'])
    parser.add_argument('--max-dd-constraint', type=float, default=None,
                       help='Maximum drawdown constraint in dollars (e.g., 5000)')
    parser.add_argument('--output-dir', type=str, default='results')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Discover symbols
    logger.info("="*100)
    logger.info("PORTFOLIO SIMULATOR WITH TRAIN/TEST SPLIT")
    logger.info("="*100)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Train Period: {args.train_start} to {args.train_end}")
    logger.info(f"Test Period: {args.test_start} to {args.test_end}")
    logger.info(f"Using existing trade data (4-tick slippage already applied)")
    logger.info("")

    available_symbols = discover_available_symbols(results_dir)

    # Exclude PL
    if 'PL' in available_symbols:
        available_symbols.remove('PL')
        logger.info("✂️  Excluded PL (Platinum) due to slippage issues")

    logger.info(f"📦 Loading {len(available_symbols)} symbols: {', '.join(sorted(available_symbols))}\n")

    # Load all trade data
    all_symbol_trades = {}
    all_symbol_metadata = {}

    for symbol in available_symbols:
        try:
            trades_df, metadata = load_symbol_trades(results_dir, symbol)
            all_symbol_trades[symbol] = trades_df
            all_symbol_metadata[symbol] = metadata
            logger.info(f"  ✅ {symbol}: {len(trades_df)} trades")
        except Exception as e:
            logger.warning(f"  ❌ {symbol}: Failed to load - {e}")

    if not all_symbol_trades:
        logger.error("❌ No trade data loaded!")
        return 1

    logger.info(f"\n✅ Loaded {len(all_symbol_trades)} symbols successfully\n")

    # ======== FILTER TRAIN PERIOD ========
    logger.info(f"{'#'*100}")
    logger.info(f"# TRAIN PERIOD: {args.train_start} to {args.train_end}")
    logger.info(f"{'#'*100}\n")

    train_symbol_trades = {}
    for symbol, trades_df in all_symbol_trades.items():
        filtered = filter_trades_by_date(trades_df, args.train_start, args.train_end)
        if len(filtered) > 0:
            train_symbol_trades[symbol] = filtered
            logger.info(f"  {symbol}: {len(filtered)} trades in train period")

    if not train_symbol_trades:
        logger.error("❌ No trades in train period!")
        return 1

    # Calculate and print symbol statistics for train
    train_stats = calculate_symbol_statistics(train_symbol_trades)
    print_symbol_statistics(train_stats, f"TRAIN: {args.train_start} to {args.train_end}")

    # Run optimization on train
    logger.info(f"\n{'='*100}")
    logger.info(f"OPTIMIZING MAX POSITIONS ON TRAIN PERIOD")
    logger.info(f"{'='*100}\n")

    train_results_df = optimize_max_positions(
        symbol_trades=train_symbol_trades,
        symbol_metadata=all_symbol_metadata,
        min_positions=args.min_positions,
        max_positions=args.max_positions,
        initial_capital=args.initial_capital,
        daily_stop_loss=args.daily_stop_loss,
        ranking_method=args.ranking_method
    )

    # Get optimal configuration (with optional constraint)
    if args.max_dd_constraint:
        logger.info(f"\n🎯 APPLYING CONSTRAINT: Max Drawdown < ${args.max_dd_constraint:,.0f}")

        # Filter to only configs that meet constraint
        valid_configs = train_results_df[
            abs(train_results_df['max_drawdown_dollars']) < args.max_dd_constraint
        ]

        if len(valid_configs) == 0:
            logger.error(f"❌ No configurations meet drawdown constraint < ${args.max_dd_constraint:,.0f}")
            logger.info("\nAll tested configurations:")
            for _, row in train_results_df.iterrows():
                logger.info(f"  max_positions={int(row['max_positions'])}: "
                          f"Sharpe={row['sharpe_ratio']:.3f}, "
                          f"DD=${abs(row['max_drawdown_dollars']):,.0f}")
            logger.info("\n💡 Try relaxing the constraint or using fewer max_positions")
            return 1

        # Get best Sharpe among valid configs
        optimal_row = valid_configs.iloc[0]
        optimal_max_positions = int(optimal_row['max_positions'])
        train_optimal_sharpe = optimal_row['sharpe_ratio']

        logger.info(f"✅ Found {len(valid_configs)} configurations meeting constraint")
        logger.info(f"🏆 Best valid config: max_positions={optimal_max_positions}, "
                   f"Sharpe={train_optimal_sharpe:.3f}, "
                   f"DD=${abs(optimal_row['max_drawdown_dollars']):,.0f}\n")
    else:
        # No constraint - just pick best Sharpe
        optimal_max_positions = int(train_results_df.iloc[0]['max_positions'])
        train_optimal_sharpe = train_results_df.iloc[0]['sharpe_ratio']
        logger.info(f"\n✅ TRAIN PERIOD: Optimal max_positions = {optimal_max_positions} (Sharpe: {train_optimal_sharpe:.3f})\n")

    # ======== FILTER TEST PERIOD ========
    logger.info(f"{'#'*100}")
    logger.info(f"# TEST PERIOD: {args.test_start} to {args.test_end}")
    logger.info(f"{'#'*100}\n")

    test_symbol_trades = {}
    for symbol, trades_df in all_symbol_trades.items():
        filtered = filter_trades_by_date(trades_df, args.test_start, args.test_end)
        if len(filtered) > 0:
            test_symbol_trades[symbol] = filtered
            logger.info(f"  {symbol}: {len(filtered)} trades in test period")

    if not test_symbol_trades:
        logger.error("❌ No trades in test period!")
        return 1

    # Calculate and print symbol statistics for test
    test_stats = calculate_symbol_statistics(test_symbol_trades)
    print_symbol_statistics(test_stats, f"TEST: {args.test_start} to {args.test_end}")

    # Apply optimal configuration to test period
    logger.info(f"\n{'='*100}")
    logger.info(f"VALIDATING ON TEST PERIOD (max_positions={optimal_max_positions})")
    logger.info(f"{'='*100}\n")

    test_equity_df, test_metrics = simulate_portfolio_intraday(
        symbol_trades=test_symbol_trades,
        symbol_metadata=all_symbol_metadata,
        max_positions=optimal_max_positions,
        initial_capital=args.initial_capital,
        daily_stop_loss=args.daily_stop_loss,
        ranking_method=args.ranking_method
    )

    logger.info(f"TEST PERIOD RESULTS:")
    logger.info(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  CAGR: {test_metrics['cagr']*100:.2f}%")
    logger.info(f"  Total Return: ${test_metrics['total_return_dollars']:,.2f}")
    logger.info(f"  Max Drawdown: ${test_metrics['max_drawdown_dollars']:,.2f}")
    logger.info(f"  Profit Factor: {test_metrics['profit_factor']:.2f}")
    logger.info(f"  Daily Stops Hit: {test_metrics['daily_stops_hit']:.0f}\n")

    # ======== COMPARISON ========
    print_comparison_summary(
        train_results_df,
        test_metrics,
        optimal_max_positions,
        f"{args.train_start} to {args.train_end}",
        f"{args.test_start} to {args.test_end}"
    )

    # ======== SAVE RESULTS ========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save train results
    train_file = output_dir / f'portfolio_train_{timestamp}.csv'
    train_results_df.to_csv(train_file, index=False)
    logger.info(f"💾 Train results saved to: {train_file}")

    # Save test results
    test_results_df = pd.DataFrame([{
        'max_positions': optimal_max_positions,
        **test_metrics
    }])
    test_file = output_dir / f'portfolio_test_{timestamp}.csv'
    test_results_df.to_csv(test_file, index=False)
    logger.info(f"💾 Test results saved to: {test_file}")

    # Save symbol statistics
    train_stats_file = output_dir / f'symbol_stats_train_{timestamp}.csv'
    train_stats.to_csv(train_stats_file, index=False)
    logger.info(f"💾 Train symbol stats saved to: {train_stats_file}")

    test_stats_file = output_dir / f'symbol_stats_test_{timestamp}.csv'
    test_stats.to_csv(test_stats_file, index=False)
    logger.info(f"💾 Test symbol stats saved to: {test_stats_file}")

    # Save summary
    summary = {
        'train_period': {'start': args.train_start, 'end': args.train_end},
        'test_period': {'start': args.test_start, 'end': args.test_end},
        'optimal_max_positions': optimal_max_positions,
        'train_sharpe': float(train_optimal_sharpe),
        'test_sharpe': float(test_metrics['sharpe_ratio']),
        'generalization_ratio': float(test_metrics['sharpe_ratio'] / train_optimal_sharpe),
        'symbols_used': sorted(list(train_symbol_trades.keys())),
        'excluded_symbols': ['PL'],
        'slippage': '4-tick round-trip (already applied in trade data)',
        'timestamp': timestamp
    }

    summary_file = output_dir / f'portfolio_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"💾 Summary saved to: {summary_file}\n")

    logger.info(f"{'='*100}")
    logger.info(f"✅ OPTIMIZATION COMPLETE")
    logger.info(f"{'='*100}\n")
    logger.info(f"🏆 Optimal Configuration: max_positions = {optimal_max_positions}")
    logger.info(f"📈 Train Sharpe: {train_optimal_sharpe:.3f}")
    logger.info(f"📉 Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"📊 Generalization: {(test_metrics['sharpe_ratio'] / train_optimal_sharpe) * 100:.1f}%\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
