#!/usr/bin/env python3
"""
Portfolio Optimizer with Train/Test Split

This script:
1. Trains on period 1 (e.g., 2019-2021) to find optimal max_positions
2. Validates on period 2 (e.g., 2022-2024) using the optimal configuration
3. Shows detailed results and per-instrument statistics for both periods

Usage:
    python research/portfolio_optimizer_train_test.py \
        --train-start 2019-01-01 --train-end 2021-12-31 \
        --test-start 2022-01-01 --test-end 2024-12-31 \
        --min-positions 1 --max-positions 10
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from portfolio_optimizer_full import (
    run_backtest_with_signals,
    combine_portfolio_signals,
    simulate_portfolio_returns,
    optimize_portfolio
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_symbol_metrics(signal_df: pd.DataFrame, symbol: str) -> Dict:
    """Calculate detailed metrics for a single symbol."""
    total_signals = signal_df['signal'].sum()

    if total_signals == 0:
        return {
            'symbol': symbol,
            'total_signals': 0,
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'total_return_pct': 0
        }

    # Get returns only when signals are generated
    signal_returns = signal_df.loc[signal_df['signal'] == True, 'returns'].dropna()

    if len(signal_returns) == 0:
        return {
            'symbol': symbol,
            'total_signals': int(total_signals),
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'total_return_pct': 0
        }

    # Calculate metrics
    win_rate = (signal_returns > 0).sum() / len(signal_returns) if len(signal_returns) > 0 else 0
    avg_return = signal_returns.mean()
    std_return = signal_returns.std()
    sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
    total_return_pct = (1 + signal_returns).prod() - 1

    return {
        'symbol': symbol,
        'total_signals': int(total_signals),
        'trades_executed': len(signal_returns),
        'win_rate': win_rate,
        'avg_return_pct': avg_return * 100,
        'sharpe': sharpe,
        'total_return_pct': total_return_pct * 100
    }


def print_symbol_statistics(signals_dict: Dict[str, pd.DataFrame], period_name: str):
    """Print detailed per-symbol statistics."""
    logger.info(f"\n{'='*90}")
    logger.info(f"PER-INSTRUMENT STATISTICS - {period_name}")
    logger.info(f"{'='*90}\n")

    # Calculate metrics for each symbol
    symbol_metrics = []
    for symbol, signal_df in signals_dict.items():
        metrics = calculate_symbol_metrics(signal_df, symbol)
        symbol_metrics.append(metrics)

    # Sort by Sharpe
    symbol_metrics_df = pd.DataFrame(symbol_metrics)
    symbol_metrics_df = symbol_metrics_df.sort_values('sharpe', ascending=False)

    # Print table
    print(f"{'Symbol':<8}{'Signals':<10}{'Trades':<10}{'Win%':<10}{'Avg Ret%':<12}{'Sharpe':<10}{'Total Ret%':<12}")
    print("-" * 90)

    for _, row in symbol_metrics_df.iterrows():
        print(f"{row['symbol']:<8}"
              f"{row['total_signals']:<10.0f}"
              f"{row.get('trades_executed', 0):<10.0f}"
              f"{row['win_rate']*100:<10.1f}"
              f"{row['avg_return_pct']:<12.3f}"
              f"{row['sharpe']:<10.2f}"
              f"{row['total_return_pct']:<12.1f}")

    print("=" * 90)

    # Summary statistics
    logger.info(f"\nSummary:")
    logger.info(f"  Total Instruments: {len(symbol_metrics)}")
    logger.info(f"  Avg Win Rate: {symbol_metrics_df['win_rate'].mean()*100:.1f}%")
    logger.info(f"  Avg Sharpe: {symbol_metrics_df['sharpe'].mean():.2f}")
    logger.info(f"  Best Sharpe: {symbol_metrics_df['sharpe'].max():.2f} ({symbol_metrics_df.iloc[0]['symbol']})")
    logger.info(f"  Worst Sharpe: {symbol_metrics_df['sharpe'].min():.2f} ({symbol_metrics_df.iloc[-1]['symbol']})\n")

    return symbol_metrics_df


def run_optimization_with_validation(
    train_symbols: Dict[str, pd.DataFrame],
    test_symbols: Dict[str, pd.DataFrame],
    min_positions: int,
    max_positions: int,
    initial_capital: float,
    daily_stop_loss: float,
    commission_pct: float
):
    """
    Run optimization on train set, validate on test set.

    Returns:
        train_results_df: Full optimization results on train
        test_metrics: Single result using optimal config on test
        optimal_max_positions: Best max_positions from train
    """

    # ======== TRAIN PERIOD ========
    logger.info(f"\n{'#'*90}")
    logger.info(f"# TRAIN PERIOD OPTIMIZATION")
    logger.info(f"{'#'*90}\n")

    # Print per-symbol stats for train
    train_symbol_stats = print_symbol_statistics(train_symbols, "TRAIN PERIOD")

    # Run optimization on train
    train_results_df = optimize_portfolio(
        train_symbols,
        min_positions=min_positions,
        max_positions=max_positions,
        commission_pct=commission_pct,
        initial_capital=initial_capital,
        daily_stop_loss=daily_stop_loss
    )

    # Get optimal configuration
    optimal_max_positions = int(train_results_df.iloc[0]['max_positions'])
    train_optimal_sharpe = train_results_df.iloc[0]['sharpe_ratio']

    logger.info(f"\n✅ TRAIN PERIOD: Optimal max_positions = {optimal_max_positions} (Sharpe: {train_optimal_sharpe:.3f})")

    # ======== TEST PERIOD ========
    logger.info(f"\n{'#'*90}")
    logger.info(f"# TEST PERIOD VALIDATION (Using optimal max_positions = {optimal_max_positions})")
    logger.info(f"{'#'*90}\n")

    # Print per-symbol stats for test
    test_symbol_stats = print_symbol_statistics(test_symbols, "TEST PERIOD")

    # Apply optimal configuration to test period
    logger.info(f"\nApplying optimal configuration (max_positions={optimal_max_positions}) to test period...\n")

    test_portfolio_signals = combine_portfolio_signals(
        test_symbols,
        max_positions=optimal_max_positions
    )

    test_equity, test_metrics = simulate_portfolio_returns(
        test_symbols,
        test_portfolio_signals,
        commission_pct=commission_pct,
        initial_capital=initial_capital,
        daily_stop_loss=daily_stop_loss
    )

    # Print test results
    logger.info(f"{'='*90}")
    logger.info(f"TEST PERIOD RESULTS (max_positions={optimal_max_positions})")
    logger.info(f"{'='*90}\n")
    logger.info(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  CAGR: {test_metrics['cagr']*100:.2f}%")
    logger.info(f"  Total Return: ${test_metrics['total_return_dollars']:,.2f} ({test_metrics['total_return']*100:.1f}%)")
    logger.info(f"  Max Drawdown: ${test_metrics['max_drawdown_dollars']:,.2f} ({test_metrics['max_drawdown']*100:.1f}%)")
    logger.info(f"  Profit Factor: {test_metrics['profit_factor']:.2f}")
    logger.info(f"  Daily Stops Hit: {test_metrics['daily_stops_hit']:.0f}\n")

    return train_results_df, test_metrics, optimal_max_positions, train_symbol_stats, test_symbol_stats


def print_comparison_summary(
    train_results_df: pd.DataFrame,
    test_metrics: Dict,
    optimal_max_positions: int,
    train_period: str,
    test_period: str
):
    """Print side-by-side comparison of train vs test."""
    logger.info(f"\n{'='*90}")
    logger.info(f"TRAIN vs TEST COMPARISON")
    logger.info(f"{'='*90}\n")

    # Get optimal train metrics
    train_optimal = train_results_df.iloc[0]

    comparison_data = {
        'Metric': [
            'Period',
            'Max Positions',
            'Sharpe Ratio',
            'CAGR',
            'Total Return $',
            'Total Return %',
            'Max Drawdown $',
            'Max Drawdown %',
            'Profit Factor',
            'Daily Stops Hit'
        ],
        'Train (In-Sample)': [
            train_period,
            f"{int(train_optimal['max_positions'])}",
            f"{train_optimal['sharpe_ratio']:.3f}",
            f"{train_optimal['cagr']*100:.2f}%",
            f"${train_optimal['total_return_dollars']:,.0f}",
            f"{train_optimal['total_return']*100:.1f}%",
            f"${train_optimal['max_drawdown_dollars']:,.0f}",
            f"{train_optimal['max_drawdown']*100:.1f}%",
            f"{train_optimal['profit_factor']:.2f}",
            f"{train_optimal['daily_stops_hit']:.0f}"
        ],
        'Test (Out-of-Sample)': [
            test_period,
            f"{optimal_max_positions}",
            f"{test_metrics['sharpe_ratio']:.3f}",
            f"{test_metrics['cagr']*100:.2f}%",
            f"${test_metrics['total_return_dollars']:,.0f}",
            f"{test_metrics['total_return']*100:.1f}%",
            f"${test_metrics['max_drawdown_dollars']:,.0f}",
            f"{test_metrics['max_drawdown']*100:.1f}%",
            f"{test_metrics['profit_factor']:.2f}",
            f"{test_metrics['daily_stops_hit']:.0f}"
        ],
        'Change': [
            '-',
            '-',
            f"{((test_metrics['sharpe_ratio'] / train_optimal['sharpe_ratio']) - 1) * 100:+.1f}%",
            f"{((test_metrics['cagr'] / train_optimal['cagr']) - 1) * 100:+.1f}%" if train_optimal['cagr'] != 0 else 'N/A',
            f"{test_metrics['total_return_dollars'] - train_optimal['total_return_dollars']:+,.0f}",
            f"{((test_metrics['total_return'] / train_optimal['total_return']) - 1) * 100:+.1f}%" if train_optimal['total_return'] != 0 else 'N/A',
            f"{test_metrics['max_drawdown_dollars'] - train_optimal['max_drawdown_dollars']:+,.0f}",
            f"{((test_metrics['max_drawdown'] / train_optimal['max_drawdown']) - 1) * 100:+.1f}%" if train_optimal['max_drawdown'] != 0 else 'N/A',
            f"{test_metrics['profit_factor'] - train_optimal['profit_factor']:+.2f}",
            f"{test_metrics['daily_stops_hit'] - train_optimal['daily_stops_hit']:+.0f}"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("=" * 90)

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
        description='Portfolio optimizer with train/test split'
    )

    # Date ranges
    parser.add_argument('--train-start', type=str, required=True, help='Train start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, required=True, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, required=True, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, required=True, help='Test end date (YYYY-MM-DD)')

    # Other parameters
    parser.add_argument('--symbols', nargs='+', help='Symbols to include')
    parser.add_argument('--data-dir', type=str, default='data/resampled')
    parser.add_argument('--models-dir', type=str, default='src/models')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--commission-pct', type=float, default=0.0001)
    parser.add_argument('--initial-cash', type=float, default=250000.0)
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML filtering')

    args = parser.parse_args()

    # Auto-discover symbols if not specified
    if args.symbols:
        symbols = args.symbols
    else:
        models_dir = Path(args.models_dir)
        symbols = [
            f.stem.replace('_best', '')
            for f in models_dir.glob('*_best.json')
            if (models_dir / f.name.replace('_best.json', '_rf_model.pkl')).exists()
        ]

        # Exclude PL (Platinum) due to excessive real-world slippage
        if 'PL' in symbols:
            symbols.remove('PL')
            logger.info("✂️  Excluded PL (Platinum) from optimization due to slippage")

        logger.info(f"📦 Auto-discovered {len(symbols)} symbols: {', '.join(sorted(symbols))}")

    if not symbols:
        logger.error("❌ No symbols to process")
        return 1

    # ======== RUN BACKTESTS ========
    logger.info(f"\n{'='*90}")
    logger.info(f"RUNNING BACKTESTS")
    logger.info(f"{'='*90}\n")
    logger.info(f"Train Period: {args.train_start} to {args.train_end}")
    logger.info(f"Test Period: {args.test_start} to {args.test_end}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"4-tick round-trip slippage applied")
    logger.info(f"")

    # Run backtests for train period
    logger.info(f"🔄 Running backtests for TRAIN period...")
    train_signals = {}
    for symbol in symbols:
        signal_df = run_backtest_with_signals(
            symbol,
            args.train_start,
            args.train_end,
            data_dir=args.data_dir,
            initial_cash=args.initial_cash,
            use_ml=not args.no_ml
        )
        if signal_df is not None:
            train_signals[symbol] = signal_df

    if not train_signals:
        logger.error("❌ No signals extracted from train period")
        return 1

    logger.info(f"✅ Train: Extracted signals from {len(train_signals)} symbols\n")

    # Run backtests for test period
    logger.info(f"🔄 Running backtests for TEST period...")
    test_signals = {}
    for symbol in symbols:
        signal_df = run_backtest_with_signals(
            symbol,
            args.test_start,
            args.test_end,
            data_dir=args.data_dir,
            initial_cash=args.initial_cash,
            use_ml=not args.no_ml
        )
        if signal_df is not None:
            test_signals[symbol] = signal_df

    if not test_signals:
        logger.error("❌ No signals extracted from test period")
        return 1

    logger.info(f"✅ Test: Extracted signals from {len(test_signals)} symbols\n")

    # ======== OPTIMIZE AND VALIDATE ========
    train_results_df, test_metrics, optimal_max_positions, train_symbol_stats, test_symbol_stats = run_optimization_with_validation(
        train_signals,
        test_signals,
        args.min_positions,
        args.max_positions,
        args.initial_cash,
        args.daily_stop_loss,
        args.commission_pct
    )

    # ======== PRINT COMPARISON ========
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
    train_symbol_file = output_dir / f'symbol_stats_train_{timestamp}.csv'
    train_symbol_stats.to_csv(train_symbol_file, index=False)
    logger.info(f"💾 Train symbol stats saved to: {train_symbol_file}")

    test_symbol_file = output_dir / f'symbol_stats_test_{timestamp}.csv'
    test_symbol_stats.to_csv(test_symbol_file, index=False)
    logger.info(f"💾 Test symbol stats saved to: {test_symbol_file}")

    # Save summary JSON
    summary = {
        'train_period': {'start': args.train_start, 'end': args.train_end},
        'test_period': {'start': args.test_start, 'end': args.test_end},
        'optimal_max_positions': optimal_max_positions,
        'train_sharpe': float(train_results_df.iloc[0]['sharpe_ratio']),
        'test_sharpe': float(test_metrics['sharpe_ratio']),
        'symbols': sorted(list(train_signals.keys())),
        'excluded_symbols': ['PL'],
        'slippage': '4-tick round-trip',
        'timestamp': timestamp
    }

    summary_file = output_dir / f'portfolio_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"💾 Summary saved to: {summary_file}\n")

    logger.info(f"{'='*90}")
    logger.info(f"✅ OPTIMIZATION COMPLETE")
    logger.info(f"{'='*90}\n")
    logger.info(f"🏆 Optimal Configuration: max_positions = {optimal_max_positions}")
    logger.info(f"📈 Train Sharpe: {train_results_df.iloc[0]['sharpe_ratio']:.3f}")
    logger.info(f"📉 Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"📊 Generalization: {(test_metrics['sharpe_ratio'] / train_results_df.iloc[0]['sharpe_ratio']) * 100:.1f}%\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
