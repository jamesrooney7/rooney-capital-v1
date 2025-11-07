#!/usr/bin/env python3
"""
Greedy Portfolio Optimizer with Train/Test Split

Uses existing trade data and greedy instrument removal to find optimal portfolio:
1. Loads trades from results/ directory
2. Filters by date for train/test split
3. Greedy optimization on train: removes instruments until constraints met
4. Tests multiple max_positions to find best Sharpe with DD < $5000
5. Validates winning configuration on test period

Much faster than re-running backtests!

Usage:
    python research/portfolio_optimizer_greedy_train_test.py \
        --results-dir results \
        --train-start 2019-01-01 --train-end 2021-12-31 \
        --test-start 2022-01-01 --test-end 2024-12-31 \
        --min-positions 1 --max-positions 10 \
        --max-dd-limit 5000
"""

import argparse
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from copy import deepcopy

# Import from portfolio_simulator
from portfolio_simulator import (
    discover_available_symbols,
    load_symbol_trades,
    simulate_portfolio_intraday
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
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    filtered = trades_df[
        (trades_df['entry_time'] >= start) &
        (trades_df['entry_time'] <= end)
    ].copy()

    return filtered


def evaluate_portfolio(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    symbols_to_include: Set[str],
    max_positions: int,
    initial_capital: float = 150000.0,
    daily_stop_loss: float = 2500.0
) -> Dict:
    """Run portfolio simulation for a specific symbol set."""
    filtered_trades = {s: symbol_trades[s] for s in symbols_to_include if s in symbol_trades}
    filtered_metadata = {s: symbol_metadata[s] for s in symbols_to_include if s in symbol_metadata}

    if not filtered_trades:
        return {'sharpe_ratio': 0, 'max_drawdown_dollars': 0}

    equity_df, metrics = simulate_portfolio_intraday(
        symbol_trades=filtered_trades,
        symbol_metadata=filtered_metadata,
        max_positions=max_positions,
        initial_capital=initial_capital,
        daily_stop_loss=daily_stop_loss
    )

    return metrics


def identify_worst_symbol(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    current_symbols: Set[str],
    max_positions: int,
    initial_capital: float,
    daily_stop_loss: float
) -> str:
    """Identify which symbol contributes most to drawdown."""
    baseline_metrics = evaluate_portfolio(
        symbol_trades, symbol_metadata, current_symbols,
        max_positions, initial_capital, daily_stop_loss
    )
    baseline_dd = abs(baseline_metrics['max_drawdown_dollars'])

    logger.info(f"    Baseline DD: ${baseline_dd:,.0f} with {len(current_symbols)} symbols")

    best_symbol_to_remove = None
    best_dd_improvement = 0

    for symbol in current_symbols:
        test_symbols = current_symbols - {symbol}
        if len(test_symbols) == 0:
            continue

        test_metrics = evaluate_portfolio(
            symbol_trades, symbol_metadata, test_symbols,
            max_positions, initial_capital, daily_stop_loss
        )
        test_dd = abs(test_metrics['max_drawdown_dollars'])
        improvement = baseline_dd - test_dd

        if improvement > best_dd_improvement:
            best_dd_improvement = improvement
            best_symbol_to_remove = symbol

    if best_symbol_to_remove:
        logger.info(f"    → Removing {best_symbol_to_remove} (DD improvement: ${best_dd_improvement:,.0f})")
    else:
        # No improvement found, remove lowest Sharpe
        sharpe_dict = {s: symbol_metadata.get(s, {}).get('Sharpe_OOS_CPCV', 0) for s in current_symbols}
        best_symbol_to_remove = min(sharpe_dict.keys(), key=lambda s: sharpe_dict[s])
        logger.info(f"    → Removing {best_symbol_to_remove} (lowest Sharpe)")

    return best_symbol_to_remove


def greedy_optimize(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    max_positions: int,
    max_dd_limit: float,
    initial_capital: float,
    daily_stop_loss: float
) -> Tuple[Set[str], Dict]:
    """Greedy optimization: iteratively remove worst symbols until DD constraint met."""
    current_symbols = set(symbol_trades.keys())
    best_valid_config = None

    logger.info(f"  Greedy optimization for max_positions={max_positions}")
    logger.info(f"  Starting with {len(current_symbols)} symbols")
    logger.info(f"  Target: DD < ${max_dd_limit:,.0f}\n")

    iteration = 0

    while len(current_symbols) >= 1:
        iteration += 1

        # Evaluate current portfolio
        metrics = evaluate_portfolio(
            symbol_trades, symbol_metadata, current_symbols,
            max_positions, initial_capital, daily_stop_loss
        )

        max_dd = abs(metrics['max_drawdown_dollars'])
        sharpe = metrics['sharpe_ratio']

        logger.info(f"  Iteration {iteration}: {len(current_symbols)} symbols | "
                   f"Sharpe={sharpe:.3f} | DD=${max_dd:,.0f}")

        # Check if constraint met
        if max_dd < max_dd_limit:
            logger.info(f"  ✅ CONSTRAINT MET!")
            if best_valid_config is None or sharpe > best_valid_config['sharpe']:
                best_valid_config = {
                    'symbols': current_symbols.copy(),
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'metrics': metrics
                }
                logger.info(f"  🏆 New best: Sharpe={sharpe:.3f}\n")
            break

        # Stop if only 1 symbol left
        if len(current_symbols) <= 1:
            logger.info(f"  ⚠️  Only 1 symbol left, stopping\n")
            break

        # Remove worst symbol
        worst_symbol = identify_worst_symbol(
            symbol_trades, symbol_metadata, current_symbols,
            max_positions, initial_capital, daily_stop_loss
        )
        current_symbols.remove(worst_symbol)

    if best_valid_config:
        return best_valid_config['symbols'], best_valid_config['metrics']
    else:
        # Return last tested config even if doesn't meet constraint
        logger.warning(f"  ⚠️  Could not meet DD constraint with any combination")
        # If no symbols at all, return empty metrics
        if len(current_symbols) == 0:
            return set(), {'sharpe_ratio': 0, 'cagr': 0, 'total_return_dollars': 0,
                          'max_drawdown_dollars': 0, 'profit_factor': 0,
                          'daily_stops_hit': 0, 'avg_positions': 0}
        return current_symbols, metrics


def print_comparison_summary(
    train_result: Dict,
    test_metrics: Dict,
    train_period: str,
    test_period: str
):
    """Print side-by-side comparison."""
    logger.info(f"\n{'='*100}")
    logger.info(f"TRAIN vs TEST COMPARISON")
    logger.info(f"{'='*100}\n")

    comparison_data = {
        'Metric': [
            'Period',
            'Max Positions',
            'Instruments Used',
            'Sharpe Ratio',
            'CAGR',
            'Total Return $',
            'Max Drawdown $',
            'Profit Factor',
            'Daily Stops Hit',
            'Avg Positions'
        ],
        'Train (In-Sample)': [
            train_period,
            f"{train_result['max_positions']}",
            f"{train_result['n_symbols']}",
            f"{train_result['sharpe']:.3f}",
            f"{train_result.get('cagr', 0)*100:.2f}%",
            f"${train_result.get('total_return', 0):,.0f}",
            f"${train_result['max_dd']:,.0f}",
            f"{train_result.get('profit_factor', 0):.2f}",
            f"{train_result.get('daily_stops', 0):.0f}",
            f"{train_result.get('avg_positions', 0):.2f}"
        ],
        'Test (Out-of-Sample)': [
            test_period,
            f"{train_result['max_positions']}",
            f"{train_result['n_symbols']}",
            f"{test_metrics['sharpe_ratio']:.3f}",
            f"{test_metrics['cagr']*100:.2f}%",
            f"${test_metrics['total_return_dollars']:,.0f}",
            f"${abs(test_metrics['max_drawdown_dollars']):,.0f}",
            f"{test_metrics['profit_factor']:.2f}",
            f"{test_metrics['daily_stops_hit']:.0f}",
            f"{test_metrics['avg_positions']:.2f}"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("=" * 100)

    # Assessment
    sharpe_ratio = test_metrics['sharpe_ratio'] / train_result['sharpe']

    logger.info(f"\n📊 PERFORMANCE ASSESSMENT:")
    if sharpe_ratio > 0.9:
        logger.info(f"   ✅ EXCELLENT: Test Sharpe is {sharpe_ratio*100:.1f}% of train")
    elif sharpe_ratio > 0.7:
        logger.info(f"   ⚠️  MODERATE: Test Sharpe is {sharpe_ratio*100:.1f}% of train")
    else:
        logger.info(f"   ❌ POOR: Test Sharpe is {sharpe_ratio*100:.1f}% of train")
    logger.info("")


def update_config_yml(
    config_path: Path,
    symbols: List[str],
    max_positions: int,
    daily_stop_loss: float
) -> bool:
    """
    Update config.yml with optimal portfolio settings.
    Creates backup before updating.

    Returns True if successful, False otherwise.
    """
    try:
        # Create backup
        backup_path = config_path.parent / f"{config_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
        shutil.copy2(config_path, backup_path)
        logger.info(f"📋 Created backup: {backup_path}")

        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Don't modify symbols list (it should contain ALL symbols for feature calculation)
        # Instead, set portfolio.instruments to specify which ones to trade

        # Update or create portfolio section
        if 'portfolio' not in config:
            config['portfolio'] = {}

        config['portfolio']['max_positions'] = max_positions
        config['portfolio']['daily_stop_loss'] = daily_stop_loss
        config['portfolio']['instruments'] = sorted(symbols)  # Only trade these

        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"✅ Successfully updated {config_path}")
        logger.info(f"   - Traded instruments: {symbols}")
        logger.info(f"   - Max positions: {max_positions}")
        logger.info(f"   - Daily stop loss: ${daily_stop_loss:,.0f}")
        logger.info(f"   Note: symbols list unchanged (includes all for feature calculation)")

        return True

    except Exception as exc:
        logger.error(f"❌ Failed to update config.yml: {exc}")
        logger.exception(exc)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Greedy portfolio optimizer with train/test split'
    )

    # Date ranges
    parser.add_argument('--train-start', type=str, required=True)
    parser.add_argument('--train-end', type=str, required=True)
    parser.add_argument('--test-start', type=str, required=True)
    parser.add_argument('--test-end', type=str, required=True)

    # Parameters
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--max-dd-limit', type=float, default=5000.0,
                       help='Maximum drawdown limit in dollars')
    parser.add_argument('--initial-capital', type=float, default=150000.0,
                       help='Initial capital (default $150k to match live account)')
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--update-config', action='store_true',
                       help='Automatically update config.yml with optimal settings')
    parser.add_argument('--config-path', type=str, default='config.yml',
                       help='Path to config.yml file to update (default: config.yml)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    logger.info("="*100)
    logger.info("GREEDY PORTFOLIO OPTIMIZER WITH TRAIN/TEST SPLIT")
    logger.info("="*100)
    logger.info(f"Train Period: {args.train_start} to {args.train_end}")
    logger.info(f"Test Period: {args.test_start} to {args.test_end}")
    logger.info(f"Constraint: Max Drawdown < ${args.max_dd_limit:,.0f}")
    logger.info(f"Testing max_positions: {args.min_positions} to {args.max_positions}")
    logger.info("")

    # Load symbols
    available_symbols = discover_available_symbols(results_dir)

    if 'PL' in available_symbols:
        available_symbols.remove('PL')
        logger.info("✂️  Excluded PL (Platinum)")

    logger.info(f"📦 Loading {len(available_symbols)} symbols\n")

    # Load all trade data
    all_symbol_trades = {}
    all_symbol_metadata = {}

    for symbol in available_symbols:
        try:
            trades_df, metadata = load_symbol_trades(results_dir, symbol)
            all_symbol_trades[symbol] = trades_df
            all_symbol_metadata[symbol] = metadata
        except Exception as e:
            logger.warning(f"  ❌ {symbol}: {e}")

    logger.info(f"✅ Loaded {len(all_symbol_trades)} symbols\n")

    # ======== TRAIN PERIOD ========
    logger.info(f"{'#'*100}")
    logger.info(f"# TRAIN PERIOD: {args.train_start} to {args.train_end}")
    logger.info(f"{'#'*100}\n")

    train_symbol_trades = {}
    for symbol, trades_df in all_symbol_trades.items():
        filtered = filter_trades_by_date(trades_df, args.train_start, args.train_end)
        if len(filtered) > 0:
            train_symbol_trades[symbol] = filtered

    logger.info(f"Train period: {len(train_symbol_trades)} symbols with trades\n")

    if len(train_symbol_trades) == 0:
        logger.error(f"❌ NO TRADES FOUND IN TRAIN PERIOD: {args.train_start} to {args.train_end}")
        logger.info("\nYour trade data date range:")
        for symbol in list(all_symbol_trades.keys())[:3]:  # Sample 3 symbols
            trades = all_symbol_trades[symbol]
            first_date = trades['entry_time'].min()
            last_date = trades['entry_time'].max()
            logger.info(f"  {symbol}: {first_date} to {last_date}")
        logger.info("\n💡 Your data likely only covers 2023-2024.")
        logger.info("   Try: --train-start 2023-01-01 --train-end 2023-12-31")
        logger.info("        --test-start 2024-01-01 --test-end 2024-12-31")
        return 1

    # Run greedy optimization for each max_positions
    train_results = []

    for max_pos in range(args.min_positions, args.max_positions + 1):
        logger.info(f"{'='*100}")
        logger.info(f"TESTING max_positions = {max_pos}")
        logger.info(f"{'='*100}\n")

        best_symbols, metrics = greedy_optimize(
            symbol_trades=train_symbol_trades,
            symbol_metadata=all_symbol_metadata,
            max_positions=max_pos,
            max_dd_limit=args.max_dd_limit,
            initial_capital=args.initial_capital,
            daily_stop_loss=args.daily_stop_loss
        )

        result = {
            'max_positions': max_pos,
            'n_symbols': len(best_symbols),
            'symbols': sorted(best_symbols),
            'sharpe': metrics['sharpe_ratio'],
            'cagr': metrics['cagr'],
            'total_return': metrics['total_return_dollars'],
            'max_dd': abs(metrics['max_drawdown_dollars']),
            'profit_factor': metrics['profit_factor'],
            'daily_stops': metrics['daily_stops_hit'],
            'avg_positions': metrics['avg_positions']
        }
        train_results.append(result)

    # Find best configuration that meets constraint
    valid_configs = [r for r in train_results if r['max_dd'] < args.max_dd_limit]

    if not valid_configs:
        logger.error(f"\n❌ NO CONFIGURATIONS MET CONSTRAINT: DD < ${args.max_dd_limit:,.0f}")
        logger.info("\nAll tested configurations:")
        for r in sorted(train_results, key=lambda x: x['sharpe'], reverse=True):
            logger.info(f"  max_positions={r['max_positions']}, symbols={r['n_symbols']}: "
                       f"Sharpe={r['sharpe']:.3f}, DD=${r['max_dd']:,.0f}")
        logger.info("\n💡 Try relaxing --max-dd-limit or using fewer --max-positions")
        return 1

    # Get best valid config
    best_train = sorted(valid_configs, key=lambda x: x['sharpe'], reverse=True)[0]

    logger.info(f"\n{'='*100}")
    logger.info(f"TRAIN PERIOD: BEST VALID CONFIGURATION")
    logger.info(f"{'='*100}\n")
    logger.info(f"  Max Positions: {best_train['max_positions']}")
    logger.info(f"  Instruments ({best_train['n_symbols']}): {', '.join(best_train['symbols'])}")
    logger.info(f"  Sharpe Ratio: {best_train['sharpe']:.3f}")
    logger.info(f"  Max Drawdown: ${best_train['max_dd']:,.0f} ✅")
    logger.info(f"  CAGR: {best_train['cagr']*100:.2f}%")
    logger.info(f"  Profit Factor: {best_train['profit_factor']:.2f}\n")

    # ======== TEST PERIOD ========
    logger.info(f"{'#'*100}")
    logger.info(f"# TEST PERIOD: {args.test_start} to {args.test_end}")
    logger.info(f"{'#'*100}\n")

    test_symbol_trades = {}
    for symbol in best_train['symbols']:
        if symbol in all_symbol_trades:
            filtered = filter_trades_by_date(all_symbol_trades[symbol], args.test_start, args.test_end)
            if len(filtered) > 0:
                test_symbol_trades[symbol] = filtered

    logger.info(f"Applying optimal config to test period...")
    logger.info(f"  Using {len(test_symbol_trades)} instruments: {', '.join(sorted(test_symbol_trades.keys()))}\n")

    test_metrics = evaluate_portfolio(
        symbol_trades=test_symbol_trades,
        symbol_metadata=all_symbol_metadata,
        symbols_to_include=set(test_symbol_trades.keys()),
        max_positions=best_train['max_positions'],
        initial_capital=args.initial_capital,
        daily_stop_loss=args.daily_stop_loss
    )

    logger.info(f"TEST PERIOD RESULTS:")
    logger.info(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  CAGR: {test_metrics['cagr']*100:.2f}%")
    logger.info(f"  Total Return: ${test_metrics['total_return_dollars']:,.0f}")
    logger.info(f"  Max Drawdown: ${abs(test_metrics['max_drawdown_dollars']):,.0f}")
    logger.info(f"  Profit Factor: {test_metrics['profit_factor']:.2f}\n")

    # Comparison
    print_comparison_summary(
        best_train,
        test_metrics,
        f"{args.train_start} to {args.train_end}",
        f"{args.test_start} to {args.test_end}"
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary
    summary = {
        'train_period': {'start': args.train_start, 'end': args.train_end},
        'test_period': {'start': args.test_start, 'end': args.test_end},
        'constraint': f'Max DD < ${args.max_dd_limit:,.0f}',
        'optimal_config': {
            'max_positions': best_train['max_positions'],
            'instruments': best_train['symbols'],
            'n_instruments': best_train['n_symbols']
        },
        'train_metrics': {
            'sharpe': float(best_train['sharpe']),
            'max_dd': float(best_train['max_dd']),
            'cagr': float(best_train['cagr'])
        },
        'test_metrics': {
            'sharpe': float(test_metrics['sharpe_ratio']),
            'max_dd': float(abs(test_metrics['max_drawdown_dollars'])),
            'cagr': float(test_metrics['cagr'])
        },
        'generalization': float(test_metrics['sharpe_ratio'] / best_train['sharpe']),
        'timestamp': timestamp
    }

    summary_file = output_dir / f'greedy_optimization_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"💾 Summary saved to: {summary_file}\n")

    # Update config.yml if requested
    if args.update_config:
        logger.info("=" * 100)
        logger.info("UPDATING CONFIG.YML")
        logger.info("=" * 100)

        config_path = Path(args.config_path)
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info("   Skipping config update. Run optimizer again with correct --config-path")
        else:
            success = update_config_yml(
                config_path=config_path,
                symbols=best_train['symbols'],
                max_positions=best_train['max_positions'],
                daily_stop_loss=args.daily_stop_loss
            )
            if success:
                logger.info("\n🎉 Config updated! You can now deploy with:")
                logger.info("   git add config.yml && git commit -m 'Update portfolio config'")
                logger.info("   git push && sudo systemctl restart pine-runner.service")
            else:
                logger.error("\n⚠️  Config update failed. Please update manually.")
        logger.info("")

    logger.info(f"{'='*100}")
    logger.info(f"✅ OPTIMIZATION COMPLETE")
    logger.info(f"{'='*100}\n")
    logger.info(f"🏆 Optimal: max_positions={best_train['max_positions']}, "
               f"{best_train['n_symbols']} instruments")
    logger.info(f"📈 Train Sharpe: {best_train['sharpe']:.3f} (DD: ${best_train['max_dd']:,.0f})")
    logger.info(f"📉 Test Sharpe: {test_metrics['sharpe_ratio']:.3f} "
               f"(DD: ${abs(test_metrics['max_drawdown_dollars']):,.0f})")
    logger.info(f"📊 Generalization: {(test_metrics['sharpe_ratio'] / best_train['sharpe']) * 100:.1f}%\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
