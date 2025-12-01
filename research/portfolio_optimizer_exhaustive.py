#!/usr/bin/env python3
"""
Exhaustive Portfolio Optimizer

Tests ALL possible portfolio combinations to find the mathematically optimal
portfolio that maximizes Sharpe ratio while meeting drawdown constraints.

With 12 instruments and max 5 positions: 1,585 combinations (trivial to compute).

Usage:
    python research/portfolio_optimizer_exhaustive.py \
        --results-dir results \
        --train-start 2021-01-01 --train-end 2022-12-31 \
        --test-start 2023-01-01 --test-end 2024-12-31 \
        --max-positions 5 --max-dd-limit 5000 \
        --update-config
"""

import argparse
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from itertools import combinations
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass

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


@dataclass
class PortfolioResult:
    """Result from evaluating a portfolio combination."""
    symbols: Tuple[str, ...]
    max_positions: int
    sharpe: float
    max_dd: float
    cagr: float
    total_return: float
    profit_factor: float
    total_trades: int
    daily_stops: int
    avg_positions: float
    meets_constraint: bool


def filter_trades_by_date(
    trades_df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Filter trades to only include those within date range."""
    trades_df = trades_df.copy()
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
    symbols: Tuple[str, ...],
    max_positions: int,
    max_dd_limit: float,
    initial_capital: float,
    daily_stop_loss: float
) -> PortfolioResult:
    """Evaluate a specific portfolio combination."""
    filtered_trades = {s: symbol_trades[s] for s in symbols if s in symbol_trades}
    filtered_metadata = {s: symbol_metadata[s] for s in symbols if s in symbol_metadata}

    if not filtered_trades:
        return PortfolioResult(
            symbols=symbols,
            max_positions=max_positions,
            sharpe=0, max_dd=0, cagr=0, total_return=0,
            profit_factor=0, total_trades=0, daily_stops=0,
            avg_positions=0, meets_constraint=False
        )

    # Count total trades across all symbols
    total_trades = sum(len(df) for df in filtered_trades.values())

    equity_df, metrics = simulate_portfolio_intraday(
        symbol_trades=filtered_trades,
        symbol_metadata=filtered_metadata,
        max_positions=max_positions,
        initial_capital=initial_capital,
        daily_stop_loss=daily_stop_loss
    )

    max_dd = abs(metrics['max_drawdown_dollars'])

    return PortfolioResult(
        symbols=symbols,
        max_positions=max_positions,
        sharpe=metrics['sharpe_ratio'],
        max_dd=max_dd,
        cagr=metrics['cagr'],
        total_return=metrics['total_return_dollars'],
        profit_factor=metrics['profit_factor'],
        total_trades=total_trades,
        daily_stops=metrics['daily_stops_hit'],
        avg_positions=metrics['avg_positions'],
        meets_constraint=max_dd < max_dd_limit
    )


def generate_all_combinations(
    symbols: List[str],
    min_size: int,
    max_size: int
) -> List[Tuple[str, ...]]:
    """Generate all possible symbol combinations."""
    all_combos = []
    for size in range(min_size, max_size + 1):
        for combo in combinations(symbols, size):
            all_combos.append(combo)
    return all_combos


def update_config_yml(
    config_path: Path,
    symbols: List[str],
    max_positions: int,
    daily_stop_loss: float
) -> bool:
    """Update config.yml with optimal portfolio settings."""
    try:
        # Create backup
        backup_path = config_path.parent / f"{config_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update portfolio section
        if 'portfolio' not in config:
            config['portfolio'] = {}

        config['portfolio']['max_positions'] = max_positions
        config['portfolio']['daily_stop_loss'] = daily_stop_loss
        config['portfolio']['instruments'] = sorted(symbols)

        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated {config_path}")
        return True

    except Exception as exc:
        logger.error(f"Failed to update config.yml: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Exhaustive portfolio optimizer - tests ALL combinations'
    )

    # Date ranges
    parser.add_argument('--train-start', type=str, required=True)
    parser.add_argument('--train-end', type=str, required=True)
    parser.add_argument('--test-start', type=str, required=True)
    parser.add_argument('--test-end', type=str, required=True)

    # Parameters
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--max-dd-limit', type=float, default=5000.0)
    parser.add_argument('--initial-capital', type=float, default=150000.0)
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)

    # Config update
    parser.add_argument('--update-config', action='store_true',
                       help='Update config.yml with winning portfolio')
    parser.add_argument('--config-path', type=str, default='config.yml')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    logger.info("=" * 80)
    logger.info("EXHAUSTIVE PORTFOLIO OPTIMIZER")
    logger.info("=" * 80)
    logger.info(f"Train: {args.train_start} to {args.train_end}")
    logger.info(f"Test:  {args.test_start} to {args.test_end}")
    logger.info(f"Constraint: Max Drawdown < ${args.max_dd_limit:,.0f}")
    logger.info(f"Daily Stop Loss: ${args.daily_stop_loss:,.0f}")
    logger.info(f"Position range: {args.min_positions} to {args.max_positions}")
    logger.info("")

    # Discover and load symbols
    available_symbols = discover_available_symbols(results_dir)

    if 'PL' in available_symbols:
        available_symbols.remove('PL')
        logger.info("Excluded PL (Platinum)")

    logger.info(f"Loading {len(available_symbols)} symbols...")

    all_symbol_trades = {}
    all_symbol_metadata = {}

    for symbol in available_symbols:
        try:
            trades_df, metadata = load_symbol_trades(results_dir, symbol)
            all_symbol_trades[symbol] = trades_df
            all_symbol_metadata[symbol] = metadata
        except Exception as e:
            logger.warning(f"  {symbol}: {e}")

    logger.info(f"Loaded {len(all_symbol_trades)} symbols\n")

    # Filter to train period
    train_symbol_trades = {}
    for symbol, trades_df in all_symbol_trades.items():
        filtered = filter_trades_by_date(trades_df, args.train_start, args.train_end)
        if len(filtered) > 0:
            train_symbol_trades[symbol] = filtered

    symbols = list(train_symbol_trades.keys())
    logger.info(f"Symbols with train data: {len(symbols)}")
    logger.info(f"Symbols: {', '.join(sorted(symbols))}\n")

    # Generate all combinations
    all_combos = generate_all_combinations(
        symbols,
        args.min_positions,
        args.max_positions
    )

    logger.info(f"Testing {len(all_combos)} portfolio combinations...")
    logger.info("=" * 80)

    # Evaluate all combinations
    results: List[PortfolioResult] = []

    for i, combo in enumerate(all_combos):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(all_combos)} ({100*(i+1)/len(all_combos):.1f}%)")

        # Test with max_positions = size of combo (use all selected instruments)
        max_pos = min(len(combo), args.max_positions)

        result = evaluate_portfolio(
            symbol_trades=train_symbol_trades,
            symbol_metadata=all_symbol_metadata,
            symbols=combo,
            max_positions=max_pos,
            max_dd_limit=args.max_dd_limit,
            initial_capital=args.initial_capital,
            daily_stop_loss=args.daily_stop_loss
        )
        results.append(result)

    logger.info(f"  Progress: {len(all_combos)}/{len(all_combos)} (100%)\n")

    # Filter to valid portfolios
    valid_results = [r for r in results if r.meets_constraint and r.sharpe > 0]

    logger.info("=" * 80)
    logger.info("TRAIN PERIOD RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total combinations tested: {len(all_combos)}")
    logger.info(f"Valid portfolios (DD < ${args.max_dd_limit:,.0f}): {len(valid_results)}")

    if not valid_results:
        logger.error("No portfolios met the constraint!")
        return 1

    # Sort by Sharpe ratio
    valid_results.sort(key=lambda x: x.sharpe, reverse=True)

    # Show top 10
    logger.info("\nTop 10 portfolios by Sharpe (train period):")
    logger.info("-" * 80)
    logger.info(f"{'Rank':<5} {'Sharpe':<8} {'MaxDD':<10} {'PF':<6} {'Trades':<8} {'Instruments'}")
    logger.info("-" * 80)

    for i, r in enumerate(valid_results[:10]):
        instruments = ', '.join(sorted(r.symbols))
        logger.info(f"{i+1:<5} {r.sharpe:<8.3f} ${r.max_dd:<9,.0f} {r.profit_factor:<6.2f} {r.total_trades:<8} {instruments}")

    # Best portfolio
    best = valid_results[0]

    logger.info("\n" + "=" * 80)
    logger.info("BEST PORTFOLIO (TRAIN)")
    logger.info("=" * 80)
    logger.info(f"Instruments ({len(best.symbols)}): {', '.join(sorted(best.symbols))}")
    logger.info(f"Max Positions: {best.max_positions}")
    logger.info(f"Sharpe Ratio: {best.sharpe:.3f}")
    logger.info(f"Max Drawdown: ${best.max_dd:,.0f}")
    logger.info(f"CAGR: {best.cagr*100:.2f}%")
    logger.info(f"Total Return: ${best.total_return:,.0f}")
    logger.info(f"Profit Factor: {best.profit_factor:.2f}")
    logger.info(f"Total Trades: {best.total_trades}")
    logger.info(f"Daily Stops Hit: {best.daily_stops}")

    # Validate on test period
    logger.info("\n" + "=" * 80)
    logger.info("TEST PERIOD VALIDATION")
    logger.info("=" * 80)

    test_symbol_trades = {}
    for symbol in best.symbols:
        if symbol in all_symbol_trades:
            filtered = filter_trades_by_date(
                all_symbol_trades[symbol],
                args.test_start,
                args.test_end
            )
            if len(filtered) > 0:
                test_symbol_trades[symbol] = filtered

    test_result = evaluate_portfolio(
        symbol_trades=test_symbol_trades,
        symbol_metadata=all_symbol_metadata,
        symbols=best.symbols,
        max_positions=best.max_positions,
        max_dd_limit=args.max_dd_limit,
        initial_capital=args.initial_capital,
        daily_stop_loss=args.daily_stop_loss
    )

    logger.info(f"Sharpe Ratio: {test_result.sharpe:.3f}")
    logger.info(f"Max Drawdown: ${test_result.max_dd:,.0f}")
    logger.info(f"CAGR: {test_result.cagr*100:.2f}%")
    logger.info(f"Total Return: ${test_result.total_return:,.0f}")
    logger.info(f"Profit Factor: {test_result.profit_factor:.2f}")
    logger.info(f"Total Trades: {test_result.total_trades}")
    logger.info(f"Daily Stops Hit: {test_result.daily_stops}")

    # Generalization ratio
    if best.sharpe > 0:
        gen_ratio = test_result.sharpe / best.sharpe
        logger.info(f"\nGeneralization: {gen_ratio*100:.1f}% (test Sharpe / train Sharpe)")
        if gen_ratio > 0.8:
            logger.info("EXCELLENT - portfolio generalizes well")
        elif gen_ratio > 0.5:
            logger.info("MODERATE - some overfitting detected")
        else:
            logger.info("POOR - significant overfitting")

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN vs TEST COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    logger.info("-" * 50)
    logger.info(f"{'Sharpe Ratio':<20} {best.sharpe:<15.3f} {test_result.sharpe:<15.3f}")
    logger.info(f"{'Max Drawdown':<20} ${best.max_dd:<14,.0f} ${test_result.max_dd:<14,.0f}")
    logger.info(f"{'CAGR':<20} {best.cagr*100:<14.2f}% {test_result.cagr*100:<14.2f}%")
    logger.info(f"{'Profit Factor':<20} {best.profit_factor:<15.2f} {test_result.profit_factor:<15.2f}")
    logger.info(f"{'Total Trades':<20} {best.total_trades:<15} {test_result.total_trades:<15}")

    # Update config.yml if requested
    if args.update_config:
        logger.info("\n" + "=" * 80)
        logger.info("UPDATING CONFIG.YML")
        logger.info("=" * 80)

        config_path = Path(args.config_path)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
        else:
            success = update_config_yml(
                config_path=config_path,
                symbols=list(best.symbols),
                max_positions=best.max_positions,
                daily_stop_loss=args.daily_stop_loss
            )
            if success:
                logger.info(f"\nconfig.yml updated with:")
                logger.info(f"  portfolio:")
                logger.info(f"    max_positions: {best.max_positions}")
                logger.info(f"    daily_stop_loss: {args.daily_stop_loss}")
                logger.info(f"    instruments: {sorted(best.symbols)}")
                logger.info("\nTo deploy:")
                logger.info("  git add config.yml && git commit -m 'Update portfolio config'")
                logger.info("  git push && sudo systemctl restart pine-runner.service")

    # Save results
    output_file = results_dir / f"exhaustive_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'train_period': {'start': args.train_start, 'end': args.train_end},
        'test_period': {'start': args.test_start, 'end': args.test_end},
        'constraint': f'Max DD < ${args.max_dd_limit:,.0f}',
        'daily_stop_loss': args.daily_stop_loss,
        'combinations_tested': len(all_combos),
        'valid_portfolios': len(valid_results),
        'best_portfolio': {
            'instruments': sorted(best.symbols),
            'max_positions': best.max_positions,
            'train_sharpe': best.sharpe,
            'train_max_dd': best.max_dd,
            'train_pf': best.profit_factor,
            'test_sharpe': test_result.sharpe,
            'test_max_dd': test_result.max_dd,
            'test_pf': test_result.profit_factor,
            'generalization': test_result.sharpe / best.sharpe if best.sharpe > 0 else 0
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
