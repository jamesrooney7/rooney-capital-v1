#!/usr/bin/env python3
"""
Greedy Portfolio Optimizer with Broker Constraint Satisfaction

Finds optimal symbol combinations and max_positions that satisfy:
- Max drawdown < $9,000
- Drawdown breach events <= 2
- Maximize Sharpe ratio

Uses greedy removal: iteratively removes symbols that contribute most to drawdowns
until constraints are satisfied.

WARNING: This optimizes on the full dataset - results will be optimistic.
Use paper trading to validate before going live.

Usage:
    python research/portfolio_optimizer_greedy.py \
        --results-dir results \
        --max-dd-limit 9000 \
        --max-breach-events 2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import logging
from copy import deepcopy

# Import from portfolio_simulator
from portfolio_simulator import (
    discover_available_symbols,
    load_symbol_trades,
    simulate_portfolio_intraday,
    calculate_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_portfolio(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    symbols_to_include: Set[str],
    max_positions: int,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> Dict:
    """Run portfolio simulation for a specific symbol set."""

    # Filter to only included symbols
    filtered_trades = {s: symbol_trades[s] for s in symbols_to_include}
    filtered_metadata = {s: symbol_metadata[s] for s in symbols_to_include}

    # Run simulation
    equity_df, position_counts, symbol_usage, stop_count = simulate_portfolio_intraday(
        symbol_trades=filtered_trades,
        symbol_metadata=filtered_metadata,
        max_positions=max_positions,
        initial_capital=initial_capital,
        daily_stop_loss=daily_stop_loss
    )

    # Calculate metrics
    metrics = calculate_metrics(
        equity_df=equity_df,
        position_counts=position_counts,
        symbol_usage=symbol_usage,
        stop_count=stop_count,
        initial_capital=initial_capital,
        n_symbols_total=len(symbols_to_include)
    )

    return metrics


def identify_worst_symbol(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    current_symbols: Set[str],
    max_positions: int,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> str:
    """
    Identify which symbol contributes most to drawdown.

    Strategy: Remove each symbol one at a time, see which removal
    improves drawdown the most.
    """

    baseline_metrics = evaluate_portfolio(
        symbol_trades, symbol_metadata, current_symbols,
        max_positions, initial_capital, daily_stop_loss
    )
    baseline_dd = abs(baseline_metrics['max_drawdown_dollars'])

    logger.info(f"  Baseline DD: ${baseline_dd:,.0f} with {len(current_symbols)} symbols")

    best_symbol_to_remove = None
    best_dd_improvement = 0

    for symbol in current_symbols:
        # Test portfolio without this symbol
        test_symbols = current_symbols - {symbol}
        if len(test_symbols) == 0:
            continue

        test_metrics = evaluate_portfolio(
            symbol_trades, symbol_metadata, test_symbols,
            max_positions, initial_capital, daily_stop_loss
        )
        test_dd = abs(test_metrics['max_drawdown_dollars'])
        improvement = baseline_dd - test_dd

        logger.debug(f"    Without {symbol}: DD=${test_dd:,.0f} (improvement: ${improvement:,.0f})")

        if improvement > best_dd_improvement:
            best_dd_improvement = improvement
            best_symbol_to_remove = symbol

    if best_symbol_to_remove:
        logger.info(f"  → Removing {best_symbol_to_remove} (DD improvement: ${best_dd_improvement:,.0f})")
    else:
        # No improvement found, just remove lowest Sharpe
        sharpe_dict = {s: symbol_metadata[s].get('Sharpe', 0) for s in current_symbols}
        best_symbol_to_remove = min(sharpe_dict.keys(), key=lambda s: sharpe_dict[s])
        logger.info(f"  → Removing {best_symbol_to_remove} (lowest Sharpe, no DD improvement found)")

    return best_symbol_to_remove


def greedy_optimize(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    max_positions: int,
    max_dd_limit: float = 9000.0,
    max_breach_events: int = 2,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> Tuple[Set[str], Dict, List[Dict]]:
    """
    Greedy optimization: iteratively remove worst symbols until constraints met.

    Returns:
        best_symbols: Set of symbols in best valid configuration
        best_metrics: Performance metrics for best config
        history: List of all tested configurations
    """

    current_symbols = set(symbol_trades.keys())
    history = []
    best_valid_config = None

    logger.info(f"\n{'='*80}")
    logger.info(f"GREEDY OPTIMIZATION: max_positions={max_positions}")
    logger.info(f"  Starting with {len(current_symbols)} symbols: {sorted(current_symbols)}")
    logger.info(f"  Constraints: MaxDD < ${max_dd_limit:,.0f}, Breaches <= {max_breach_events}")
    logger.info(f"{'='*80}\n")

    iteration = 0

    while len(current_symbols) >= 1:
        iteration += 1
        logger.info(f"Iteration {iteration}: Testing {len(current_symbols)} symbols...")

        # Evaluate current portfolio
        metrics = evaluate_portfolio(
            symbol_trades, symbol_metadata, current_symbols,
            max_positions, initial_capital, daily_stop_loss
        )

        config = {
            'iteration': iteration,
            'max_positions': max_positions,
            'n_symbols': len(current_symbols),
            'symbols': sorted(current_symbols),
            'sharpe': metrics['sharpe_ratio'],
            'cagr': metrics['cagr'],
            'total_return': metrics['total_return_dollars'],
            'max_dd': abs(metrics['max_drawdown_dollars']),
            'breach_events': metrics['dd_breach_6k_events'],
            'breach_periods': metrics['dd_breach_6k_periods'],
            'daily_stops': metrics['daily_stops_hit'],
            'profit_factor': metrics['profit_factor']
        }
        history.append(config)

        # Check constraints
        meets_dd_constraint = config['max_dd'] < max_dd_limit
        meets_breach_constraint = config['breach_events'] <= max_breach_events

        logger.info(f"  Results: Sharpe={config['sharpe']:.3f}, DD=${config['max_dd']:,.0f}, "
                   f"Breaches={config['breach_events']:.0f}")

        if meets_dd_constraint and meets_breach_constraint:
            logger.info(f"  ✅ CONSTRAINTS MET!")
            if best_valid_config is None or config['sharpe'] > best_valid_config['sharpe']:
                best_valid_config = config
                logger.info(f"  🏆 New best valid config (Sharpe={config['sharpe']:.3f})")
        else:
            reasons = []
            if not meets_dd_constraint:
                reasons.append(f"DD=${config['max_dd']:,.0f} > ${max_dd_limit:,.0f}")
            if not meets_breach_constraint:
                reasons.append(f"Breaches={config['breach_events']:.0f} > {max_breach_events}")
            logger.info(f"  ❌ Constraints violated: {', '.join(reasons)}")

        # Stop if only 1 symbol left
        if len(current_symbols) <= 1:
            break

        # Remove worst symbol
        worst_symbol = identify_worst_symbol(
            symbol_trades, symbol_metadata, current_symbols,
            max_positions, initial_capital, daily_stop_loss
        )
        current_symbols.remove(worst_symbol)
        logger.info("")

    if best_valid_config:
        best_symbols = set(best_valid_config['symbols'])
        best_metrics = evaluate_portfolio(
            symbol_trades, symbol_metadata, best_symbols,
            max_positions, initial_capital, daily_stop_loss
        )
    else:
        # No valid config found, return closest
        logger.warning(f"⚠️  No configuration met constraints for max_positions={max_positions}")
        best_symbols = None
        best_metrics = None

    return best_symbols, best_metrics, history


def main():
    parser = argparse.ArgumentParser(description='Greedy portfolio optimizer with broker constraints')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing optimization results')
    parser.add_argument('--min-positions', type=int, default=1,
                       help='Minimum max_positions to test')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='Maximum max_positions to test')
    parser.add_argument('--max-dd-limit', type=float, default=9000.0,
                       help='Maximum drawdown limit ($)')
    parser.add_argument('--max-breach-events', type=int, default=2,
                       help='Maximum number of $6k DD breach events')
    parser.add_argument('--initial-capital', type=float, default=250000.0,
                       help='Initial portfolio capital')
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0,
                       help='Daily stop loss limit ($)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for results')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # Discover and load all symbols
    logger.info("="*80)
    logger.info("GREEDY PORTFOLIO OPTIMIZER")
    logger.info("="*80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Constraints: MaxDD < ${args.max_dd_limit:,.0f}, Breaches <= {args.max_breach_events}")
    logger.info(f"Testing max_positions from {args.min_positions} to {args.max_positions}")

    available_symbols = discover_available_symbols(results_dir)
    logger.info(f"\nDiscovered {len(available_symbols)} symbols with trade data")

    # Load all symbol data
    logger.info("\nLoading trade data...")
    symbol_trades = {}
    symbol_metadata = {}

    for symbol in available_symbols:
        try:
            trades_df, metadata = load_symbol_trades(results_dir, symbol)
            symbol_trades[symbol] = trades_df
            symbol_metadata[symbol] = metadata
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded {len(symbol_trades)} symbols successfully")

    # Run greedy optimization for each max_positions
    all_results = []

    for max_pos in range(args.min_positions, args.max_positions + 1):
        best_symbols, best_metrics, history = greedy_optimize(
            symbol_trades=symbol_trades,
            symbol_metadata=symbol_metadata,
            max_positions=max_pos,
            max_dd_limit=args.max_dd_limit,
            max_breach_events=args.max_breach_events,
            initial_capital=args.initial_capital,
            daily_stop_loss=args.daily_stop_loss
        )

        if best_symbols and best_metrics:
            result = {
                'max_positions': max_pos,
                'n_symbols': len(best_symbols),
                'symbols': sorted(best_symbols),
                'sharpe': best_metrics['sharpe_ratio'],
                'cagr': best_metrics['cagr'],
                'total_return': best_metrics['total_return_dollars'],
                'max_dd': abs(best_metrics['max_drawdown_dollars']),
                'breach_events': best_metrics['dd_breach_6k_events'],
                'breach_periods': best_metrics['dd_breach_6k_periods'],
                'pct_time_in_breach': best_metrics['dd_breach_6k_pct_time'],
                'daily_stops': best_metrics['daily_stops_hit'],
                'profit_factor': best_metrics['profit_factor'],
                'avg_positions': best_metrics['avg_positions']
            }
            all_results.append(result)

    # Display results
    logger.info("\n" + "="*100)
    logger.info("SUMMARY: Valid Configurations (Meet Constraints)")
    logger.info("="*100)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('sharpe', ascending=False)

        print(f"\n{'MaxPos':<10}{'#Sym':<8}{'Sharpe':<10}{'CAGR%':<10}{'MaxDD $':<12}"
              f"{'Breaches':<12}{'Stops':<10}{'Symbols'}")
        print("-" * 100)

        for _, row in results_df.iterrows():
            symbols_str = ', '.join(row['symbols'][:5])
            if row['n_symbols'] > 5:
                symbols_str += f" +{row['n_symbols']-5} more"

            print(f"{row['max_positions']:<10.0f}"
                  f"{row['n_symbols']:<8.0f}"
                  f"{row['sharpe']:<10.3f}"
                  f"{row['cagr']*100:<10.2f}"
                  f"${row['max_dd']:>10,.0f}"
                  f"{row['breach_events']:>11.0f}"
                  f"{row['daily_stops']:>10.0f}  "
                  f"{symbols_str}")

        # Best configuration
        best = results_df.iloc[0]
        logger.info("\n" + "="*100)
        logger.info("🏆 RECOMMENDED CONFIGURATION (Highest Sharpe)")
        logger.info("="*100)
        logger.info(f"  Max Positions: {int(best['max_positions'])}")
        logger.info(f"  Symbols ({int(best['n_symbols'])}): {', '.join(best['symbols'])}")
        logger.info(f"  Sharpe Ratio: {best['sharpe']:.3f}")
        logger.info(f"  CAGR: {best['cagr']*100:.2f}%")
        logger.info(f"  Total Return: ${best['total_return']:,.2f}")
        logger.info(f"  Max Drawdown: ${best['max_dd']:,.2f}")
        logger.info(f"  DD Breach Events: {int(best['breach_events'])}")
        logger.info(f"  DD Breach Periods: {int(best['breach_periods'])}")
        logger.info(f"  % Time in Breach: {best['pct_time_in_breach']:.2f}%")
        logger.info(f"  Daily Stops Hit: {int(best['daily_stops'])}")
        logger.info(f"  Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"  Avg Positions: {best['avg_positions']:.2f}")

        # Save to CSV
        if args.output:
            results_df.to_csv(args.output, index=False)
            logger.info(f"\n✅ Results saved to: {args.output}")

    else:
        logger.warning("\n⚠️  No configurations met the constraints!")
        logger.warning("Consider:")
        logger.warning("  1. Relaxing constraints (higher max_dd_limit or max_breach_events)")
        logger.warning("  2. Adjusting daily_stop_loss")
        logger.warning("  3. Using fewer max_positions")


if __name__ == '__main__':
    main()
