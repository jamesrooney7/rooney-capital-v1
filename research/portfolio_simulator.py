#!/usr/bin/env python3
"""
Portfolio Simulator Using Intraday Trade Data

This script loads detailed trade-level data with entry/exit timestamps and
simulates portfolio performance with:
- Intraday overlapping position tracking
- Max positions constraint (based on actual open positions)
- Daily stop loss with immediate exit of all positions
- Proper P&L calculation (slippage already included in strategy factory layer)

Usage:
    python research/portfolio_simulator.py \
        --results-dir results \
        --min-positions 1 \
        --max-positions 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def discover_available_symbols(results_dir: Path) -> List[str]:
    """Discover symbols with available trade data."""
    symbols = []

    for item in results_dir.iterdir():
        if item.is_dir() and item.name.endswith('_optimization'):
            symbol = item.name.replace('_optimization', '')
            trades_file = item / f"{symbol}_rf_best_trades.csv"

            if trades_file.exists():
                symbols.append(symbol)

    return sorted(symbols)


def load_symbol_trades(results_dir: Path, symbol: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load detailed trade data with entry/exit timestamps.

    Returns:
        trades_df: DataFrame with Date/Time, Exit Date/Time, PnL, etc.
        metadata: Dict with symbol optimization metrics
    """
    opt_dir = results_dir / f"{symbol}_optimization"

    # Load detailed trades
    trades_file = opt_dir / f"{symbol}_rf_best_trades.csv"
    if not trades_file.exists():
        raise FileNotFoundError(f"Detailed trade data not found: {trades_file}")

    trades_df = pd.read_csv(trades_file)

    # Parse timestamps
    trades_df['entry_time'] = pd.to_datetime(trades_df['Date/Time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['Exit Date/Time'])

    # Only keep trades that were selected by ML model
    if 'Model_Selected' in trades_df.columns:
        trades_df = trades_df[trades_df['Model_Selected'] == 1].copy()
    elif 'selected' in trades_df.columns:
        trades_df = trades_df[trades_df['selected'] == 1].copy()

    # Get P&L in dollars (slippage already included in strategy factory layer)
    if 'Model_PnL_USD_When_Selected' in trades_df.columns:
        trades_df['pnl_usd'] = trades_df['Model_PnL_USD_When_Selected']
    elif 'pnl_usd_when_selected' in trades_df.columns:
        trades_df['pnl_usd'] = trades_df['pnl_usd_when_selected']
    elif 'Model_PnL_USD' in trades_df.columns:
        trades_df['pnl_usd'] = trades_df['Model_PnL_USD']
    else:
        raise ValueError(f"No P&L column found in {symbol} trades")

    # Remove trades with missing data
    trades_df = trades_df.dropna(subset=['entry_time', 'exit_time', 'pnl_usd'])

    # Sort by entry time
    trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

    # Load metadata
    metadata = {}
    summary_file = opt_dir / f"{symbol}_rf_best_summary.txt"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            metadata = json.load(f)

    best_file = opt_dir / "best.json"
    if best_file.exists():
        with open(best_file, 'r') as f:
            best_meta = json.load(f)
            metadata.update(best_meta)

    logger.info(f"  {symbol}: Loaded {len(trades_df)} ML-filtered trades")

    return trades_df, metadata


def simulate_portfolio_intraday(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    max_positions: int,
    initial_capital: float = 150000.0,
    daily_stop_loss: float = 2500.0,
    exit_slippage: float = 0.0,
    ranking_method: str = 'sharpe'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate portfolio with intraday position tracking and daily stop loss.

    Parameters:
        symbol_trades: {symbol: trades_df with entry/exit times}
        symbol_metadata: {symbol: metadata_dict}
        max_positions: Maximum positions open simultaneously
        initial_capital: Starting capital (default $150k to match live account)
        daily_stop_loss: Daily loss limit in dollars
        exit_slippage: P&L penalty for forced exits (default 0, slippage already in P&L)
        ranking_method: How to rank symbols when limiting positions

    Returns:
        equity_df: DataFrame with timestamp, equity, positions, etc.
        metrics: Performance metrics dict
    """
    # Create ranking scores
    symbol_scores = {}
    for symbol in symbol_metadata.keys():
        if ranking_method == 'sharpe':
            symbol_scores[symbol] = symbol_metadata[symbol].get('Sharpe_OOS_CPCV', 0)
        elif ranking_method == 'profit_factor':
            symbol_scores[symbol] = symbol_metadata[symbol].get('Profit_Factor', 0)
        else:
            symbol_scores[symbol] = 0

    # Get all trade events (entries and exits) sorted by time
    all_events = []

    for symbol, trades_df in symbol_trades.items():
        for _, trade in trades_df.iterrows():
            # Entry event
            all_events.append({
                'time': trade['entry_time'],
                'type': 'entry',
                'symbol': symbol,
                'trade_id': trade.name,
                'pnl_usd': trade['pnl_usd'],
                'exit_time': trade['exit_time']
            })

            # Exit event
            all_events.append({
                'time': trade['exit_time'],
                'type': 'exit',
                'symbol': symbol,
                'trade_id': trade.name,
                'pnl_usd': trade['pnl_usd'],
                'exit_time': trade['exit_time']
            })

    # Sort all events by time
    all_events_df = pd.DataFrame(all_events).sort_values('time').reset_index(drop=True)

    # Simulation state
    equity = initial_capital
    open_positions = {}  # {(symbol, trade_id): {entry_time, pnl_usd, position_size}}
    equity_curve = []

    current_day = None
    daily_pnl = 0.0
    stopped_out = False
    stop_count = 0

    symbol_usage = {}
    position_counts = []

    logger.info(f"\n  Simulating {len(all_events_df)} trade events...")

    for _, event in all_events_df.iterrows():
        time = event['time']
        day = time.date()

        # Check if new trading day
        if current_day is None or day != current_day:
            if stopped_out:
                logger.debug(f"    Day {current_day}: Stopped out, Daily P&L: ${daily_pnl:,.2f}")

            current_day = day
            daily_pnl = 0.0
            stopped_out = False

        # If stopped out, only process exits (close positions)
        if stopped_out:
            if event['type'] == 'exit':
                key = (event['symbol'], event['trade_id'])
                if key in open_positions:
                    # Close position (already counted the P&L when we stopped out)
                    del open_positions[key]

            # Record state
            equity_curve.append({
                'time': time,
                'equity': equity,
                'n_positions': len(open_positions),
                'daily_pnl': daily_pnl,
                'stopped_out': 1
            })
            continue

        # Process entry event
        if event['type'] == 'entry':
            symbol = event['symbol']

            # Check if we're at max positions
            if len(open_positions) >= max_positions:
                # Need to rank and potentially reject this entry
                # Get all symbols with open positions + this new one
                active_symbols = set(s for s, _ in open_positions.keys())
                candidate_symbols = active_symbols | {symbol}

                # Rank by score
                ranked = sorted(candidate_symbols, key=lambda s: symbol_scores.get(s, 0), reverse=True)

                # If this symbol isn't in top N, reject entry
                if symbol not in ranked[:max_positions]:
                    continue  # Skip this trade

                # If we need to make room, close lowest-ranked position
                if len(open_positions) >= max_positions:
                    # Find lowest ranked symbol currently open
                    current_symbols_ranked = [s for s in ranked if s in active_symbols]
                    symbol_to_close = current_symbols_ranked[-1]

                    # Find one position of this symbol to close
                    for key in list(open_positions.keys()):
                        if key[0] == symbol_to_close:
                            pos = open_positions[key]
                            # Realize P&L with slippage penalty
                            pnl_with_slippage = pos['pnl_usd'] * (1 - exit_slippage)
                            equity += pnl_with_slippage
                            daily_pnl += pnl_with_slippage
                            del open_positions[key]
                            break

            # Calculate position size (equal weight)
            n_positions = len(open_positions) + 1
            position_value = equity / n_positions

            # Open position
            key = (symbol, event['trade_id'])
            open_positions[key] = {
                'entry_time': time,
                'pnl_usd': event['pnl_usd'],
                'position_size': position_value,
                'exit_time': event['exit_time']
            }

            # Track usage
            symbol_usage[symbol] = symbol_usage.get(symbol, 0) + 1

        # Process exit event
        elif event['type'] == 'exit':
            key = (event['symbol'], event['trade_id'])

            if key in open_positions:
                pos = open_positions[key]

                # Realize P&L (no slippage for normal exits)
                equity += pos['pnl_usd']
                daily_pnl += pos['pnl_usd']

                # Close position
                del open_positions[key]

                # Check for daily stop loss
                if daily_pnl <= -daily_stop_loss:
                    stopped_out = True
                    stop_count += 1

                    logger.debug(f"    STOP LOSS HIT at {time}: Daily P&L: ${daily_pnl:,.2f}")

                    # Immediately close ALL open positions with slippage
                    for open_key in list(open_positions.keys()):
                        open_pos = open_positions[open_key]
                        # Apply slippage penalty for emergency exit
                        pnl_with_slippage = open_pos['pnl_usd'] * (1 - exit_slippage)
                        equity += pnl_with_slippage
                        daily_pnl += pnl_with_slippage
                        del open_positions[open_key]

                    logger.debug(f"    Closed all positions. Final daily P&L: ${daily_pnl:,.2f}")

        # Record state
        equity_curve.append({
            'time': time,
            'equity': equity,
            'n_positions': len(open_positions),
            'daily_pnl': daily_pnl,
            'stopped_out': 1 if stopped_out else 0
        })

        position_counts.append(len(open_positions))

    # Create DataFrame
    equity_df = pd.DataFrame(equity_curve)

    if equity_df.empty:
        return pd.DataFrame(), {}

    # Calculate metrics
    metrics = calculate_metrics(
        equity_df,
        initial_capital,
        stop_count,
        position_counts,
        symbol_usage,
        len(symbol_trades)
    )

    return equity_df, metrics


def calculate_metrics(
    equity_df: pd.DataFrame,
    initial_capital: float,
    stop_count: int,
    position_counts: List[int],
    symbol_usage: Dict[str, int],
    n_symbols_total: int
) -> Dict:
    """Calculate portfolio performance metrics."""
    if len(equity_df) < 2:
        return {
            'total_return_dollars': 0,
            'total_return_pct': 0,
            'cagr': 0,
            'sharpe_ratio': 0,
            'max_drawdown_dollars': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'daily_stops_hit': 0,
            'avg_positions': 0,
            'n_symbols_used': 0,
            'n_symbols_total': n_symbols_total
        }

    # Get equity curve
    equity_curve = equity_df['equity']

    # Calculate returns between events
    returns = equity_curve.pct_change().fillna(0)

    # Total return
    final_equity = equity_curve.iloc[-1]
    total_return_dollars = final_equity - initial_capital
    total_return_pct = total_return_dollars / initial_capital

    # Annualized metrics (estimate based on time span)
    time_span_days = (equity_df['time'].iloc[-1] - equity_df['time'].iloc[0]).days
    years = time_span_days / 365.25

    cagr = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0

    # Approximate volatility (this is rough with irregular timestamps)
    # Resample to daily for better vol estimate
    daily_equity = equity_df.set_index('time')['equity'].resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    annualized_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    sharpe_ratio = cagr / annualized_vol if annualized_vol > 0 else 0

    # Profit factor
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    gross_profits = (positive_returns * initial_capital).sum()
    gross_losses = abs((negative_returns * initial_capital).sum())

    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    # Max drawdown
    cummax = equity_curve.expanding().max()
    drawdown = equity_curve - cummax
    drawdown_pct = drawdown / cummax

    max_drawdown_dollars = drawdown.min()
    max_drawdown_pct = drawdown_pct.min()

    # Drawdown breach analysis (for broker limits like $6000 max DD)
    drawdown_breach_6k = (drawdown <= -6000)
    n_breach_periods = drawdown_breach_6k.sum()
    pct_time_in_breach = (n_breach_periods / len(drawdown)) * 100 if len(drawdown) > 0 else 0

    # Count number of separate breach events
    breach_events = 0
    in_breach = False
    for is_breach in drawdown_breach_6k:
        if is_breach and not in_breach:
            breach_events += 1
            in_breach = True
        elif not is_breach:
            in_breach = False

    # Average positions
    avg_positions = np.mean(position_counts) if position_counts else 0

    # Symbol usage
    most_used_symbols = sorted(symbol_usage.items(), key=lambda x: x[1], reverse=True)

    return {
        'total_return_dollars': total_return_dollars,
        'total_return_pct': total_return_pct,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'annualized_volatility': annualized_vol,
        'max_drawdown_dollars': max_drawdown_dollars,
        'max_drawdown_pct': max_drawdown_pct,
        'profit_factor': profit_factor,
        'gross_profits': gross_profits,
        'gross_losses': gross_losses,
        'daily_stops_hit': stop_count,
        'avg_positions': avg_positions,
        'n_symbols_used': len(symbol_usage),
        'n_symbols_total': n_symbols_total,
        'symbol_usage': symbol_usage,
        'most_used_symbols': [s for s, _ in most_used_symbols[:5]],
        'dd_breach_6k_events': breach_events,
        'dd_breach_6k_periods': n_breach_periods,
        'dd_breach_6k_pct_time': pct_time_in_breach
    }


def optimize_max_positions(
    symbol_trades: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    min_positions: int,
    max_positions: int,
    initial_capital: float,
    daily_stop_loss: float,
    exit_slippage: float,
    ranking_method: str = 'sharpe'
) -> pd.DataFrame:
    """Optimize max_positions parameter."""
    n_symbols = len(symbol_trades)
    max_pos = min(max_positions, n_symbols)

    logger.info(f"\n{'='*90}")
    logger.info(f"PORTFOLIO OPTIMIZATION - INTRADAY POSITION TRACKING")
    logger.info(f"Using detailed trade data with entry/exit timestamps")
    logger.info(f"Initial Capital: ${initial_capital:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    logger.info(f"Tracks ACTUAL overlapping positions (prevents look-ahead bias)")
    logger.info(f"{'='*90}\n")

    results = []

    for max_pos_val in range(min_positions, max_pos + 1):
        logger.info(f"Testing max_positions = {max_pos_val}")

        equity_df, metrics = simulate_portfolio_intraday(
            symbol_trades,
            symbol_metadata,
            max_pos_val,
            initial_capital=initial_capital,
            daily_stop_loss=daily_stop_loss,
            exit_slippage=exit_slippage,
            ranking_method=ranking_method
        )

        result = {'max_positions': max_pos_val, **metrics}
        results.append(result)

        logger.info(f"  Sharpe: {metrics['sharpe_ratio']:>7.3f} | "
                   f"CAGR: {metrics['cagr']*100:>6.2f}% | "
                   f"Return: ${metrics['total_return_dollars']:>10,.0f} | "
                   f"MaxDD: ${metrics['max_drawdown_dollars']:>10,.0f} | "
                   f"PF: {metrics['profit_factor']:>5.2f} | "
                   f"Stops: {metrics['daily_stops_hit']:>3.0f}\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Drop dict/list columns for CSV
    columns_to_drop = ['symbol_usage', 'most_used_symbols']
    for col in columns_to_drop:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    # Print summary
    logger.info(f"\n{'='*90}")
    logger.info("OPTIMIZATION RESULTS (sorted by Sharpe)")
    logger.info(f"{'='*90}\n")

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
    best_idx = results_df.index[0]
    best = results[best_idx]

    logger.info(f"\n🏆 OPTIMAL CONFIGURATION:")
    logger.info(f"   Max Positions: {int(best['max_positions'])}")
    logger.info(f"   Symbols Used: {best['n_symbols_used']} of {best['n_symbols_total']} available")
    logger.info(f"   Most Frequently Used: {', '.join(best.get('most_used_symbols', []))}")
    logger.info(f"   Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    logger.info(f"   CAGR: {best['cagr']*100:.2f}%")
    logger.info(f"   Total Return: ${best['total_return_dollars']:,.2f}")
    logger.info(f"   Max Drawdown: ${best['max_drawdown_dollars']:,.2f} ({best['max_drawdown_pct']*100:.2f}%)")
    logger.info(f"   Profit Factor: {best['profit_factor']:.2f}")
    logger.info(f"   Avg Positions: {best['avg_positions']:.2f}")
    logger.info(f"   Daily Stops Hit: {best['daily_stops_hit']:.0f}")
    logger.info(f"\n⚠️  DRAWDOWN BREACH ANALYSIS ($6000 broker limit):")
    logger.info(f"   Number of Breach Events: {best['dd_breach_6k_events']:.0f}")
    logger.info(f"   Total Periods in Breach: {best['dd_breach_6k_periods']:.0f}")
    logger.info(f"   % Time in Breach: {best['dd_breach_6k_pct_time']:.2f}%\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio simulator using intraday trade data'
    )

    parser.add_argument('--symbols', nargs='+', help='Symbols to include')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory with optimization results')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--initial-cash', type=float, default=150000.0,
                       help='Initial capital (default: $150k to match live account)')
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)
    parser.add_argument('--exit-slippage', type=float, default=0.0002,
                       help='Slippage when forced to exit (default: 0.02%%)')
    parser.add_argument('--ranking-method', type=str, default='sharpe',
                       choices=['sharpe', 'profit_factor'],
                       help='How to rank symbols')
    parser.add_argument('--output', type=str, help='Output CSV file')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    # Discover or use specified symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = discover_available_symbols(results_dir)
        logger.info(f"Auto-discovered {len(symbols)} symbols: {', '.join(symbols)}")

    if not symbols:
        logger.error("No symbols to process")
        return 1

    # Load trade data
    logger.info("\nLoading detailed trade data with entry/exit timestamps...")
    logger.info("=" * 80)

    symbol_trades = {}
    symbol_metadata = {}

    for symbol in symbols:
        try:
            trades_df, metadata = load_symbol_trades(results_dir, symbol)
            symbol_trades[symbol] = trades_df
            symbol_metadata[symbol] = metadata
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info("=" * 80)
    logger.info(f"✅ Loaded {len(symbol_trades)} symbols\n")

    if not symbol_trades:
        logger.error("No trade data loaded")
        return 1

    # Run optimization
    max_pos = args.max_positions or len(symbol_trades)

    results_df = optimize_max_positions(
        symbol_trades,
        symbol_metadata,
        args.min_positions,
        max_pos,
        args.initial_cash,
        args.daily_stop_loss,
        args.exit_slippage,
        args.ranking_method
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"💾 Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
