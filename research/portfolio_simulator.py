#!/usr/bin/env python3
"""
Portfolio Simulator Using Pre-Computed Trade Data

This script loads the pre-computed, ML-filtered trade data from optimization
results and simulates portfolio performance with:
- Max positions constraint
- Daily stop loss ($2,500 default)
- Position sizing across symbols

Usage:
    # Simulate portfolio with pre-computed trade data
    python research/portfolio_simulator.py \
        --results-dir results \
        --min-positions 1 \
        --max-positions 10

    # Specify custom symbols
    python research/portfolio_simulator.py \
        --symbols ES NQ YM RTY GC SI \
        --max-positions 6
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_available_symbols(results_dir: Path) -> List[str]:
    """Discover symbols with available trade data."""
    symbols = []

    # Look for {SYMBOL}_optimization directories
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.endswith('_optimization'):
            symbol = item.name.replace('_optimization', '')
            trades_file = item / f"{symbol}_trades.csv"

            if trades_file.exists():
                symbols.append(symbol)

    return sorted(symbols)


def load_symbol_trades(results_dir: Path, symbol: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load daily trade data for a symbol.

    Returns:
        daily_returns: DataFrame with Date, Symbol, Return columns
        metadata: Dict with symbol optimization metrics
    """
    opt_dir = results_dir / f"{symbol}_optimization"

    # Load daily returns
    trades_file = opt_dir / f"{symbol}_trades.csv"
    if not trades_file.exists():
        raise FileNotFoundError(f"Trade data not found: {trades_file}")

    daily_df = pd.read_csv(trades_file)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])

    # Load metadata
    metadata = {}
    summary_file = opt_dir / f"{symbol}_rf_best_summary.txt"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            metadata = json.load(f)

    # Also try best.json
    best_file = opt_dir / "best.json"
    if best_file.exists():
        with open(best_file, 'r') as f:
            best_meta = json.load(f)
            metadata.update(best_meta)

    return daily_df, metadata


def simulate_portfolio(
    symbol_returns: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    max_positions: int,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0,
    ranking_method: str = 'sharpe'
) -> Tuple[pd.Series, Dict]:
    """
    Simulate portfolio with DYNAMIC position selection and daily stop loss.

    Key: Positions are selected day-by-day based on which symbols have active
    trades. This prevents look-ahead bias and simulates real-time trading.

    Parameters:
        symbol_returns: {symbol: daily_returns_df}
        symbol_metadata: {symbol: metadata_dict}
        max_positions: Maximum positions to hold simultaneously
        initial_capital: Starting capital
        daily_stop_loss: Daily loss limit in dollars
        ranking_method: How to rank symbols ('sharpe', 'profit_factor')

    Returns:
        equity_curve: Series of portfolio equity
        metrics: Performance metrics dict
    """
    # Create ranking scores for symbols (used as tiebreaker when >max_positions signal)
    symbol_scores = {}
    for symbol in symbol_metadata.keys():
        if ranking_method == 'sharpe':
            symbol_scores[symbol] = symbol_metadata[symbol].get('Sharpe_OOS_CPCV', 0)
        elif ranking_method == 'profit_factor':
            symbol_scores[symbol] = symbol_metadata[symbol].get('Profit_Factor', 0)
        else:
            symbol_scores[symbol] = 0

    # Get all unique dates across all symbols
    all_dates = set()
    for symbol in symbol_returns.keys():
        all_dates.update(symbol_returns[symbol]['Date'].tolist())

    all_dates = sorted(all_dates)

    # Create portfolio time series
    portfolio_data = []
    equity = initial_capital
    current_day = None
    daily_pnl = 0.0
    stopped_out = False
    stop_count = 0

    # Track which symbols are selected each day (for reporting)
    daily_symbol_counts = {}

    for date in all_dates:
        # Check if new trading day
        day = date.date() if hasattr(date, 'date') else date

        if current_day is None or day != current_day:
            current_day = day
            daily_pnl = 0.0
            stopped_out = False

        # If stopped out for the day, skip trading
        if stopped_out:
            portfolio_data.append({
                'Date': date,
                'Equity': equity,
                'Daily_PnL': daily_pnl,
                'Stopped_Out': 1,
                'N_Positions': 0
            })
            continue

        # DYNAMIC SELECTION: Find all symbols with active trades TODAY
        symbols_with_trades = []
        symbol_returns_today = {}

        for symbol in symbol_returns.keys():
            symbol_df = symbol_returns[symbol]
            date_rows = symbol_df[symbol_df['Date'] == date]

            if len(date_rows) > 0:
                ret = date_rows['Return'].iloc[0]
                # Only include if there's an actual trade (non-zero return)
                if pd.notna(ret) and ret != 0:
                    symbols_with_trades.append(symbol)
                    symbol_returns_today[symbol] = ret

        # If more than max_positions have signals, rank and select top N
        if len(symbols_with_trades) > max_positions:
            # Sort by ranking score (Sharpe or PF)
            ranked = sorted(
                symbols_with_trades,
                key=lambda s: symbol_scores.get(s, 0),
                reverse=True
            )
            selected_today = ranked[:max_positions]
        else:
            selected_today = symbols_with_trades

        # Track symbol usage
        for sym in selected_today:
            daily_symbol_counts[sym] = daily_symbol_counts.get(sym, 0) + 1

        # Calculate portfolio return
        if not selected_today:
            portfolio_data.append({
                'Date': date,
                'Equity': equity,
                'Daily_PnL': daily_pnl,
                'Stopped_Out': 0,
                'N_Positions': 0
            })
            continue

        # Equal weight across selected positions
        n_positions = len(selected_today)
        position_value = equity / n_positions

        period_pnl = 0.0
        for symbol in selected_today:
            symbol_return = symbol_returns_today[symbol]
            position_pnl = position_value * symbol_return
            period_pnl += position_pnl

        # Update daily P&L
        daily_pnl += period_pnl

        # Check for daily stop loss
        if daily_pnl <= -daily_stop_loss:
            stopped_out = True
            stop_count += 1
            logger.debug(f"  Daily stop hit on {date}: ${daily_pnl:,.2f}")

            portfolio_data.append({
                'Date': date,
                'Equity': equity,
                'Daily_PnL': daily_pnl,
                'Stopped_Out': 1,
                'N_Positions': n_positions
            })
            continue

        # Update equity
        equity += period_pnl

        portfolio_data.append({
            'Date': date,
            'Equity': equity,
            'Daily_PnL': period_pnl,
            'Stopped_Out': 0,
            'N_Positions': n_positions
        })

    # Create DataFrame
    portfolio_df = pd.DataFrame(portfolio_data)

    if portfolio_df.empty:
        return pd.Series(dtype=float), {}

    equity_curve = portfolio_df.set_index('Date')['Equity']

    # Calculate metrics
    metrics = calculate_metrics(
        equity_curve,
        initial_capital,
        stop_count
    )

    # Add symbol usage statistics
    total_days_traded = sum(daily_symbol_counts.values())
    most_used_symbols = sorted(
        daily_symbol_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    metrics['symbol_usage'] = daily_symbol_counts
    metrics['most_used_symbols'] = [s for s, _ in most_used_symbols[:5]]
    metrics['n_symbols_total'] = len(symbol_returns)
    metrics['n_symbols_used'] = len(daily_symbol_counts)

    # Log symbol usage
    logger.info(f"  Symbol usage (days traded):")
    for symbol, days in most_used_symbols[:10]:
        pct = (days / len(all_dates)) * 100 if len(all_dates) > 0 else 0
        logger.info(f"    {symbol}: {days} days ({pct:.1f}%)")

    return equity_curve, metrics


def calculate_metrics(
    equity_curve: pd.Series,
    initial_capital: float,
    stop_count: int
) -> Dict:
    """Calculate portfolio performance metrics."""
    if len(equity_curve) < 2:
        return {
            'total_return_dollars': 0,
            'total_return_pct': 0,
            'cagr': 0,
            'sharpe_ratio': 0,
            'max_drawdown_dollars': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'daily_stops_hit': 0
        }

    # Calculate daily returns
    daily_returns = equity_curve.pct_change().fillna(0)

    # Total return
    final_equity = equity_curve.iloc[-1]
    total_return_dollars = final_equity - initial_capital
    total_return_pct = total_return_dollars / initial_capital

    # Annualized metrics
    n_days = len(equity_curve)
    years = n_days / 252

    cagr = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0

    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = cagr / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    cummax = equity_curve.expanding().max()
    drawdown = equity_curve - cummax
    drawdown_pct = drawdown / cummax

    max_drawdown_dollars = drawdown.min()
    max_drawdown_pct = drawdown_pct.min()

    # Profit factor
    positive_returns = daily_returns[daily_returns > 0]
    negative_returns = daily_returns[daily_returns < 0]

    gross_profits = (positive_returns * initial_capital).sum()
    gross_losses = abs((negative_returns * initial_capital).sum())

    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

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
        'n_days': n_days,
        'daily_stops_hit': stop_count
    }


def optimize_max_positions(
    symbol_returns: Dict[str, pd.DataFrame],
    symbol_metadata: Dict[str, dict],
    min_positions: int,
    max_positions: int,
    initial_capital: float,
    daily_stop_loss: float,
    ranking_method: str = 'sharpe'
) -> pd.DataFrame:
    """Optimize max_positions parameter."""
    n_symbols = len(symbol_returns)
    max_pos = min(max_positions, n_symbols)

    logger.info(f"\n{'='*90}")
    logger.info(f"PORTFOLIO OPTIMIZATION - DYNAMIC POSITION SELECTION")
    logger.info(f"Using pre-computed ML-filtered trade data")
    logger.info(f"Initial Capital: ${initial_capital:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    logger.info(f"Positions selected DAILY based on active signals (prevents look-ahead bias)")
    logger.info(f"{'='*90}\n")

    results = []

    for max_pos_val in range(min_positions, max_pos + 1):
        logger.info(f"Testing max_positions = {max_pos_val}")

        equity, metrics = simulate_portfolio(
            symbol_returns,
            symbol_metadata,
            max_pos_val,
            initial_capital=initial_capital,
            daily_stop_loss=daily_stop_loss,
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

    # Drop dict/list columns for CSV compatibility
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

    # Best result (get from original results list to access all metrics)
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
    logger.info(f"   Daily Stops Hit: {best['daily_stops_hit']:.0f}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio simulator using pre-computed trade data'
    )

    parser.add_argument('--symbols', nargs='+', help='Symbols to include')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory with optimization results')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--initial-cash', type=float, default=250000.0)
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0)
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
    logger.info("\nLoading pre-computed trade data...")
    logger.info("=" * 80)

    symbol_returns = {}
    symbol_metadata = {}

    for symbol in symbols:
        try:
            daily_df, metadata = load_symbol_trades(results_dir, symbol)
            symbol_returns[symbol] = daily_df
            symbol_metadata[symbol] = metadata

            sharpe = metadata.get('Sharpe_OOS_CPCV', 0)
            pf = metadata.get('Profit_Factor', 0)
            trades = metadata.get('Trades_Selected', 0)

            logger.info(f"{symbol:6s} | Sharpe: {sharpe:6.3f} | PF: {pf:6.2f} | Trades: {trades:4d}")

        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info("=" * 80)
    logger.info(f"✅ Loaded {len(symbol_returns)} symbols\n")

    if not symbol_returns:
        logger.error("No trade data loaded")
        return 1

    # Run optimization
    max_pos = args.max_positions or len(symbol_returns)

    results_df = optimize_max_positions(
        symbol_returns,
        symbol_metadata,
        args.min_positions,
        max_pos,
        args.initial_cash,
        args.daily_stop_loss,
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
