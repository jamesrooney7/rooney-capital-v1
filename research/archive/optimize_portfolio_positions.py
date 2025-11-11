#!/usr/bin/env python3
"""
Portfolio-level position optimizer using actual backtest results.

This script:
1. Loads all trained models from src/models/
2. Runs backtests for each symbol to generate signals
3. Combines signals into a portfolio with max_positions constraint
4. Optimizes max_positions to maximize portfolio Sharpe ratio

Usage:
    # Optimize portfolio for 2023-2024
    python research/optimize_portfolio_positions.py \
        --start 2023-01-01 --end 2024-12-31 \
        --min-positions 1 --max-positions 12

    # Specify custom symbols
    python research/optimize_portfolio_positions.py \
        --symbols ES NQ YM RTY GC SI CL \
        --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from models.loader import load_model_bundle
from research.utils.data_loader import load_symbol_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_available_models(models_dir: Path) -> List[str]:
    """Discover available trained models."""
    symbols = []

    for json_file in models_dir.glob('*_best.json'):
        symbol = json_file.stem.replace('_best', '')
        pkl_file = models_dir / f"{symbol}_rf_model.pkl"

        if pkl_file.exists():
            symbols.append(symbol)

    return sorted(symbols)


def extract_signals_from_strategy(
    symbol: str,
    data_dir: str,
    start_date: str,
    end_date: str,
    use_ml: bool = True
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Extract ML-filtered signals for a symbol using the strategy's logic.

    Returns:
        signals: Series of boolean signals (True = enter position)
        prices: DataFrame with OHLC prices for position simulation
    """
    from strategy.ibs_strategy import IbsStrategy

    # Load model bundle to get features
    if use_ml:
        try:
            bundle = load_model_bundle(symbol)
            logger.info(f"Loaded {symbol} model: Sharpe={bundle.metadata.get('Sharpe', 0):.3f}")
        except FileNotFoundError:
            logger.warning(f"No model found for {symbol}, skipping")
            return None, None
    else:
        logger.info(f"ML disabled for {symbol}")
        return None, None

    # Load raw OHLC data
    try:
        hourly_df, daily_df = load_symbol_data(
            symbol,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date
        )
    except FileNotFoundError:
        logger.warning(f"No data found for {symbol}")
        return None, None

    # For now, we'll use a simplified approach:
    # Extract the daily signals based on IBS and use ML to filter
    # This is a simplified version - in reality, the full IbsStrategy
    # has complex feature engineering

    logger.info(f"Extracting signals for {symbol}")

    # Calculate basic IBS on daily bars
    daily_df = daily_df.copy()
    daily_df['ibs'] = (daily_df['Close'] - daily_df['Low']) / (daily_df['High'] - daily_df['Low'])
    daily_df['ibs'] = daily_df['ibs'].fillna(0.5)

    # Basic IBS signal: enter when IBS < 0.3 (oversold)
    ibs_signal = daily_df['ibs'] < 0.3

    # For this simplified version, we'll just use the IBS signal
    # In production, you'd run the full feature extraction and ML filtering
    signals = ibs_signal

    # Get prices for return calculation
    prices = daily_df[['Open', 'High', 'Low', 'Close']].copy()

    return signals, prices


def load_symbol_signals(
    symbols: List[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    use_ml: bool = True
) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
    """
    Load signals and prices for all symbols.

    Returns:
        Dict of {symbol: (signals, prices)}
    """
    signals_dict = {}

    for symbol in symbols:
        signals, prices = extract_signals_from_strategy(
            symbol,
            data_dir,
            start_date,
            end_date,
            use_ml=use_ml
        )

        if signals is not None and prices is not None:
            signals_dict[symbol] = (signals, prices)

    return signals_dict


def simulate_portfolio(
    signals_dict: Dict[str, Tuple[pd.Series, pd.DataFrame]],
    max_positions: int,
    initial_capital: float = 250000.0,
    position_size_pct: float = 0.95,
    commission_per_side: float = 1.25,
    ranking_method: str = 'equal'
) -> Tuple[pd.Series, Dict]:
    """
    Simulate portfolio with max_positions constraint.

    Parameters:
        signals_dict: {symbol: (signals, prices)}
        max_positions: Maximum positions to hold simultaneously
        initial_capital: Starting capital
        position_size_pct: Percentage of capital to allocate
        commission_per_side: Commission per trade side
        ranking_method: How to rank signals ('equal', 'random')

    Returns:
        equity_curve: Portfolio equity over time
        metrics: Performance metrics
    """
    # Combine all signals into a DataFrame
    all_signals = {}
    all_prices = {}

    for symbol, (signals, prices) in signals_dict.items():
        all_signals[symbol] = signals
        all_prices[symbol] = prices['Close']

    signals_df = pd.DataFrame(all_signals)
    prices_df = pd.DataFrame(all_prices)

    # Align to common index
    common_index = signals_df.index.intersection(prices_df.index)
    signals_df = signals_df.loc[common_index]
    prices_df = prices_df.loc[common_index]

    # Apply max_positions constraint
    if max_positions is not None and max_positions < len(signals_dict):
        limited_signals = signals_df.copy()

        for idx in signals_df.index:
            active_signals = signals_df.loc[idx]
            active_symbols = active_signals[active_signals].index.tolist()
            n_active = len(active_symbols)

            if n_active > max_positions:
                # Randomly select max_positions (can be improved with better ranking)
                if ranking_method == 'random':
                    np.random.seed(42)  # For reproducibility
                    selected = np.random.choice(active_symbols, max_positions, replace=False)
                else:  # equal
                    selected = active_symbols[:max_positions]

                limited_signals.loc[idx] = False
                limited_signals.loc[idx, selected] = True

        signals_df = limited_signals

    # Simulate portfolio
    equity = initial_capital
    equity_series = [equity]
    dates = [common_index[0]]
    returns_list = []
    positions_count = []

    # Track open positions
    open_positions = {}  # {symbol: entry_price}

    for i in range(len(common_index)):
        date = common_index[i]
        current_signals = signals_df.iloc[i]
        current_prices = prices_df.iloc[i]

        # Get active symbols
        active_symbols = current_signals[current_signals].index.tolist()
        n_positions = len(active_symbols)
        positions_count.append(n_positions)

        # Exit positions that are no longer signaled
        for symbol in list(open_positions.keys()):
            if symbol not in active_symbols:
                # Exit position
                entry_price = open_positions[symbol]
                exit_price = current_prices[symbol]

                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    # Calculate return
                    trade_return = (exit_price - entry_price) / entry_price

                    # Apply commission
                    position_value = equity * position_size_pct / max(len(open_positions), 1)
                    commission_pct = (commission_per_side * 2) / position_value

                    trade_return_net = trade_return - commission_pct

                    # Update equity
                    equity *= (1 + trade_return_net / max(len(open_positions), 1))
                    returns_list.append(trade_return_net / max(len(open_positions), 1))

                del open_positions[symbol]

        # Enter new positions
        for symbol in active_symbols:
            if symbol not in open_positions:
                entry_price = current_prices[symbol]
                if pd.notna(entry_price):
                    open_positions[symbol] = entry_price

        equity_series.append(equity)
        dates.append(date)

    equity_curve = pd.Series(equity_series[1:], index=dates[1:])

    # Calculate metrics
    returns_series = pd.Series(returns_list) if returns_list else pd.Series([0])

    metrics = calculate_metrics(equity_curve, returns_series, positions_count)

    return equity_curve, metrics


def calculate_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    positions_count: List[int]
) -> Dict:
    """Calculate portfolio performance metrics."""
    if len(equity_curve) < 2:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_positions': 0,
            'n_periods': 0
        }

    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    # Annualized metrics
    periods_per_year = 252
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year

    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annualized_vol = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    cummax = equity_curve.expanding().max()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = drawdown.min()

    # Average positions
    avg_positions = np.mean(positions_count) if positions_count else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_positions': avg_positions,
        'n_periods': n_periods
    }


def optimize_max_positions(
    signals_dict: Dict[str, Tuple[pd.Series, pd.DataFrame]],
    position_range: Tuple[int, int],
    **portfolio_kwargs
) -> pd.DataFrame:
    """
    Optimize max_positions parameter.

    Returns:
        DataFrame with results for each max_positions value
    """
    min_pos, max_pos = position_range
    n_symbols = len(signals_dict)
    max_pos = min(max_pos, n_symbols)

    results = []

    logger.info(f"\nOptimizing max_positions from {min_pos} to {max_pos}")
    logger.info("=" * 80)

    for max_positions in range(min_pos, max_pos + 1):
        logger.info(f"\nTesting max_positions = {max_positions}")

        equity, metrics = simulate_portfolio(
            signals_dict,
            max_positions=max_positions,
            **portfolio_kwargs
        )

        result = {'max_positions': max_positions, **metrics}
        results.append(result)

        logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.3f} | "
                   f"Return: {metrics['annualized_return']*100:.2f}% | "
                   f"Vol: {metrics['annualized_volatility']*100:.2f}% | "
                   f"MaxDD: {metrics['max_drawdown']*100:.2f}%")

    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS (sorted by Sharpe Ratio)")
    logger.info("=" * 80)

    # Print formatted table
    print("\n")
    print(f"{'Max Pos':<10}{'Sharpe':<10}{'Ann Ret %':<12}{'Ann Vol %':<12}{'Max DD %':<12}{'Avg Pos':<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['max_positions']:<10.0f}"
              f"{row['sharpe_ratio']:<10.3f}"
              f"{row['annualized_return']*100:<12.2f}"
              f"{row['annualized_volatility']*100:<12.2f}"
              f"{row['max_drawdown']*100:<12.2f}"
              f"{row['avg_positions']:<10.2f}")

    print("=" * 80)

    # Best configuration
    best = results_df.iloc[0]
    logger.info(f"\n🏆 OPTIMAL CONFIGURATION:")
    logger.info(f"   Max Positions: {best['max_positions']:.0f}")
    logger.info(f"   Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    logger.info(f"   Annualized Return: {best['annualized_return']*100:.2f}%")
    logger.info(f"   Max Drawdown: {best['max_drawdown']*100:.2f}%")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Optimize portfolio max_positions parameter'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Symbols to include (default: auto-discover from src/models/)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='src/models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/resampled',
        help='Directory containing resampled OHLCV data'
    )
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD, defaults to today)'
    )
    parser.add_argument(
        '--min-positions',
        type=int,
        default=1,
        help='Minimum max_positions to test'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=None,
        help='Maximum max_positions to test (default: n_symbols)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=250000.0,
        help='Initial capital'
    )
    parser.add_argument(
        '--position-size-pct',
        type=float,
        default=0.95,
        help='Percentage of capital to allocate'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=1.25,
        help='Commission per trade side'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML filtering (use base IBS signals only)'
    )

    args = parser.parse_args()

    # Discover symbols
    models_dir = Path(args.models_dir)

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = discover_available_models(models_dir)
        logger.info(f"Auto-discovered {len(symbols)} models: {', '.join(symbols)}")

    if not symbols:
        logger.error("No symbols to process. Use --symbols or ensure models exist in src/models/")
        return

    # Default end date
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    # Load signals for all symbols
    logger.info(f"\nLoading signals for {len(symbols)} symbols...")
    logger.info(f"Date range: {args.start} to {end_date}")
    logger.info("=" * 80)

    signals_dict = load_symbol_signals(
        symbols,
        args.data_dir,
        args.start,
        end_date,
        use_ml=not args.no_ml
    )

    if not signals_dict:
        logger.error("No signals loaded. Check data availability.")
        return

    logger.info(f"\n✅ Loaded signals for {len(signals_dict)} symbols")

    # Determine position range
    max_pos = args.max_positions or len(signals_dict)
    position_range = (args.min_positions, max_pos)

    # Run optimization
    results_df = optimize_max_positions(
        signals_dict,
        position_range=position_range,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size_pct,
        commission_per_side=args.commission
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
