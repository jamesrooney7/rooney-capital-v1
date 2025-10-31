#!/usr/bin/env python3
"""
Full Portfolio Optimizer with Backtrader Integration.

This script:
1. Runs full backtests for each symbol using IbsStrategy with ML filters
2. Extracts daily signals and returns from each symbol's backtest
3. Combines into a portfolio with max_positions constraint
4. Optimizes max_positions to maximize Sharpe ratio

This approach ensures we use the EXACT production strategy logic with all
feature engineering and ML filtering.

Usage:
    # Run portfolio optimization for 2023-2024
    python research/portfolio_optimizer_full.py \
        --start 2023-01-01 --end 2024-12-31 \
        --min-positions 1 --max-positions 10

    # Custom symbols and save results
    python research/portfolio_optimizer_full.py \
        --symbols ES NQ YM RTY \
        --start 2023-01-01 --end 2024-12-31 \
        --output results/portfolio_optimization.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import backtrader as bt
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.utils.data_loader import setup_cerebro_with_data
from strategy.ibs_strategy import IbsStrategy
from models.loader import load_model_bundle
from config import COMMISSION_PER_SIDE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalCaptureStrategy(IbsStrategy):
    """
    Extended IbsStrategy that captures signals and returns.

    This strategy runs the full production logic but also records
    when signals are generated for portfolio-level analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store signals and returns
        self.signal_log = []  # List of (datetime, signal_generated)

    def next(self):
        """Override next() to capture signals before execution."""
        # Get current datetime
        current_dt = self.datas[0].datetime.datetime(0)

        # Check if we would generate a signal (before position logic)
        signal_generated = False

        # Only generate signals if we have no position
        if not self.position:
            # Run all the filter checks (same as parent class)
            if self._should_enter_position():
                signal_generated = True

        # Log the signal
        self.signal_log.append({
            'datetime': current_dt,
            'signal': signal_generated,
            'close': self.datas[0].close[0]
        })

        # Call parent next() to execute trades
        super().next()

    def _should_enter_position(self) -> bool:
        """
        Check if we should enter a position (without actually entering).

        This replicates the entry logic from IbsStrategy.
        """
        # Get references to data feeds
        hour_data = self.getdatabyname(f"{self.symbol}_hour")
        day_data = self.getdatabyname(f"{self.symbol}_day")

        if hour_data is None or day_data is None:
            return False

        # Basic checks
        if len(hour_data) < 100 or len(day_data) < 100:
            return False

        # IBS check (basic entry condition)
        close = day_data.close[0]
        high = day_data.high[0]
        low = day_data.low[0]

        if high == low:
            return False

        ibs = (close - low) / (high - low)

        # Default IBS threshold for entry (oversold)
        if ibs > 0.3:  # Only enter when IBS < 0.3
            return False

        # If ML filter is enabled, check it
        if self.use_ml_filter and self.ml_model is not None:
            filter_values = self.collect_filter_values()

            if filter_values is None:
                return False

            # Get ML prediction
            try:
                ml_features = []
                for feature_name in self.ml_feature_names:
                    if feature_name in filter_values:
                        ml_features.append(filter_values[feature_name])
                    else:
                        return False  # Missing feature

                if not ml_features:
                    return False

                # Predict
                ml_features_array = np.array(ml_features).reshape(1, -1)
                prediction_proba = self.ml_model.predict_proba(ml_features_array)[0, 1]

                if prediction_proba < self.ml_threshold:
                    return False

            except Exception as e:
                logger.debug(f"ML filter error: {e}")
                return False

        return True


def run_backtest_with_signals(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'data/resampled',
    initial_cash: float = 100000.0,
    use_ml: bool = True
) -> Optional[pd.DataFrame]:
    """
    Run backtest and extract signal log.

    Returns:
        DataFrame with columns: datetime, signal, close, returns
    """
    logger.info(f"Running backtest for {symbol}...")

    # Create Cerebro
    cerebro = bt.Cerebro(runonce=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=COMMISSION_PER_SIDE)

    # Load model and setup data
    symbols_to_load = {symbol, 'TLT'}
    strat_params = {'symbol': symbol}

    if use_ml:
        try:
            bundle = load_model_bundle(symbol)
            strat_params = bundle.strategy_kwargs()
            strat_params['symbol'] = symbol

            logger.info(f"  ✅ Loaded model: Sharpe={bundle.metadata.get('Sharpe', 0):.3f}")
        except FileNotFoundError:
            logger.warning(f"  ⚠️  No model found for {symbol}")
            return None

    # Load data
    try:
        setup_cerebro_with_data(
            cerebro,
            symbols=list(symbols_to_load),
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        logger.error(f"  ❌ Failed to load data for {symbol}: {e}")
        return None

    # Add strategy
    cerebro.addstrategy(SignalCaptureStrategy, **strat_params)

    # Run backtest
    try:
        results = cerebro.run()
        strat = results[0]

        # Extract signal log
        signal_df = pd.DataFrame(strat.signal_log)

        if len(signal_df) == 0:
            logger.warning(f"  ⚠️  No signals captured for {symbol}")
            return None

        signal_df['datetime'] = pd.to_datetime(signal_df['datetime'])
        signal_df = signal_df.set_index('datetime')

        # Calculate forward returns
        signal_df['returns'] = signal_df['close'].pct_change().shift(-1)

        logger.info(f"  📊 Captured {len(signal_df)} bars, "
                   f"{signal_df['signal'].sum()} signals")

        return signal_df

    except Exception as e:
        logger.error(f"  ❌ Backtest failed for {symbol}: {e}")
        return None


def combine_portfolio_signals(
    signals_dict: Dict[str, pd.DataFrame],
    max_positions: Optional[int] = None,
    ranking_method: str = 'equal'
) -> pd.DataFrame:
    """
    Combine signals from multiple symbols into portfolio signals.

    Parameters:
        signals_dict: {symbol: signal_df}
        max_positions: Maximum positions to hold at once
        ranking_method: How to rank signals when limiting positions

    Returns:
        DataFrame with columns for each symbol's signal (boolean)
    """
    # Create signals DataFrame
    all_signals = {}

    for symbol, df in signals_dict.items():
        all_signals[symbol] = df['signal']

    signals_df = pd.DataFrame(all_signals)

    # Apply max_positions constraint if specified
    if max_positions is not None:
        limited_signals = signals_df.copy()

        for idx in signals_df.index:
            active_signals = signals_df.loc[idx]
            active_symbols = active_signals[active_signals].index.tolist()

            if len(active_symbols) > max_positions:
                # For now, use simple first-N selection
                # Can be enhanced with probability-based ranking
                selected = active_symbols[:max_positions]

                limited_signals.loc[idx] = False
                limited_signals.loc[idx, selected] = True

        signals_df = limited_signals

    return signals_df


def simulate_portfolio_returns(
    signals_dict: Dict[str, pd.DataFrame],
    portfolio_signals: pd.DataFrame,
    commission_pct: float = 0.0001,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> Tuple[pd.Series, Dict]:
    """
    Simulate portfolio returns based on signals with daily stop loss.

    Parameters:
        signals_dict: {symbol: signal_df with returns}
        portfolio_signals: DataFrame of boolean signals per symbol
        commission_pct: Commission as percentage
        initial_capital: Starting capital in dollars
        daily_stop_loss: Maximum daily loss in dollars before stopping trading

    Returns:
        equity_curve: Series of portfolio equity
        metrics: Performance metrics dict
    """
    # Initialize
    equity = initial_capital
    equity_curve = [equity]
    dates = [portfolio_signals.index[0]]
    portfolio_returns = []

    current_day = None
    daily_pnl = 0.0
    stopped_out = False
    stop_count = 0

    for date in portfolio_signals.index:
        # Check if new trading day
        day = date.date() if hasattr(date, 'date') else date

        if current_day is None or day != current_day:
            current_day = day
            daily_pnl = 0.0
            stopped_out = False

        # If stopped out for the day, skip all trading
        if stopped_out:
            portfolio_returns.append(0.0)
            equity_curve.append(equity)
            dates.append(date)
            continue

        # Get active positions for this date
        active_signals = portfolio_signals.loc[date]
        active_symbols = active_signals[active_signals].index.tolist()

        if len(active_symbols) == 0:
            portfolio_returns.append(0.0)
            equity_curve.append(equity)
            dates.append(date)
            continue

        # Equal weight across active positions
        weight_per_symbol = 1.0 / len(active_symbols)

        # Calculate portfolio return
        period_return = 0.0
        period_pnl = 0.0

        for symbol in active_symbols:
            if symbol not in signals_dict:
                continue

            symbol_df = signals_dict[symbol]

            if date not in symbol_df.index:
                continue

            symbol_return = symbol_df.loc[date, 'returns']

            if pd.notna(symbol_return):
                # Apply commission
                net_return = symbol_return - commission_pct
                position_return = weight_per_symbol * net_return
                period_return += position_return

                # Calculate P&L in dollars
                position_pnl = position_return * equity
                period_pnl += position_pnl

        # Update daily P&L
        daily_pnl += period_pnl

        # Check if daily stop loss hit
        if daily_pnl <= -daily_stop_loss:
            stopped_out = True
            stop_count += 1
            # Exit immediately
            portfolio_returns.append(0.0)
            equity_curve.append(equity)
            dates.append(date)
            continue

        portfolio_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)
        dates.append(date)

    # Create series
    equity_series = pd.Series(equity_curve[1:], index=dates[1:])
    returns_series = pd.Series(portfolio_returns)

    # Calculate metrics
    metrics = calculate_performance_metrics(equity_series, returns_series, initial_capital, stop_count)

    return equity_series, metrics


def calculate_performance_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    initial_capital: float = 250000.0,
    stop_count: int = 0
) -> Dict:
    """Calculate portfolio performance metrics with dollar amounts."""
    if len(equity_curve) < 2:
        return {
            'total_return': 0,
            'total_return_dollars': 0,
            'cagr': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_dollars': 0,
            'profit_factor': 0,
            'n_periods': 0,
            'daily_stops_hit': 0
        }

    # Total return (percentage and dollars)
    total_return_pct = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    total_return_dollars = equity_curve.iloc[-1] - initial_capital

    # Annualized metrics
    periods_per_year = 252
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year

    # CAGR (Compound Annual Growth Rate)
    cagr = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0

    annualized_vol = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
    sharpe_ratio = cagr / annualized_vol if annualized_vol > 0 else 0

    # Win/Loss metrics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    # Gross profits and losses for profit factor
    gross_profits = positive_returns.sum() * initial_capital if len(positive_returns) > 0 else 0
    gross_losses = abs(negative_returns.sum() * initial_capital) if len(negative_returns) > 0 else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    # Max drawdown (percentage and dollars)
    cummax = equity_curve.expanding().max()
    drawdown = equity_curve - cummax
    drawdown_pct = drawdown / cummax
    max_drawdown_pct = drawdown_pct.min()
    max_drawdown_dollars = drawdown.min()

    return {
        'total_return': total_return_pct,
        'total_return_dollars': total_return_dollars,
        'cagr': cagr,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown_pct,
        'max_drawdown_dollars': max_drawdown_dollars,
        'profit_factor': profit_factor,
        'gross_profits': gross_profits,
        'gross_losses': gross_losses,
        'n_periods': n_periods,
        'daily_stops_hit': stop_count
    }


def optimize_portfolio(
    signals_dict: Dict[str, pd.DataFrame],
    min_positions: int = 1,
    max_positions: Optional[int] = None,
    commission_pct: float = 0.0001,
    initial_capital: float = 250000.0,
    daily_stop_loss: float = 2500.0
) -> pd.DataFrame:
    """
    Optimize max_positions parameter with daily stop loss.

    Returns:
        DataFrame with optimization results
    """
    n_symbols = len(signals_dict)
    max_pos = max_positions or n_symbols

    results = []

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZING MAX POSITIONS: {min_positions} to {max_pos}")
    logger.info(f"Initial Capital: ${initial_capital:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    logger.info(f"{'='*80}\n")

    for max_pos_val in range(min_positions, max_pos + 1):
        logger.info(f"Testing max_positions = {max_pos_val}")

        # Combine signals with constraint
        portfolio_signals = combine_portfolio_signals(
            signals_dict,
            max_positions=max_pos_val
        )

        # Simulate portfolio
        equity, metrics = simulate_portfolio_returns(
            signals_dict,
            portfolio_signals,
            commission_pct=commission_pct,
            initial_capital=initial_capital,
            daily_stop_loss=daily_stop_loss
        )

        # Store results
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
    best = results_df.iloc[0]
    logger.info(f"\n🏆 OPTIMAL CONFIGURATION:")
    logger.info(f"   Max Positions: {int(best['max_positions'])}")
    logger.info(f"   Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    logger.info(f"   CAGR: {best['cagr']*100:.2f}%")
    logger.info(f"   Total Return: ${best['total_return_dollars']:,.2f}")
    logger.info(f"   Max Drawdown: ${best['max_drawdown_dollars']:,.2f} ({best['max_drawdown']*100:.2f}%)")
    logger.info(f"   Profit Factor: {best['profit_factor']:.2f}")
    logger.info(f"   Daily Stops Hit: {best['daily_stops_hit']:.0f}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio optimizer with full backtest integration'
    )

    parser.add_argument('--symbols', nargs='+', help='Symbols to include')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data/resampled')
    parser.add_argument('--models-dir', type=str, default='src/models')
    parser.add_argument('--min-positions', type=int, default=1)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--commission-pct', type=float, default=0.0001)
    parser.add_argument('--initial-cash', type=float, default=250000.0)
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0,
                       help='Daily stop loss in dollars (default: 2500)')
    parser.add_argument('--output', type=str, help='Output CSV file')
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
        logger.info(f"Auto-discovered {len(symbols)} symbols: {', '.join(sorted(symbols))}")

    if not symbols:
        logger.error("No symbols to process")
        return

    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    # Run backtests for all symbols
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BACKTESTS: {args.start} to {end_date}")
    logger.info(f"{'='*80}\n")

    signals_dict = {}

    for symbol in symbols:
        signal_df = run_backtest_with_signals(
            symbol,
            args.start,
            end_date,
            data_dir=args.data_dir,
            initial_cash=args.initial_cash,
            use_ml=not args.no_ml
        )

        if signal_df is not None:
            signals_dict[symbol] = signal_df

    if not signals_dict:
        logger.error("No signals extracted. Check data and models.")
        return

    logger.info(f"\n✅ Successfully extracted signals from {len(signals_dict)} symbols")

    # Run optimization
    results_df = optimize_portfolio(
        signals_dict,
        min_positions=args.min_positions,
        max_positions=args.max_positions,
        commission_pct=args.commission_pct,
        initial_capital=args.initial_cash,
        daily_stop_loss=args.daily_stop_loss
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
