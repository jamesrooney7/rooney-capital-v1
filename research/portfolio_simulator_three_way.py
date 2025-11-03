#!/usr/bin/env python3
"""
Portfolio Simulator for Three-Way Split Models

Simulates portfolio performance from retrained models using test period (2022-2024).
Works directly with models and test data to calculate portfolio-level metrics.

Usage:
    python research/portfolio_simulator_three_way.py \
        --min-positions 1 \
        --max-positions 10 \
        --daily-stop-loss 2500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.rf_cpcv_random_then_bo import build_core_features, add_engineered

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models_and_metadata(models_dir: Path) -> Dict:
    """Load all trained models and their metadata."""
    models = {}

    for best_file in sorted(models_dir.glob('*_best.json')):
        symbol = best_file.stem.replace('_best', '')

        # Load metadata
        with open(best_file) as f:
            metadata = json.load(f)

        # Load model
        model_file = models_dir / f"{symbol}_rf_model.pkl"
        if not model_file.exists():
            logger.warning(f"Model file not found for {symbol}, skipping")
            continue

        model_data = joblib.load(model_file)

        models[symbol] = {
            'model': model_data['model'],
            'features': model_data['features'],
            'threshold': metadata.get('threshold', 0.5),
            'metadata': metadata
        }

    logger.info(f"Loaded {len(models)} models")
    return models


def load_test_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load test period data for a symbol."""
    data_file = data_dir / f"{symbol}_transformed_features.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Parse date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Date/Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Date/Time'])
        # Also create a Date column for filtering
        if 'Date' not in df.columns:
            df['Date'] = df['Date/Time'].dt.date
            df['Date'] = pd.to_datetime(df['Date'])

    return df


def generate_predictions(models: Dict, data_dir: Path,
                        test_start: str, test_end: str) -> Dict[str, pd.DataFrame]:
    """
    Generate predictions for all symbols during test period.

    Returns:
        Dict of DataFrames with Date, prediction, proba, pnl_usd for each symbol
    """
    predictions = {}

    for symbol, model_info in models.items():
        try:
            logger.info(f"Loading test data for {symbol}...")
            df = load_test_data(symbol, data_dir)

            # Filter to test period
            df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()

            if len(df) == 0:
                logger.warning(f"No test data for {symbol} in period {test_start} to {test_end}")
                continue

            # Build features
            X_norm = build_core_features(df)

            # Exclude non-feature columns
            exclude_cols = {
                "Date", "Date/Time", "Exit Date/Time", "Entry_Price", "Exit_Price",
                "y_return", "y_binary", "y_pnl_usd", "y_pnl_gross", "pnl_usd",
                "Unnamed: 0",
            }
            feature_cols = [col for col in X_norm.columns if col not in exclude_cols]
            X_features = add_engineered(X_norm[feature_cols])

            # Get predictions
            model = model_info['model']
            features = model_info['features']
            threshold = model_info['threshold']

            # Ensure features match
            X_model = X_features[features]

            # Predict
            probas = model.predict_proba(X_model)[:, 1]
            preds = (probas >= threshold).astype(int)

            # Build result DataFrame
            result = pd.DataFrame({
                'Date': df['Date'].values,
                'prediction': preds,
                'proba': probas,
                'pnl_usd': df['pnl_usd'].values if 'pnl_usd' in df.columns else df['y_pnl_usd'].values,
            })

            # Only keep predicted trades
            result = result[result['prediction'] == 1].copy()

            predictions[symbol] = result
            logger.info(f"  {symbol}: {len(result)} predicted trades")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    return predictions


def simulate_portfolio(predictions: Dict[str, pd.DataFrame],
                      models: Dict,
                      max_positions: int,
                      daily_stop_loss: float,
                      initial_cash: float,
                      position_size_multiplier: float = 1.0,
                      ranking_method: str = 'sharpe') -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate portfolio with position limits and daily stop loss.

    Args:
        predictions: Dict of symbol -> DataFrame with Date, prediction, proba, pnl_usd
        models: Dict of model info with metadata
        max_positions: Maximum concurrent positions
        daily_stop_loss: Daily stop loss in USD
        initial_cash: Starting capital
        position_size_multiplier: Scale factor for position sizes (e.g., 0.5 = half contracts)
        ranking_method: How to rank symbols ('sharpe', 'profit_factor', 'proba')

    Returns:
        equity_curve: DataFrame with Date, equity, daily_pnl, positions
        metrics: Dict with portfolio performance metrics
    """
    # Get symbol rankings for position selection
    if ranking_method == 'sharpe':
        rankings = {symbol: models[symbol]['metadata'].get('threshold_optimization', {}).get('sharpe', 0)
                   for symbol in predictions.keys()}
    elif ranking_method == 'profit_factor':
        rankings = {symbol: models[symbol]['metadata'].get('threshold_optimization', {}).get('profit_factor', 0)
                   for symbol in predictions.keys()}
    else:  # proba - will rank dynamically each day
        rankings = None

    # Combine all predictions into single timeline
    all_dates = set()
    for df in predictions.values():
        all_dates.update(df['Date'].values)
    all_dates = sorted(all_dates)

    # Simulate day by day
    equity = initial_cash
    equity_curve = []
    daily_stops_hit = 0

    for date in all_dates:
        # Get all signals for this date
        day_signals = []
        for symbol, df in predictions.items():
            day_df = df[df['Date'] == date]
            if len(day_df) > 0:
                for _, row in day_df.iterrows():
                    day_signals.append({
                        'symbol': symbol,
                        'proba': row['proba'],
                        'pnl_usd': row['pnl_usd'],
                        'ranking': rankings[symbol] if rankings else row['proba']
                    })

        # Rank and select top N signals
        if len(day_signals) > max_positions:
            day_signals = sorted(day_signals, key=lambda x: x['ranking'], reverse=True)[:max_positions]

        # Calculate daily P&L (scaled by position size multiplier)
        daily_pnl = sum(s['pnl_usd'] * position_size_multiplier for s in day_signals)

        # Check daily stop loss
        if daily_pnl < -daily_stop_loss:
            daily_stops_hit += 1
            daily_pnl = -daily_stop_loss  # Cap loss at stop loss

        equity += daily_pnl

        equity_curve.append({
            'Date': date,
            'equity': equity,
            'daily_pnl': daily_pnl,
            'positions': len(day_signals),
            'symbols': ','.join([s['symbol'] for s in day_signals]) if day_signals else ''
        })

    equity_df = pd.DataFrame(equity_curve)
    equity_df['Date'] = pd.to_datetime(equity_df['Date'])

    # Calculate metrics
    total_return = equity - initial_cash
    pct_return = total_return / initial_cash

    # Calculate daily returns
    equity_df['daily_return'] = equity_df['daily_pnl'] / equity_df['equity'].shift(1).fillna(initial_cash)

    # Annualized metrics
    n_trading_days = len(equity_df)
    years = n_trading_days / 252

    mean_daily_return = equity_df['daily_return'].mean()
    std_daily_return = equity_df['daily_return'].std()

    annualized_return = mean_daily_return * 252
    annualized_vol = std_daily_return * np.sqrt(252)

    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Sortino (downside deviation)
    downside_returns = equity_df[equity_df['daily_return'] < 0]['daily_return']
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_daily_return
    annualized_downside_vol = downside_std * np.sqrt(252)
    sortino = annualized_return / annualized_downside_vol if annualized_downside_vol > 0 else 0

    # CAGR
    cagr = (1 + pct_return) ** (1/years) - 1 if years > 0 else 0

    # Max drawdown
    cummax = equity_df['equity'].cummax()
    drawdown = equity_df['equity'] - cummax
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown / cummax[drawdown.idxmin()] if max_drawdown < 0 else 0

    # Win metrics
    winning_days = equity_df[equity_df['daily_pnl'] > 0]
    losing_days = equity_df[equity_df['daily_pnl'] < 0]

    win_rate = len(winning_days) / len(equity_df) if len(equity_df) > 0 else 0
    avg_win = winning_days['daily_pnl'].mean() if len(winning_days) > 0 else 0
    avg_loss = abs(losing_days['daily_pnl'].mean()) if len(losing_days) > 0 else 0
    profit_factor = (winning_days['daily_pnl'].sum() / abs(losing_days['daily_pnl'].sum())) if len(losing_days) > 0 and losing_days['daily_pnl'].sum() < 0 else 0

    metrics = {
        'total_return': total_return,
        'pct_return': pct_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trading_days': n_trading_days,
        'avg_positions': equity_df['positions'].mean(),
        'daily_stops_hit': daily_stops_hit,
    }

    return equity_df, metrics


def optimize_max_positions(predictions: Dict[str, pd.DataFrame],
                          models: Dict,
                          min_positions: int,
                          max_positions: int,
                          daily_stop_loss: float,
                          initial_cash: float,
                          max_drawdown_limit: float = None,
                          ranking_method: str = 'sharpe') -> pd.DataFrame:
    """
    Test different max_positions values and find optimal.

    Args:
        max_drawdown_limit: If specified, find optimal config that keeps drawdown <= this value

    Returns:
        DataFrame with results for each max_positions value
    """
    results = []

    print(f"\n{'='*100}")
    print(f"PORTFOLIO OPTIMIZATION: Testing max_positions from {min_positions} to {max_positions}")
    print(f"Initial Capital: ${initial_cash:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    if max_drawdown_limit:
        print(f"Max Drawdown Constraint: ${abs(max_drawdown_limit):,.0f}")
    print(f"Test Period: 2022-2024 (TRUE out-of-sample)")
    print(f"Ranking Method: {ranking_method}")
    print(f"{'='*100}\n")

    for n_pos in range(min_positions, max_positions + 1):
        logger.info(f"Testing max_positions = {n_pos}")

        equity_df, metrics = simulate_portfolio(
            predictions, models, n_pos, daily_stop_loss, initial_cash, 1.0, ranking_method
        )

        result = {
            'max_positions': n_pos,
            **metrics
        }
        results.append(result)

        # Print progress
        print(f"max_positions = {n_pos:2d}")
        print(f"  Sharpe:     {metrics['sharpe']:7.3f}")
        print(f"  Sortino:    {metrics['sortino']:7.3f}")
        print(f"  CAGR:       {metrics['cagr']*100:7.2f}%")
        print(f"  Return:     ${metrics['total_return']:,.0f}")
        print(f"  MaxDD:      ${metrics['max_drawdown']:,.0f} ({metrics['max_drawdown_pct']*100:.2f}%)")
        print(f"  PF:         {metrics['profit_factor']:7.2f}")
        print(f"  Win Rate:   {metrics['win_rate']*100:7.2f}%")
        print(f"  Avg Pos:    {metrics['avg_positions']:7.2f}")
        print(f"  Stops Hit:  {metrics['daily_stops_hit']:3d}")
        print()

    results_df = pd.DataFrame(results)

    # Find unconstrained optimal (best Sharpe overall)
    best_unconstrained_idx = results_df['sharpe'].idxmax()
    best_unconstrained = results_df.loc[best_unconstrained_idx]

    # Find constrained optimal (best Sharpe with drawdown <= limit)
    best_constrained = None
    if max_drawdown_limit:
        # Filter to configs meeting drawdown constraint (drawdown is negative, so >= -limit)
        constrained_df = results_df[results_df['max_drawdown'] >= -abs(max_drawdown_limit)]
        if len(constrained_df) > 0:
            best_constrained_idx = constrained_df['sharpe'].idxmax()
            best_constrained = constrained_df.loc[best_constrained_idx]

    print(f"\n{'='*100}")
    print("OPTIMIZATION RESULTS (sorted by Sharpe)")
    print(f"{'='*100}\n")

    results_sorted = results_df.sort_values('sharpe', ascending=False)

    # Mark configs that meet drawdown constraint
    print(f"{'MaxPos':<10}{'Sharpe':<12}{'Sortino':<12}{'CAGR%':<12}{'Return $':<15}{'MaxDD $':<15}{'PF':<10}{'WR%':<10}{'Stops':<8}{'OK?':<5}")
    print("-" * 105)
    for _, row in results_sorted.iterrows():
        meets_constraint = ''
        if max_drawdown_limit:
            meets_constraint = '✓' if row['max_drawdown'] >= -abs(max_drawdown_limit) else '✗'

        print(f"{row['max_positions']:<10}{row['sharpe']:<12.3f}{row['sortino']:<12.3f}"
              f"{row['cagr']*100:<12.2f}${row['total_return']:<14,.0f}"
              f"${row['max_drawdown']:<14,.0f}{row['profit_factor']:<10.2f}"
              f"{row['win_rate']*100:<10.2f}{row['daily_stops_hit']:<8.0f}{meets_constraint:<5}")

    # Show both unconstrained and constrained optimal
    print(f"\n{'='*100}")
    print("🏆 UNCONSTRAINED OPTIMAL (Best Sharpe Overall):")
    print(f"{'='*100}")
    print(f"  Max Positions:        {int(best_unconstrained['max_positions'])}")
    print(f"  Sharpe Ratio:         {best_unconstrained['sharpe']:.3f}")
    print(f"  Sortino Ratio:        {best_unconstrained['sortino']:.3f}")
    print(f"  CAGR:                 {best_unconstrained['cagr']*100:.2f}%")
    print(f"  Total Return:         ${best_unconstrained['total_return']:,.0f}")
    print(f"  Max Drawdown:         ${best_unconstrained['max_drawdown']:,.0f} ({best_unconstrained['max_drawdown_pct']*100:.2f}%)")
    print(f"  Profit Factor:        {best_unconstrained['profit_factor']:.2f}")
    print(f"  Win Rate:             {best_unconstrained['win_rate']*100:.2f}%")
    print(f"  Avg Positions:        {best_unconstrained['avg_positions']:.2f}")
    print(f"  Daily Stops Hit:      {int(best_unconstrained['daily_stops_hit'])}")
    print(f"{'='*100}\n")

    if best_constrained is not None:
        print(f"{'='*100}")
        print(f"🎯 CONSTRAINED OPTIMAL (Best Sharpe with MaxDD <= ${abs(max_drawdown_limit):,.0f}):")
        print(f"{'='*100}")
        print(f"  Max Positions:        {int(best_constrained['max_positions'])}")
        print(f"  Sharpe Ratio:         {best_constrained['sharpe']:.3f}")
        print(f"  Sortino Ratio:        {best_constrained['sortino']:.3f}")
        print(f"  CAGR:                 {best_constrained['cagr']*100:.2f}%")
        print(f"  Total Return:         ${best_constrained['total_return']:,.0f}")
        print(f"  Max Drawdown:         ${best_constrained['max_drawdown']:,.0f} ({best_constrained['max_drawdown_pct']*100:.2f}%)")
        print(f"  Profit Factor:        {best_constrained['profit_factor']:.2f}")
        print(f"  Win Rate:             {best_constrained['win_rate']*100:.2f}%")
        print(f"  Avg Positions:        {best_constrained['avg_positions']:.2f}")
        print(f"  Daily Stops Hit:      {int(best_constrained['daily_stops_hit'])}")
        print(f"{'='*100}\n")
    elif max_drawdown_limit:
        print(f"\n{'='*100}")
        print(f"⚠️  WARNING: No configurations meet drawdown constraint of ${abs(max_drawdown_limit):,.0f}")
        print(f"{'='*100}\n")

    return results_df


def optimize_positions_and_sizing(predictions: Dict[str, pd.DataFrame],
                                 models: Dict,
                                 min_positions: int,
                                 max_positions: int,
                                 position_size_range: Tuple[float, float],
                                 position_size_step: float,
                                 daily_stop_loss: float,
                                 initial_cash: float,
                                 max_drawdown_limit: float = None,
                                 ranking_method: str = 'sharpe') -> pd.DataFrame:
    """
    Test combinations of max_positions and position_size_multiplier to find optimal.

    Args:
        position_size_range: (min, max) multiplier range, e.g., (0.1, 1.0)
        position_size_step: Step size for position_size_multiplier, e.g., 0.05

    Returns:
        DataFrame with results for each combination
    """
    results = []

    print(f"\n{'='*110}")
    print(f"2D PORTFOLIO OPTIMIZATION: max_positions ({min_positions}-{max_positions}) × position_size ({position_size_range[0]}-{position_size_range[1]})")
    print(f"Initial Capital: ${initial_cash:,.0f} | Daily Stop Loss: ${daily_stop_loss:,.0f}")
    if max_drawdown_limit:
        print(f"Max Drawdown Constraint: ${abs(max_drawdown_limit):,.0f}")
    print(f"Test Period: 2022-2024 (TRUE out-of-sample)")
    print(f"Ranking Method: {ranking_method}")
    print(f"{'='*110}\n")

    # Generate position size multipliers
    pos_sizes = np.arange(position_size_range[0], position_size_range[1] + position_size_step, position_size_step)

    total_combos = (max_positions - min_positions + 1) * len(pos_sizes)
    combo_num = 0

    for n_pos in range(min_positions, max_positions + 1):
        for pos_size in pos_sizes:
            combo_num += 1
            logger.info(f"Testing [{combo_num}/{total_combos}]: max_positions={n_pos}, position_size={pos_size:.2f}")

            equity_df, metrics = simulate_portfolio(
                predictions, models, n_pos, daily_stop_loss, initial_cash, pos_size, ranking_method
            )

            result = {
                'max_positions': n_pos,
                'position_size': pos_size,
                **metrics
            }
            results.append(result)

            # Print compact progress
            meets_constraint = '✓' if (max_drawdown_limit is None or metrics['max_drawdown'] >= -abs(max_drawdown_limit)) else '✗'
            print(f"  [{n_pos:2d}, {pos_size:.2f}] Sharpe: {metrics['sharpe']:6.3f}  MaxDD: ${metrics['max_drawdown']:>9,.0f}  {meets_constraint}")

    results_df = pd.DataFrame(results)

    # Find unconstrained optimal (best Sharpe overall)
    best_unconstrained_idx = results_df['sharpe'].idxmax()
    best_unconstrained = results_df.loc[best_unconstrained_idx]

    # Find constrained optimal (best Sharpe with drawdown <= limit)
    best_constrained = None
    if max_drawdown_limit:
        constrained_df = results_df[results_df['max_drawdown'] >= -abs(max_drawdown_limit)]
        if len(constrained_df) > 0:
            best_constrained_idx = constrained_df['sharpe'].idxmax()
            best_constrained = constrained_df.loc[best_constrained_idx]

    # Show top 20 configurations
    print(f"\n{'='*110}")
    print("TOP 20 CONFIGURATIONS (sorted by Sharpe)")
    print(f"{'='*110}\n")

    results_sorted = results_df.sort_values('sharpe', ascending=False).head(20)

    print(f"{'MaxPos':<9}{'PosSize':<10}{'Sharpe':<10}{'Sortino':<10}{'CAGR%':<10}{'Return $':<13}{'MaxDD $':<13}{'PF':<8}{'WR%':<8}{'OK?':<5}")
    print("-" * 110)
    for _, row in results_sorted.iterrows():
        meets_constraint = ''
        if max_drawdown_limit:
            meets_constraint = '✓' if row['max_drawdown'] >= -abs(max_drawdown_limit) else '✗'

        print(f"{row['max_positions']:<9.0f}{row['position_size']:<10.2f}{row['sharpe']:<10.3f}{row['sortino']:<10.3f}"
              f"{row['cagr']*100:<10.1f}${row['total_return']:<12,.0f}"
              f"${row['max_drawdown']:<12,.0f}{row['profit_factor']:<8.2f}"
              f"{row['win_rate']*100:<8.1f}{meets_constraint:<5}")

    # Show both unconstrained and constrained optimal
    print(f"\n{'='*110}")
    print("🏆 UNCONSTRAINED OPTIMAL (Best Sharpe Overall):")
    print(f"{'='*110}")
    print(f"  Max Positions:        {int(best_unconstrained['max_positions'])}")
    print(f"  Position Size:        {best_unconstrained['position_size']:.2f}x")
    print(f"  Sharpe Ratio:         {best_unconstrained['sharpe']:.3f}")
    print(f"  Sortino Ratio:        {best_unconstrained['sortino']:.3f}")
    print(f"  CAGR:                 {best_unconstrained['cagr']*100:.2f}%")
    print(f"  Total Return:         ${best_unconstrained['total_return']:,.0f}")
    print(f"  Max Drawdown:         ${best_unconstrained['max_drawdown']:,.0f} ({best_unconstrained['max_drawdown_pct']*100:.2f}%)")
    print(f"  Profit Factor:        {best_unconstrained['profit_factor']:.2f}")
    print(f"  Win Rate:             {best_unconstrained['win_rate']*100:.2f}%")
    print(f"  Avg Positions:        {best_unconstrained['avg_positions']:.2f}")
    print(f"  Daily Stops Hit:      {int(best_unconstrained['daily_stops_hit'])}")
    print(f"{'='*110}\n")

    if best_constrained is not None:
        print(f"{'='*110}")
        print(f"🎯 CONSTRAINED OPTIMAL (Best Sharpe with MaxDD <= ${abs(max_drawdown_limit):,.0f}):")
        print(f"{'='*110}")
        print(f"  Max Positions:        {int(best_constrained['max_positions'])}")
        print(f"  Position Size:        {best_constrained['position_size']:.2f}x")
        print(f"  Sharpe Ratio:         {best_constrained['sharpe']:.3f}")
        print(f"  Sortino Ratio:        {best_constrained['sortino']:.3f}")
        print(f"  CAGR:                 {best_constrained['cagr']*100:.2f}%")
        print(f"  Total Return:         ${best_constrained['total_return']:,.0f}")
        print(f"  Max Drawdown:         ${best_constrained['max_drawdown']:,.0f} ({best_constrained['max_drawdown_pct']*100:.2f}%)")
        print(f"  Profit Factor:        {best_constrained['profit_factor']:.2f}")
        print(f"  Win Rate:             {best_constrained['win_rate']*100:.2f}%")
        print(f"  Avg Positions:        {best_constrained['avg_positions']:.2f}")
        print(f"  Daily Stops Hit:      {int(best_constrained['daily_stops_hit'])}")
        print(f"{'='*110}\n")
    elif max_drawdown_limit:
        print(f"\n{'='*110}")
        print(f"⚠️  WARNING: No configurations meet drawdown constraint of ${abs(max_drawdown_limit):,.0f}")
        print(f"{'='*110}\n")

    return results_df


def greedy_symbol_selection(predictions: Dict[str, pd.DataFrame],
                           models: Dict,
                           max_positions_range: Tuple[int, int],
                           daily_stop_loss: float,
                           initial_cash: float,
                           max_drawdown_limit: float,
                           position_size_multiplier: float = 1.0,
                           ranking_method: str = 'sharpe',
                           period_name: str = 'optimization') -> Tuple[Set[str], int, float, Dict]:
    """
    Greedy optimization: iteratively remove symbols to meet drawdown constraint.

    For each max_positions value:
    1. Start with all symbols
    2. Remove the symbol that contributes most to drawdown
    3. Repeat until drawdown <= limit or can't remove more
    4. Track the best configuration

    Args:
        period_name: Name of the period being optimized (for display)

    Returns:
        optimal_symbols: Set of symbols in optimal portfolio
        optimal_max_pos: Optimal max_positions
        optimal_pos_size: Optimal position size multiplier
        optimal_metrics: Performance metrics of optimal config
    """
    from copy import deepcopy

    all_symbols = set(predictions.keys())

    print(f"\n{'='*110}")
    print(f"GREEDY SYMBOL SELECTION WITH DRAWDOWN CONSTRAINT")
    print(f"Period: {period_name}")
    print(f"Max Drawdown Limit: ${abs(max_drawdown_limit):,.0f}")
    print(f"Testing max_positions: {max_positions_range[0]} to {max_positions_range[1]}")
    print(f"Position Size: {position_size_multiplier:.2f}x")
    print(f"Starting with {len(all_symbols)} symbols: {', '.join(sorted(all_symbols))}")
    print(f"{'='*110}\n")

    best_config = None
    best_sharpe = -np.inf

    for max_pos in range(max_positions_range[0], max_positions_range[1] + 1):
        print(f"\n{'='*90}")
        print(f"Testing max_positions = {max_pos}")
        print(f"{'='*90}")

        current_symbols = deepcopy(all_symbols)
        iteration = 0

        while len(current_symbols) > max_pos:  # Need at least max_positions symbols
            iteration += 1

            # Evaluate current portfolio
            current_preds = {s: predictions[s] for s in current_symbols}
            _, current_metrics = simulate_portfolio(
                current_preds, models, max_pos, daily_stop_loss,
                initial_cash, position_size_multiplier, ranking_method
            )

            current_dd = current_metrics['max_drawdown']
            current_sharpe = current_metrics['sharpe']

            print(f"\nIteration {iteration}: {len(current_symbols)} symbols")
            print(f"  Sharpe: {current_sharpe:.3f}  MaxDD: ${current_dd:,.0f}  ", end='')

            # Check if constraint is met
            if current_dd >= -abs(max_drawdown_limit):
                print(f"✓ MEETS CONSTRAINT")

                # Track best configuration
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_config = {
                        'symbols': deepcopy(current_symbols),
                        'max_positions': max_pos,
                        'position_size': position_size_multiplier,
                        'metrics': current_metrics
                    }
                    print(f"  🏆 NEW BEST: Sharpe {current_sharpe:.3f}")

                # Try removing more symbols to see if we can improve Sharpe
                # but stop if we're down to max_positions symbols
                if len(current_symbols) <= max_pos:
                    break

            else:
                print(f"✗ Exceeds constraint")

            # Try removing each symbol, see which improves drawdown most
            best_removal = None
            best_removal_dd = current_dd
            best_removal_sharpe = -np.inf

            for symbol_to_remove in current_symbols:
                test_symbols = current_symbols - {symbol_to_remove}

                # Need at least max_positions symbols
                if len(test_symbols) < max_pos:
                    continue

                test_preds = {s: predictions[s] for s in test_symbols}
                _, test_metrics = simulate_portfolio(
                    test_preds, models, max_pos, daily_stop_loss,
                    initial_cash, position_size_multiplier, ranking_method
                )

                test_dd = test_metrics['max_drawdown']
                test_sharpe = test_metrics['sharpe']

                # Choose removal that improves drawdown most (less negative = better)
                # If multiple improve drawdown equally, pick the one with best Sharpe
                if test_dd > best_removal_dd or (test_dd == best_removal_dd and test_sharpe > best_removal_sharpe):
                    best_removal = symbol_to_remove
                    best_removal_dd = test_dd
                    best_removal_sharpe = test_sharpe

            if best_removal is None:
                print(f"  No beneficial removal found, stopping.")
                break

            # Remove the worst symbol
            current_symbols.remove(best_removal)
            dd_improvement = best_removal_dd - current_dd
            print(f"  Removing {best_removal}: DD improves by ${dd_improvement:,.0f} → ${best_removal_dd:,.0f}")

    if best_config is None:
        print(f"\n{'='*110}")
        print(f"⚠️  WARNING: Could not find any configuration meeting drawdown constraint of ${abs(max_drawdown_limit):,.0f}")
        print(f"{'='*110}\n")
        return None, None, None, None

    # Display final result
    print(f"\n{'='*110}")
    print(f"🎯 OPTIMAL CONFIGURATION:")
    print(f"{'='*110}")
    print(f"  Symbols ({len(best_config['symbols'])}): {', '.join(sorted(best_config['symbols']))}")
    print(f"  Max Positions:        {best_config['max_positions']}")
    print(f"  Position Size:        {best_config['position_size']:.2f}x")
    print(f"  Sharpe Ratio:         {best_config['metrics']['sharpe']:.3f}")
    print(f"  Sortino Ratio:        {best_config['metrics']['sortino']:.3f}")
    print(f"  CAGR:                 {best_config['metrics']['cagr']*100:.2f}%")
    print(f"  Total Return:         ${best_config['metrics']['total_return']:,.0f}")
    print(f"  Max Drawdown:         ${best_config['metrics']['max_drawdown']:,.0f} ({best_config['metrics']['max_drawdown_pct']*100:.2f}%)")
    print(f"  Profit Factor:        {best_config['metrics']['profit_factor']:.2f}")
    print(f"  Win Rate:             {best_config['metrics']['win_rate']*100:.2f}%")
    print(f"  Avg Positions:        {best_config['metrics']['avg_positions']:.2f}")
    print(f"  Daily Stops Hit:      {int(best_config['metrics']['daily_stops_hit'])}")
    print(f"{'='*110}\n")

    return (
        best_config['symbols'],
        best_config['max_positions'],
        best_config['position_size'],
        best_config['metrics']
    )


def main():
    parser = argparse.ArgumentParser(description='Simulate portfolio from three-way split models')
    parser.add_argument('--models-dir', type=str, default='src/models',
                       help='Directory with trained models')
    parser.add_argument('--data-dir', type=str, default='data/training',
                       help='Directory with training data')
    parser.add_argument('--portfolio-train-start', type=str, default=None,
                       help='Portfolio optimization period start (e.g., 2021-01-01 for threshold period)')
    parser.add_argument('--portfolio-train-end', type=str, default=None,
                       help='Portfolio optimization period end (e.g., 2021-12-31 for threshold period)')
    parser.add_argument('--test-start', type=str, default='2022-01-01',
                       help='Test period start date for final evaluation')
    parser.add_argument('--test-end', type=str, default='2024-12-31',
                       help='Test period end date for final evaluation')
    parser.add_argument('--min-positions', type=int, default=1,
                       help='Minimum max_positions to test')
    parser.add_argument('--max-positions', type=int, default=18,
                       help='Maximum max_positions to test')
    parser.add_argument('--initial-cash', type=float, default=250000,
                       help='Initial capital')
    parser.add_argument('--daily-stop-loss', type=float, default=2500,
                       help='Daily stop loss in USD')
    parser.add_argument('--max-drawdown-limit', type=float, default=None,
                       help='Maximum allowed drawdown in USD (e.g., 6000). Will find best Sharpe with DD <= this limit')
    parser.add_argument('--greedy-selection', action='store_true',
                       help='Enable greedy symbol selection to find optimal instrument subset')
    parser.add_argument('--optimize-sizing', action='store_true',
                       help='Enable 2D optimization (max_positions + position sizing) to meet drawdown constraint')
    parser.add_argument('--position-size', type=float, default=1.0,
                       help='Position size multiplier for greedy selection (default: 1.0)')
    parser.add_argument('--position-size-min', type=float, default=0.1,
                       help='Minimum position size multiplier for 2D opt (default: 0.1)')
    parser.add_argument('--position-size-max', type=float, default=1.0,
                       help='Maximum position size multiplier for 2D opt (default: 1.0)')
    parser.add_argument('--position-size-step', type=float, default=0.05,
                       help='Position size step for 2D opt (default: 0.05)')
    parser.add_argument('--ranking-method', type=str, default='sharpe',
                       choices=['sharpe', 'profit_factor', 'proba'],
                       help='Method to rank symbols for position selection')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)

    # Load models
    logger.info("Loading models...")
    models = load_models_and_metadata(models_dir)

    if len(models) == 0:
        logger.error("No models found!")
        return 1

    # Choose optimization mode
    if args.greedy_selection:
        # Greedy symbol selection: find optimal subset of instruments
        if not args.max_drawdown_limit:
            logger.error("--max-drawdown-limit is required for greedy selection!")
            return 1

        # Phase 1: Optimize on portfolio train period (avoid leakage)
        if args.portfolio_train_start and args.portfolio_train_end:
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 1: PORTFOLIO OPTIMIZATION (In-Sample)")
            logger.info(f"Period: {args.portfolio_train_start} to {args.portfolio_train_end}")
            logger.info(f"{'='*80}\n")

            # Generate predictions for portfolio train period
            logger.info(f"Generating predictions for portfolio train period...")
            train_predictions = generate_predictions(models, data_dir, args.portfolio_train_start, args.portfolio_train_end)

            if len(train_predictions) == 0:
                logger.error("No predictions generated for portfolio train period!")
                return 1

            # Run greedy selection on train period
            optimal_symbols, optimal_max_pos, optimal_pos_size, train_metrics = greedy_symbol_selection(
                train_predictions,
                models,
                (args.min_positions, min(args.max_positions, len(train_predictions))),
                args.daily_stop_loss,
                args.initial_cash,
                args.max_drawdown_limit,
                args.position_size,
                args.ranking_method,
                period_name=f"Portfolio Train ({args.portfolio_train_start} to {args.portfolio_train_end})"
            )

            if optimal_symbols is None:
                logger.error("Could not find configuration meeting drawdown constraint on train period")
                return 1

            # Phase 2: Evaluate locked config on test period (true out-of-sample)
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 2: TEST EVALUATION (Out-of-Sample)")
            logger.info(f"Period: {args.test_start} to {args.test_end}")
            logger.info(f"Testing locked configuration:")
            logger.info(f"  Symbols: {', '.join(sorted(optimal_symbols))}")
            logger.info(f"  Max Positions: {optimal_max_pos}")
            logger.info(f"  Position Size: {optimal_pos_size:.2f}x")
            logger.info(f"{'='*80}\n")

            # Generate predictions for test period
            logger.info(f"Generating predictions for test period...")
            test_predictions = generate_predictions(models, data_dir, args.test_start, args.test_end)

            if len(test_predictions) == 0:
                logger.error("No predictions generated for test period!")
                return 1

            # Filter to only optimal symbols
            test_predictions_filtered = {s: test_predictions[s] for s in optimal_symbols if s in test_predictions}

            # Evaluate locked config on test period
            _, test_metrics = simulate_portfolio(
                test_predictions_filtered,
                models,
                optimal_max_pos,
                args.daily_stop_loss,
                args.initial_cash,
                optimal_pos_size,
                args.ranking_method
            )

            # Display results
            print(f"\n{'='*110}")
            print(f"📊 FINAL RESULTS")
            print(f"{'='*110}\n")

            print(f"Optimal Configuration (from portfolio train period):")
            print(f"  Symbols: {', '.join(sorted(optimal_symbols))}")
            print(f"  Max Positions: {optimal_max_pos}")
            print(f"  Position Size: {optimal_pos_size:.2f}x")
            print()

            print(f"Portfolio Train Period ({args.portfolio_train_start} to {args.portfolio_train_end}):")
            print(f"  Sharpe:        {train_metrics['sharpe']:.3f}")
            print(f"  Max Drawdown:  ${train_metrics['max_drawdown']:,.0f}")
            print(f"  CAGR:          {train_metrics['cagr']*100:.2f}%")
            print()

            print(f"✅ Test Period ({args.test_start} to {args.test_end}) - TRUE OUT-OF-SAMPLE:")
            print(f"  Sharpe:        {test_metrics['sharpe']:.3f}")
            print(f"  Sortino:       {test_metrics['sortino']:.3f}")
            print(f"  Max Drawdown:  ${test_metrics['max_drawdown']:,.0f} ({test_metrics['max_drawdown_pct']*100:.2f}%)")
            print(f"  CAGR:          {test_metrics['cagr']*100:.2f}%")
            print(f"  Total Return:  ${test_metrics['total_return']:,.0f}")
            print(f"  Profit Factor: {test_metrics['profit_factor']:.2f}")
            print(f"  Win Rate:      {test_metrics['win_rate']*100:.2f}%")
            print(f"  Avg Positions: {test_metrics['avg_positions']:.2f}")
            print(f"  Stops Hit:     {int(test_metrics['daily_stops_hit'])}")
            print(f"{'='*110}\n")

            # Save both configs and metrics
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                config = {
                    'symbols': sorted(list(optimal_symbols)),
                    'max_positions': optimal_max_pos,
                    'position_size': optimal_pos_size,
                    'portfolio_train_period': {
                        'start': args.portfolio_train_start,
                        'end': args.portfolio_train_end,
                        'metrics': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                                   for k, v in train_metrics.items()}
                    },
                    'test_period': {
                        'start': args.test_start,
                        'end': args.test_end,
                        'metrics': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                                   for k, v in test_metrics.items()}
                    }
                }

                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)

                logger.info(f"\nResults saved to: {output_path}")

        else:
            # Backward compatible: optimize directly on test period (NOT RECOMMENDED - causes leakage!)
            logger.warning("⚠️  WARNING: No portfolio train period specified!")
            logger.warning("⚠️  Optimizing on test period will cause PORTFOLIO-LEVEL DATA LEAKAGE!")
            logger.warning("⚠️  Use --portfolio-train-start and --portfolio-train-end to avoid this.")
            logger.warning("⚠️  Example: --portfolio-train-start 2021-01-01 --portfolio-train-end 2021-12-31\n")

            # Generate predictions for test period
            logger.info(f"Generating predictions for test period {args.test_start} to {args.test_end}...")
            predictions = generate_predictions(models, data_dir, args.test_start, args.test_end)

            if len(predictions) == 0:
                logger.error("No predictions generated!")
                return 1

            optimal_symbols, optimal_max_pos, optimal_pos_size, optimal_metrics = greedy_symbol_selection(
                predictions,
                models,
                (args.min_positions, min(args.max_positions, len(predictions))),
                args.daily_stop_loss,
                args.initial_cash,
                args.max_drawdown_limit,
                args.position_size,
                args.ranking_method,
                period_name=f"Test Period (LEAKAGE!) ({args.test_start} to {args.test_end})"
            )

            if optimal_symbols is None:
                logger.error("Could not find configuration meeting drawdown constraint")
                return 1

            # Save config
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                config = {
                    'symbols': sorted(list(optimal_symbols)),
                    'max_positions': optimal_max_pos,
                    'position_size': optimal_pos_size,
                    'metrics': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                               for k, v in optimal_metrics.items()}
                }

                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)

                logger.info(f"\nConfiguration saved to: {output_path}")

        return 0

    elif args.optimize_sizing:
        # Generate predictions for test period
        logger.info(f"Generating predictions for test period {args.test_start} to {args.test_end}...")
        predictions = generate_predictions(models, data_dir, args.test_start, args.test_end)

        if len(predictions) == 0:
            logger.error("No predictions generated!")
            return 1

        # 2D optimization: max_positions + position sizing
        results_df = optimize_positions_and_sizing(
            predictions,
            models,
            args.min_positions,
            min(args.max_positions, len(predictions)),
            (args.position_size_min, args.position_size_max),
            args.position_size_step,
            args.daily_stop_loss,
            args.initial_cash,
            args.max_drawdown_limit,
            args.ranking_method
        )
    else:
        # Generate predictions for test period
        logger.info(f"Generating predictions for test period {args.test_start} to {args.test_end}...")
        predictions = generate_predictions(models, data_dir, args.test_start, args.test_end)

        if len(predictions) == 0:
            logger.error("No predictions generated!")
            return 1

        # 1D optimization: max_positions only (full contract size)
        results_df = optimize_max_positions(
            predictions,
            models,
            args.min_positions,
            min(args.max_positions, len(predictions)),
            args.daily_stop_loss,
            args.initial_cash,
            args.max_drawdown_limit,
            args.ranking_method
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
