#!/usr/bin/env python3
"""
Filter trades from training data using ML models.

This is a FAST alternative to running full backtests. It:
1. Loads pre-calculated features from data/training/
2. Applies ML model scoring
3. Filters based on threshold
4. Saves to results/ directory

Time: ~2 minutes vs 90-180 minutes for full backtest
Memory: <500MB vs 114GB+
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from models.loader import load_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_trades_for_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    training_data_dir: Path,
    output_dir: Path
):
    """
    Filter trades for a single symbol using its trained ML model.

    Returns dict with stats.
    """
    logger.info(f"Processing {symbol}...")

    # Load training data CSV
    csv_path = training_data_dir / f"{symbol}_transformed_features.csv"
    if not csv_path.exists():
        logger.error(f"Training data not found: {csv_path}")
        return None

    logger.info(f"  Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} total trades")

    # Filter to date range
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df_filtered = df[(df['Date/Time'] >= start_dt) & (df['Date/Time'] <= end_dt)].copy()
    logger.info(f"  {len(df_filtered)} trades in {start_date} to {end_date}")

    if len(df_filtered) == 0:
        logger.warning(f"  No trades in date range for {symbol}")
        return None

    # Load ML model
    try:
        bundle = load_model_bundle(symbol)
        logger.info(f"  Loaded ML model (threshold={bundle.threshold})")
    except Exception as e:
        logger.error(f"  Failed to load ML model: {e}")
        return None

    # Get features
    if bundle.features is None:
        logger.error(f"  No features in model metadata")
        return None

    # Check if all features exist in CSV
    missing_features = [f for f in bundle.features if f not in df_filtered.columns]
    if missing_features:
        logger.error(f"  Missing features in CSV: {missing_features[:5]}...")
        return None

    # Extract feature matrix
    try:
        X = df_filtered[bundle.features].values
    except Exception as e:
        logger.error(f"  Failed to extract features: {e}")
        return None

    # Check for NaN values
    nan_count = pd.isna(X).sum()
    if nan_count > 0:
        logger.warning(f"  Found {nan_count} NaN values in features, filling with 0")
        X = pd.DataFrame(X).fillna(0).values

    # Score with ML model
    logger.info(f"  Scoring {len(X)} trades with ML model...")
    try:
        probas = bundle.model.predict_proba(X)[:, 1]  # Probability of class 1 (winner)
    except Exception as e:
        logger.error(f"  Model prediction failed: {e}")
        return None

    # Apply threshold filter
    if bundle.threshold is not None:
        passed_filter = probas >= bundle.threshold
        df_filtered['ml_score'] = probas
        df_filtered['Model_Selected'] = passed_filter.astype(int)

        df_ml_trades = df_filtered[passed_filter].copy()
        logger.info(f"  {len(df_ml_trades)} / {len(df_filtered)} trades passed ML filter (threshold={bundle.threshold:.3f})")
    else:
        logger.warning(f"  No threshold in model metadata, keeping all trades")
        df_filtered['ml_score'] = probas
        df_filtered['Model_Selected'] = 1
        df_ml_trades = df_filtered.copy()

    if len(df_ml_trades) == 0:
        logger.warning(f"  No trades passed ML filter")
        return {
            'symbol': symbol,
            'total_trades': len(df_filtered),
            'ml_trades': 0,
            'pnl_usd': 0,
            'win_rate': 0,
        }

    # Calculate stats
    total_pnl = df_ml_trades['y_pnl_usd'].sum()
    win_rate = (df_ml_trades['y_binary'] == 1).mean()

    logger.info(f"  ✅ {symbol}: {len(df_ml_trades)} trades | PnL: ${total_pnl:,.2f} | Win Rate: {win_rate:.1%}")

    # Save filtered trades
    output_symbol_dir = output_dir / f"{symbol}_optimization"
    output_symbol_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_symbol_dir / f"{symbol}_rf_best_trades.csv"

    # Select columns to save (match portfolio optimizer expectations)
    output_cols = [
        'Date/Time', 'Exit Date/Time', 'Entry_Price', 'Exit_Price',
        'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross',
        'Model_Selected'
    ]

    # Add pnl_usd column (duplicate of y_pnl_usd for compatibility)
    df_ml_trades['pnl_usd'] = df_ml_trades['y_pnl_usd']
    output_cols.append('pnl_usd')

    df_ml_trades[output_cols].to_csv(output_csv, index=False)
    logger.info(f"  Saved to {output_csv}")

    return {
        'symbol': symbol,
        'total_trades': len(df_filtered),
        'ml_trades': len(df_ml_trades),
        'pnl_usd': total_pnl,
        'win_rate': win_rate,
        'threshold': bundle.threshold,
    }


def main():
    parser = argparse.ArgumentParser(description='Filter trades using ML models')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--training-data-dir', type=str, default='data/training',
                       help='Training data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to process (default: all 18)')

    args = parser.parse_args()

    # Default symbols
    if args.symbols is None:
        symbols = [
            '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S',  # FX
            'ES', 'NQ', 'RTY', 'YM',  # Indices
            'GC', 'SI', 'HG', 'PL',  # Metals
            'CL', 'NG'  # Energy
        ]
    else:
        symbols = args.symbols

    training_data_dir = Path(args.training_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Filtering trades for {len(symbols)} symbols")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Training data: {training_data_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    results = []
    for symbol in symbols:
        result = filter_trades_for_symbol(
            symbol=symbol,
            start_date=args.start,
            end_date=args.end,
            training_data_dir=training_data_dir,
            output_dir=output_dir
        )
        if result:
            results.append(result)
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    total_pnl = sum(r['pnl_usd'] for r in results)
    total_trades = sum(r['ml_trades'] for r in results)

    for r in sorted(results, key=lambda x: x['pnl_usd'], reverse=True):
        logger.info(f"{r['symbol']:>4s} | Trades: {r['ml_trades']:>4d} | PnL: ${r['pnl_usd']:>12,.2f} | Win Rate: {r['win_rate']:>5.1%}")

    logger.info("-" * 80)
    logger.info(f"TOTAL | Trades: {total_trades:>4d} | PnL: ${total_pnl:>12,.2f}")
    logger.info("")
    logger.info(f"✅ Done! Trade data saved to {output_dir}/")


if __name__ == '__main__':
    main()
