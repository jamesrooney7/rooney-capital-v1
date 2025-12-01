#!/usr/bin/env python3
"""
Prepare ML-filtered trade data for portfolio optimization.

Merges ML predictions with training data to create trade files
in the format expected by portfolio_optimizer_greedy_train_test.py.

Usage:
    python research/prepare_portfolio_data.py --strategies CL_AvgHLRangeIBS ES_AvgHLRangeIBS
    python research/prepare_portfolio_data.py --all --exclude 6S_AvgHLRangeIBS 6S_SupportResistanceBounce
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_training_file(training_dir: Path, strategy_name: str) -> Path:
    """Find the training CSV file for a strategy."""
    # Strategy name format: {symbol}_{strategy}
    # Training file format: {symbol}_{strategy}_*_training.csv
    pattern = f"{strategy_name}_*_training.csv"
    matches = list(training_dir.glob(pattern))

    if not matches:
        # Try without the trailing params
        pattern = f"{strategy_name}*.csv"
        matches = list(training_dir.glob(pattern))

    if matches:
        return matches[0]
    return None


def prepare_strategy_data(
    strategy_name: str,
    ml_results_dir: Path,
    training_dir: Path,
    output_dir: Path
) -> bool:
    """
    Prepare trade data for a single strategy.

    Returns True if successful.
    """
    # Find prediction file
    pred_dir = ml_results_dir / strategy_name
    pred_file = pred_dir / f"{strategy_name}_ml_meta_labeling_held_out_predictions.csv"

    if not pred_file.exists():
        logger.warning(f"  No predictions found: {pred_file}")
        return False

    # Find training file
    training_file = find_training_file(training_dir, strategy_name)
    if not training_file:
        logger.warning(f"  No training data found for {strategy_name}")
        return False

    logger.info(f"  Loading predictions: {pred_file.name}")
    logger.info(f"  Loading training: {training_file.name}")

    # Load data
    pred_df = pd.read_csv(pred_file)
    train_df = pd.read_csv(training_file)

    # Normalize date columns for merging
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    train_df['Date/Time'] = pd.to_datetime(train_df['Date/Time'])

    # Filter training to held-out period (2021-2024)
    train_df = train_df[train_df['Date/Time'] >= '2021-01-01'].copy()

    # Merge predictions with training data
    merged = train_df.merge(
        pred_df[['Date', 'y_pred_proba', 'y_pred_binary']],
        left_on='Date/Time',
        right_on='Date',
        how='inner'
    )

    if len(merged) == 0:
        logger.warning(f"  No matching trades after merge!")
        return False

    logger.info(f"  Merged {len(merged)} trades")

    # Create output format expected by portfolio simulator
    # Rename columns to match expected format
    output_df = pd.DataFrame({
        'Date/Time': merged['Date/Time'],
        'Exit Date/Time': merged['Exit Date/Time'],
        'Entry_Price': merged['Entry_Price'],
        'Exit_Price': merged['Exit_Price'],
        'Model_PnL_USD': merged['y_pnl_usd'],
        'Model_Selected': merged['y_pred_binary'].astype(int),
        'Model_Probability': merged['y_pred_proba'],
        'y_return': merged['y_return'],
        'Symbol': merged['Symbol'],
        'Strategy': merged['Strategy'],
    })

    # Extract symbol from strategy name (e.g., "CL_AvgHLRangeIBS" -> "CL")
    symbol = strategy_name.split('_')[0]

    # Create output directory
    symbol_output_dir = output_dir / f"{symbol}_optimization"
    symbol_output_dir.mkdir(parents=True, exist_ok=True)

    # Save trades
    output_file = symbol_output_dir / f"{symbol}_rf_best_trades.csv"
    output_df.to_csv(output_file, index=False)

    # Save metadata
    n_selected = output_df['Model_Selected'].sum()
    n_total = len(output_df)

    metadata = {
        'strategy_name': strategy_name,
        'symbol': symbol,
        'total_trades': int(n_total),
        'selected_trades': int(n_selected),
        'filter_rate': float(1 - n_selected / n_total) if n_total > 0 else 0,
    }

    meta_file = symbol_output_dir / "best.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Saved {n_selected}/{n_total} trades to {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare ML-filtered trades for portfolio optimization')
    parser.add_argument('--strategies', nargs='+', help='Strategy names to include')
    parser.add_argument('--all', action='store_true', help='Include all strategies from ML results')
    parser.add_argument('--exclude', nargs='+', default=[], help='Strategies to exclude')
    parser.add_argument('--ml-results-dir', default='research/ml_meta_labeling/results',
                       help='ML results directory')
    parser.add_argument('--training-dir', default='data/training/factory',
                       help='Training data directory')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for portfolio optimizer')

    args = parser.parse_args()

    ml_results_dir = Path(args.ml_results_dir)
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)

    # Get list of strategies
    if args.all:
        strategies = [
            d.name for d in ml_results_dir.iterdir()
            if d.is_dir() and not d.name.endswith('.db')
        ]
    elif args.strategies:
        strategies = args.strategies
    else:
        parser.error("Must specify --strategies or --all")

    # Apply exclusions
    strategies = [s for s in strategies if s not in args.exclude]

    logger.info(f"Preparing {len(strategies)} strategies for portfolio optimization")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    successful = 0
    failed = 0

    for strategy in sorted(strategies):
        logger.info(f"Processing {strategy}...")

        if prepare_strategy_data(strategy, ml_results_dir, training_dir, output_dir):
            successful += 1
        else:
            failed += 1

        logger.info("")

    logger.info("=" * 60)
    logger.info(f"Complete: {successful} successful, {failed} failed")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next step - run portfolio optimizer:")
    logger.info(f"  python research/portfolio_optimizer_greedy_train_test.py \\")
    logger.info(f"      --results-dir {output_dir} \\")
    logger.info(f"      --train-start 2021-01-01 --train-end 2022-12-31 \\")
    logger.info(f"      --test-start 2023-01-01 --test-end 2024-12-31 \\")
    logger.info(f"      --max-positions 5 --max-dd-limit 5000")


if __name__ == '__main__':
    main()
