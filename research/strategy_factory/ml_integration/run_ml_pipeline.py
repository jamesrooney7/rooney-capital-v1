#!/usr/bin/env python3
"""
ML Pipeline Runner for Strategy Factory Vectorized Strategies.

Orchestrates the full ML training workflow:
1. Load winners from database or JSON file
2. Extract training data using vectorized strategies
3. Train ML models using train_rf_cpcv_bo.py
4. Save results and models

This replaces the old run_ml_pipeline.py which expected Backtrader strategies.

Usage:
    # Run ML pipeline for all winners from extract_winners.py output
    python -m research.strategy_factory.ml_integration.run_ml_pipeline \\
        --winners winners.json \\
        --output-dir models/factory

    # Run for specific symbol only
    python -m research.strategy_factory.ml_integration.run_ml_pipeline \\
        --winners winners.json \\
        --symbol ES \\
        --output-dir models/factory
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research.strategy_factory.ml_integration.extract_training_data import (
    extract_training_data,
    extract_from_winners,
    register_strategies
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ml_training(
    training_data_path: str,
    symbol: str,
    strategy_name: str,
    output_dir: str,
    n_trials: int = 50,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Run ML training for a single strategy using train_rf_cpcv_bo.py.

    Args:
        training_data_path: Path to training data CSV
        symbol: Symbol (for output naming)
        strategy_name: Strategy name (for output naming)
        output_dir: Output directory for models
        n_trials: Number of Bayesian optimization trials
        n_folds: Number of CPCV folds

    Returns:
        Dict with training results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink or copy to expected location
    # train_rf_cpcv_bo.py expects: data/training/{SYMBOL}_transformed_features.csv
    expected_path = project_root / f'data/training/{symbol}_{strategy_name}_transformed_features.csv'
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy training data to expected location
    import shutil
    shutil.copy(training_data_path, expected_path)
    logger.info(f"Copied training data to {expected_path}")

    # Run train_rf_cpcv_bo.py
    cmd = [
        sys.executable,
        str(project_root / 'research/train_rf_cpcv_bo.py'),
        '--symbol', f'{symbol}_{strategy_name}',  # Use combined name
        '--n-trials', str(n_trials),
        '--n-folds', str(n_folds),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr}")
            return {'success': False, 'error': result.stderr}

        logger.info(f"Training output:\n{result.stdout[-2000:]}")  # Last 2000 chars

        # Look for output model
        model_path = project_root / f'models/{symbol}_{strategy_name}_rf_model.pkl'
        if model_path.exists():
            # Move to specified output dir
            final_path = output_dir / model_path.name
            shutil.move(str(model_path), str(final_path))
            logger.info(f"Saved model to {final_path}")
            return {'success': True, 'model_path': str(final_path)}

        return {'success': True, 'output': result.stdout}

    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 1 hour")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {'success': False, 'error': str(e)}


def run_pipeline_for_winners(
    winners_path: str,
    output_dir: str = "models/factory",
    symbol_filter: Optional[str] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    n_trials: int = 50,
    n_folds: int = 5,
    skip_extraction: bool = False
) -> Dict[str, Any]:
    """
    Run full ML pipeline for all winners.

    Args:
        winners_path: Path to winners.json
        output_dir: Output directory for models
        symbol_filter: Only process this symbol (optional)
        start_date: Start date for training data
        end_date: End date for training data
        n_trials: Number of BO trials
        n_folds: Number of CPCV folds
        skip_extraction: Skip extraction if training data already exists

    Returns:
        Dict with results for each strategy
    """
    with open(winners_path, 'r') as f:
        winners = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_dir = Path('data/training/factory')
    training_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_processed = 0
    total_success = 0

    # Filter symbols if specified
    if symbol_filter:
        winners = {k: v for k, v in winners.items() if k == symbol_filter}

    total_strategies = sum(len(w) for w in winners.values())
    logger.info(f"Processing {total_strategies} strategies from {len(winners)} symbols")

    for symbol, symbol_winners in winners.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"SYMBOL: {symbol} - {len(symbol_winners)} strategies")
        logger.info(f"{'='*70}")

        for winner in symbol_winners:
            strategy_name = winner['strategy_name']
            params = winner['params']
            total_processed += 1

            logger.info(f"\n[{total_processed}/{total_strategies}] {strategy_name}")
            logger.info(f"Params: {params}")

            try:
                # Step 1: Extract training data
                param_str = '_'.join(f"{k}{v}" for k, v in sorted(params.items())
                                     if k not in ['stop_loss_atr', 'take_profit_atr', 'max_bars_held'])
                training_file = training_dir / f"{symbol}_{strategy_name}_{param_str}_training.csv"

                if skip_extraction and training_file.exists():
                    logger.info(f"Skipping extraction, using existing: {training_file}")
                else:
                    df = extract_training_data(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        params=params,
                        start_date=start_date,
                        end_date=end_date,
                        output_path=str(training_file)
                    )

                    if df.empty:
                        logger.warning(f"No trades extracted, skipping ML training")
                        results[f"{symbol}_{strategy_name}"] = {'success': False, 'error': 'No trades'}
                        continue

                # Step 2: Train ML model
                ml_result = run_ml_training(
                    training_data_path=str(training_file),
                    symbol=symbol,
                    strategy_name=strategy_name,
                    output_dir=str(output_dir),
                    n_trials=n_trials,
                    n_folds=n_folds
                )

                results[f"{symbol}_{strategy_name}"] = ml_result

                if ml_result.get('success'):
                    total_success += 1
                    logger.info(f"SUCCESS: {strategy_name}")
                else:
                    logger.warning(f"FAILED: {strategy_name} - {ml_result.get('error', 'Unknown')}")

            except Exception as e:
                logger.error(f"Error processing {strategy_name}: {e}", exc_info=True)
                results[f"{symbol}_{strategy_name}"] = {'success': False, 'error': str(e)}

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_processed - total_success}")

    # Save results summary
    summary_path = output_dir / 'pipeline_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results summary to {summary_path}")

    return results


def run_pipeline_single(
    strategy_name: str,
    symbol: str,
    params: Dict[str, Any],
    output_dir: str = "models/factory",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    n_trials: int = 50,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Run ML pipeline for a single strategy.

    Args:
        strategy_name: Strategy class name
        symbol: Symbol
        params: Strategy parameters
        output_dir: Output directory
        start_date: Start date
        end_date: End date
        n_trials: Number of BO trials
        n_folds: Number of CPCV folds

    Returns:
        Training result dict
    """
    logger.info(f"Running ML pipeline for {strategy_name} on {symbol}")

    # Extract training data
    training_dir = Path('data/training/factory')
    training_dir.mkdir(parents=True, exist_ok=True)
    training_file = training_dir / f"{symbol}_{strategy_name}_training.csv"

    df = extract_training_data(
        strategy_name=strategy_name,
        symbol=symbol,
        params=params,
        start_date=start_date,
        end_date=end_date,
        output_path=str(training_file)
    )

    if df.empty:
        return {'success': False, 'error': 'No trades extracted'}

    # Train ML model
    result = run_ml_training(
        training_data_path=str(training_file),
        symbol=symbol,
        strategy_name=strategy_name,
        output_dir=output_dir,
        n_trials=n_trials,
        n_folds=n_folds
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run ML pipeline for Strategy Factory vectorized strategies'
    )

    # Input modes
    parser.add_argument('--winners', type=str, help='Path to winners.json file')
    parser.add_argument('--strategy', type=str, help='Single strategy name')
    parser.add_argument('--symbol', type=str, help='Symbol (required for --strategy, optional filter for --winners)')
    parser.add_argument('--params', type=str, help='Strategy params as JSON (for --strategy)')

    # Date range
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')

    # ML training options
    parser.add_argument('--n-trials', type=int, default=50, help='Number of BO trials')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CPCV folds')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/factory', help='Output directory')

    # Options
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if data exists')

    args = parser.parse_args()

    # Register strategies
    register_strategies()

    try:
        # Mode 1: Process winners file
        if args.winners:
            results = run_pipeline_for_winners(
                winners_path=args.winners,
                output_dir=args.output_dir,
                symbol_filter=args.symbol,
                start_date=args.start,
                end_date=args.end,
                n_trials=args.n_trials,
                n_folds=args.n_folds,
                skip_extraction=args.skip_extraction
            )

            success_count = sum(1 for r in results.values() if r.get('success'))
            logger.info(f"Pipeline complete: {success_count}/{len(results)} successful")
            return 0 if success_count > 0 else 1

        # Mode 2: Single strategy
        if args.strategy and args.symbol:
            import ast
            params = json.loads(args.params) if args.params else {}
            if isinstance(params, str):
                params = ast.literal_eval(params)

            result = run_pipeline_single(
                strategy_name=args.strategy,
                symbol=args.symbol,
                params=params,
                output_dir=args.output_dir,
                start_date=args.start,
                end_date=args.end,
                n_trials=args.n_trials,
                n_folds=args.n_folds
            )

            return 0 if result.get('success') else 1

        parser.print_help()
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
