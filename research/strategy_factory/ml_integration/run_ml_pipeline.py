#!/usr/bin/env python3
"""
ML Pipeline Runner for Strategy Factory Vectorized Strategies.

Orchestrates the full ML training workflow:
1. Load winners from database or JSON file
2. Extract training data using vectorized strategies
3. Train ML models using ml_meta_labeling_optimizer.py (LightGBM + Walk-Forward)
4. Save results and models

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

    # Use legacy RF training instead of LightGBM
    python -m research.strategy_factory.ml_integration.run_ml_pipeline \\
        --winners winners.json \\
        --use-legacy-rf
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

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


def run_ml_meta_labeling(
    training_data_path: str,
    symbol: str,
    strategy_name: str,
    output_dir: str,
    n_trials: int = 100,
    n_clusters: int = 30,
    cv_folds: int = 5,
    embargo_days: int = 60,
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """
    Run ML training using ml_meta_labeling_optimizer.py (LightGBM + Walk-Forward).

    This is the recommended training approach using:
    - Hierarchical clustering for feature selection
    - LightGBM with Optuna optimization
    - Walk-forward validation (2016-2020)
    - Held-out test period (2021-2024)
    - Optional ensemble (LightGBM + CatBoost + XGBoost)

    Args:
        training_data_path: Path to training data CSV
        symbol: Symbol (for output naming)
        strategy_name: Strategy name (for output naming)
        output_dir: Output directory for models
        n_trials: Number of Optuna trials per walk-forward window
        n_clusters: Number of feature clusters for selection
        cv_folds: Number of cross-validation folds
        embargo_days: Embargo period in days
        use_ensemble: Whether to use ensemble model

    Returns:
        Dict with training results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ml_meta_labeling expects: data/training/{SYMBOL}_transformed_features.csv
    combined_name = f"{symbol}_{strategy_name}"
    expected_path = project_root / f'data/training/{combined_name}_transformed_features.csv'
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy training data to expected location
    shutil.copy(training_data_path, expected_path)
    logger.info(f"Copied training data to {expected_path}")

    # Run ml_meta_labeling_optimizer.py
    cmd = [
        sys.executable,
        str(project_root / 'research/ml_meta_labeling/ml_meta_labeling_optimizer.py'),
        '--symbol', combined_name,
        '--n-trials', str(n_trials),
        '--n-clusters', str(n_clusters),
        '--cv-folds', str(cv_folds),
        '--embargo-days', str(embargo_days),
    ]

    if use_ensemble:
        cmd.append('--use-ensemble')

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14400,  # 4 hour timeout (walk-forward takes longer)
            cwd=str(project_root)
        )

        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr[-2000:]}")
            return {'success': False, 'error': result.stderr[-2000:]}

        logger.info(f"Training output:\n{result.stdout[-3000:]}")

        # Look for output model in ml_meta_labeling results directory
        results_dir = project_root / f'research/ml_meta_labeling/results/{combined_name}'
        model_path = results_dir / f'{combined_name}_ml_meta_labeling_final_model.pkl'

        if model_path.exists():
            # Copy model to specified output dir
            final_path = output_dir / model_path.name
            shutil.copy(str(model_path), str(final_path))
            logger.info(f"Saved model to {final_path}")

            # Also copy summary if exists
            summary_path = results_dir / f'{combined_name}_ml_meta_labeling_executive_summary.txt'
            if summary_path.exists():
                shutil.copy(str(summary_path), str(output_dir / summary_path.name))

            return {
                'success': True,
                'model_path': str(final_path),
                'results_dir': str(results_dir)
            }

        return {'success': True, 'output': result.stdout[-2000:]}

    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 4 hours")
        return {'success': False, 'error': 'Timeout after 4 hours'}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {'success': False, 'error': str(e)}


def run_rf_cpcv_training(
    training_data_path: str,
    symbol: str,
    strategy_name: str,
    output_dir: str,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Run ML training using rf_cpcv_random_then_bo.py (Random Forest + CPCV).

    Legacy approach - use run_ml_meta_labeling() for better results.

    Args:
        training_data_path: Path to training data CSV
        symbol: Symbol (for output naming)
        strategy_name: Strategy name (for output naming)
        output_dir: Output directory for models
        n_trials: Number of optimization trials

    Returns:
        Dict with training results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_name = f"{symbol}_{strategy_name}"
    expected_path = project_root / f'data/training/{combined_name}_transformed_features.csv'
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(training_data_path, expected_path)
    logger.info(f"Copied training data to {expected_path}")

    # Run rf_cpcv_random_then_bo.py
    cmd = [
        sys.executable,
        str(project_root / 'research/rf_cpcv_random_then_bo.py'),
        '--symbol', combined_name,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            cwd=str(project_root / 'research')
        )

        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr[-2000:]}")
            return {'success': False, 'error': result.stderr[-2000:]}

        logger.info(f"Training output:\n{result.stdout[-2000:]}")
        return {'success': True, 'output': result.stdout[-2000:]}

    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 2 hours")
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
    n_trials: int = 100,
    n_clusters: int = 30,
    use_ensemble: bool = True,
    use_legacy_rf: bool = False,
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
        n_trials: Number of Optuna trials per window
        n_clusters: Number of feature clusters
        use_ensemble: Use ensemble model (LightGBM + CatBoost + XGBoost)
        use_legacy_rf: Use legacy Random Forest instead of LightGBM
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
    logger.info(f"Training method: {'Legacy RF (CPCV)' if use_legacy_rf else 'LightGBM (Walk-Forward)'}")

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
                if use_legacy_rf:
                    ml_result = run_rf_cpcv_training(
                        training_data_path=str(training_file),
                        symbol=symbol,
                        strategy_name=strategy_name,
                        output_dir=str(output_dir),
                        n_trials=n_trials
                    )
                else:
                    ml_result = run_ml_meta_labeling(
                        training_data_path=str(training_file),
                        symbol=symbol,
                        strategy_name=strategy_name,
                        output_dir=str(output_dir),
                        n_trials=n_trials,
                        n_clusters=n_clusters,
                        use_ensemble=use_ensemble
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
    n_trials: int = 100,
    n_clusters: int = 30,
    use_ensemble: bool = True,
    use_legacy_rf: bool = False
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
        n_trials: Number of Optuna trials
        n_clusters: Number of feature clusters
        use_ensemble: Use ensemble model
        use_legacy_rf: Use legacy Random Forest

    Returns:
        Training result dict
    """
    logger.info(f"Running ML pipeline for {strategy_name} on {symbol}")
    logger.info(f"Training method: {'Legacy RF (CPCV)' if use_legacy_rf else 'LightGBM (Walk-Forward)'}")

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
    if use_legacy_rf:
        result = run_rf_cpcv_training(
            training_data_path=str(training_file),
            symbol=symbol,
            strategy_name=strategy_name,
            output_dir=output_dir,
            n_trials=n_trials
        )
    else:
        result = run_ml_meta_labeling(
            training_data_path=str(training_file),
            symbol=symbol,
            strategy_name=strategy_name,
            output_dir=output_dir,
            n_trials=n_trials,
            n_clusters=n_clusters,
            use_ensemble=use_ensemble
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
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials per window')
    parser.add_argument('--n-clusters', type=int, default=30, help='Number of feature clusters')
    parser.add_argument('--use-ensemble', action='store_true', default=True,
                       help='Use ensemble model (default: True)')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble model')
    parser.add_argument('--use-legacy-rf', action='store_true',
                       help='Use legacy Random Forest (CPCV) instead of LightGBM (Walk-Forward)')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/factory', help='Output directory')

    # Options
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if data exists')

    args = parser.parse_args()

    # Handle ensemble flag
    use_ensemble = args.use_ensemble and not args.no_ensemble

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
                n_clusters=args.n_clusters,
                use_ensemble=use_ensemble,
                use_legacy_rf=args.use_legacy_rf,
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
                n_clusters=args.n_clusters,
                use_ensemble=use_ensemble,
                use_legacy_rf=args.use_legacy_rf
            )

            return 0 if result.get('success') else 1

        parser.print_help()
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
