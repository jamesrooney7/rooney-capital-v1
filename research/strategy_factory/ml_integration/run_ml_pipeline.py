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
from research.strategy_factory.ml_integration.results_to_db import populate_database

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
    n_clusters: int = 10,
    cv_folds: int = 5,
    embargo_days: int = 60,
    use_ensemble: bool = True,
    min_samples_per_class: int = 200
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
        n_clusters: Number of feature clusters for selection (default 10)
        cv_folds: Number of cross-validation folds
        embargo_days: Embargo period in days
        use_ensemble: Whether to use ensemble model
        min_samples_per_class: Minimum samples per class (default 200)

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
        '--min-samples-per-class', str(min_samples_per_class),
    ]

    # Ensemble is enabled by default in ml_meta_labeling_optimizer.py
    # Only add flag when we want to disable it
    if not use_ensemble:
        cmd.append('--no-use-ensemble')

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


def filter_best_version_per_strategy(winners: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Filter winners to keep only the best performing version of each strategy per symbol.

    For each (symbol, strategy_name) combination, keeps only the winner with the highest
    Sharpe ratio (or other ranking metric).

    Args:
        winners: Dict mapping symbol -> list of winner dicts

    Returns:
        Filtered winners dict with only best version per strategy
    """
    filtered_winners = {}

    for symbol, symbol_winners in winners.items():
        # Group by strategy name
        strategy_groups = {}
        for winner in symbol_winners:
            strategy_name = winner['strategy_name']
            if strategy_name not in strategy_groups:
                strategy_groups[strategy_name] = []
            strategy_groups[strategy_name].append(winner)

        # Keep only the best version of each strategy
        best_winners = []
        for strategy_name, versions in strategy_groups.items():
            if len(versions) == 1:
                best_winners.append(versions[0])
            else:
                # Sort by Sharpe ratio (descending), then by profit factor
                def get_score(w):
                    sharpe = w.get('sharpe_ratio', w.get('sharpe', 0)) or 0
                    pf = w.get('profit_factor', 0) or 0
                    return (sharpe, pf)

                sorted_versions = sorted(versions, key=get_score, reverse=True)
                best = sorted_versions[0]
                best_winners.append(best)
                logger.info(f"  {symbol} {strategy_name}: keeping best version "
                           f"(Sharpe={get_score(best)[0]:.2f}) out of {len(versions)} versions")

        filtered_winners[symbol] = best_winners

    return filtered_winners


def run_pipeline_for_winners(
    winners_path: str,
    output_dir: str = "models/factory",
    symbol_filter: Optional[str] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    n_trials: int = 100,
    n_clusters: int = 10,
    use_ensemble: bool = True,
    use_legacy_rf: bool = False,
    skip_extraction: bool = False,
    min_samples_per_class: int = 200,
    best_version_only: bool = False,
    skip_db: bool = False
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
        n_clusters: Number of feature clusters (default 10)
        use_ensemble: Use ensemble model (LightGBM + CatBoost + XGBoost)
        use_legacy_rf: Use legacy Random Forest instead of LightGBM
        skip_extraction: Skip extraction if training data already exists
        min_samples_per_class: Minimum samples per class (default 200)
        best_version_only: Only run the best performing version of each strategy
        skip_db: Skip database population at the end

    Returns:
        Dict with results for each strategy
    """
    with open(winners_path, 'r') as f:
        raw_data = json.load(f)

    # Handle manifest format from extract_winners.py
    # Manifest has: {"version": ..., "winners": [...], "winners_by_instrument": {...}}
    # We need to group the full winners list by symbol
    if 'winners' in raw_data and isinstance(raw_data.get('winners'), list):
        # This is a manifest - group winners by symbol
        winners_list = raw_data['winners']
        winners = {}
        for w in winners_list:
            symbol = w['symbol']
            if symbol not in winners:
                winners[symbol] = []
            winners[symbol].append(w)
        logger.info(f"Loaded manifest with {len(winners_list)} total winners")
    elif isinstance(raw_data, dict) and all(isinstance(v, list) for v in raw_data.values()):
        # Already in expected format: {symbol: [winners]}
        winners = raw_data
    else:
        raise ValueError(f"Unrecognized winners.json format. Expected manifest or {{symbol: [winners]}}")

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

    # Filter to best version per strategy if specified
    if best_version_only:
        original_count = sum(len(w) for w in winners.values())
        winners = filter_best_version_per_strategy(winners)
        filtered_count = sum(len(w) for w in winners.values())
        logger.info(f"Filtered to best version per strategy: {original_count} -> {filtered_count} strategies")

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
                        use_ensemble=use_ensemble,
                        min_samples_per_class=min_samples_per_class
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

    # Populate database with results (unless skipped)
    if not skip_db:
        ml_results_dir = project_root / 'research/ml_meta_labeling/results'
        db_path = ml_results_dir / 'ml_results.db'
        logger.info(f"\nPopulating results database...")
        n_parsed = populate_database(str(ml_results_dir), str(db_path), quiet=False)
        logger.info(f"Database updated with {n_parsed} results: {db_path}")

    return results


def run_pipeline_single(
    strategy_name: str,
    symbol: str,
    params: Dict[str, Any],
    output_dir: str = "models/factory",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    n_trials: int = 100,
    n_clusters: int = 10,
    use_ensemble: bool = True,
    use_legacy_rf: bool = False,
    min_samples_per_class: int = 200
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
        n_clusters: Number of feature clusters (default 10)
        use_ensemble: Use ensemble model
        use_legacy_rf: Use legacy Random Forest
        min_samples_per_class: Minimum samples per class (default 200)

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
            use_ensemble=use_ensemble,
            min_samples_per_class=min_samples_per_class
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
    parser.add_argument('--n-clusters', type=int, default=10, help='Number of feature clusters (default: 10)')
    parser.add_argument('--min-samples-per-class', type=int, default=200,
                       help='Minimum samples per class for ML training (default: 200)')
    parser.add_argument('--use-ensemble', action='store_true', default=True,
                       help='Use ensemble model (default: True)')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble model')
    parser.add_argument('--use-legacy-rf', action='store_true',
                       help='Use legacy Random Forest (CPCV) instead of LightGBM (Walk-Forward)')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/factory', help='Output directory')

    # Options
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if data exists')
    parser.add_argument('--best-version-only', action='store_true',
                       help='Only run the best performing version of each strategy per symbol')
    parser.add_argument('--no-db', action='store_true',
                       help='Skip database population at the end')

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
                skip_extraction=args.skip_extraction,
                min_samples_per_class=args.min_samples_per_class,
                best_version_only=args.best_version_only,
                skip_db=args.no_db
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
                use_legacy_rf=args.use_legacy_rf,
                min_samples_per_class=args.min_samples_per_class
            )

            return 0 if result.get('success') else 1

        parser.print_help()
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
