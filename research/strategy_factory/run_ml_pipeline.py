#!/usr/bin/env python3
"""
Automated ML Pipeline for Strategy Factory Winners

This script automates the complete ML enhancement pipeline:
1. Read winners manifest (from extract_winners.py)
2. For each winner:
   a. Extract training data (2010-2021)
   b. Train ML model (420 trials)
   c. Validate on 2022-2024
   d. Generate reports
3. Aggregate results for portfolio optimization

Usage:
    # Run full pipeline for all winners
    python research/strategy_factory/run_ml_pipeline.py \\
        --manifest ml_pipeline/winners_manifest.json \\
        --workers 4

    # Run specific strategies only
    python research/strategy_factory/run_ml_pipeline.py \\
        --manifest ml_pipeline/winners_manifest.json \\
        --strategies 21 40 45 \\
        --symbols ES NQ

IMPORTANT: Strategies must be ported to Backtrader first!
See STRATEGY_TO_BACKTRADER_GUIDE.md for porting instructions.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipelineRunner:
    """Automates ML pipeline for strategy factory winners."""

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        workers: int = 4,
        skip_existing: bool = True
    ):
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.workers = workers
        self.skip_existing = skip_existing

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.winners = self.manifest['winners']

        # Create output directories
        self.training_data_dir = output_dir / 'training_data'
        self.models_dir = output_dir / 'models'
        self.validation_dir = output_dir / 'validation'
        self.reports_dir = output_dir / 'reports'

        for dir_path in [self.training_data_dir, self.models_dir,
                         self.validation_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_strategy_module_name(self, strategy_name: str) -> str:
        """
        Map strategy factory name to Backtrader module.

        Example:
            'RSI2_MeanReversion' -> 'rsi2_mean_reversion_bt'
            'BuyOn5BarLow' -> 'buy_on_5_bar_low_bt'
        """
        # Convert CamelCase to snake_case and add _bt suffix
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strategy_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return f"{name}_bt"

    def extract_training_data(self, winner: Dict[str, Any]) -> Path:
        """
        Extract training data for a winner.

        Args:
            winner: Winner dictionary from manifest

        Returns:
            Path to training data CSV
        """
        symbol = winner['symbol']
        strategy_id = winner['strategy_id']
        strategy_name = winner['strategy_name']

        output_csv = self.training_data_dir / f"{symbol}_{strategy_id}_training.csv"

        if self.skip_existing and output_csv.exists():
            logger.info(f"  Training data already exists: {output_csv}")
            return output_csv

        logger.info(f"  Extracting training data...")
        logger.info(f"    Symbol: {symbol}")
        logger.info(f"    Strategy: {strategy_name} (#{strategy_id})")

        # Build command
        cmd = [
            'python3', 'research/extract_training_data.py',
            '--symbol', symbol,
            '--start', '2010-01-01',
            '--end', '2021-12-31',
            '--strategy-class', self.get_strategy_module_name(strategy_name),
            '--strategy-params', json.dumps(winner['params']),
            '--output', str(output_csv)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"  ✅ Training data extracted: {output_csv}")
            return output_csv

        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to extract training data: {e}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            return None

    def train_ml_model(self, winner: Dict[str, Any], training_csv: Path) -> Path:
        """
        Train ML meta-labeling model for a winner.

        Uses the advanced ML meta-labeling system with:
        - Hierarchical feature clustering
        - Walk-forward validation (2011-2020)
        - LightGBM ensemble with Optuna optimization
        - Held-out test (2021-2024)

        Args:
            winner: Winner dictionary from manifest
            training_csv: Path to training data CSV

        Returns:
            Path to trained model file
        """
        symbol = winner['symbol']
        strategy_id = winner['strategy_id']

        # ML meta-labeling outputs to specific directory structure
        output_dir = Path('research/ml_meta_labeling/results') / f"{symbol}_{strategy_id}"
        model_path = output_dir / f"{symbol}_{strategy_id}_ml_meta_labeling_final_model.pkl"

        if self.skip_existing and model_path.exists():
            logger.info(f"  Model already exists: {model_path}")
            return model_path

        logger.info(f"  Training ML meta-labeling model...")
        logger.info(f"    Input: {training_csv}")
        logger.info(f"    Output directory: {output_dir}")
        logger.info(f"    System: LightGBM ensemble + walk-forward validation")

        # Copy training CSV to expected location
        # ML meta-labeling expects data in data/training/{symbol}_transformed_features.csv
        training_data_dir = Path('data/training')
        training_data_dir.mkdir(parents=True, exist_ok=True)
        expected_csv = training_data_dir / f"{symbol}_{strategy_id}_transformed_features.csv"

        import shutil
        shutil.copy(training_csv, expected_csv)

        # Build command for ML meta-labeling optimizer
        cmd = [
            'python3', 'research/ml_meta_labeling/ml_meta_labeling_optimizer.py',
            '--symbol', f"{symbol}_{strategy_id}",  # Use symbol_strategyid as identifier
            '--data-dir', str(training_data_dir.parent),  # Parent of training/
            '--output-dir', 'research/ml_meta_labeling/results',
            '--n-clusters', '30',          # Feature clustering
            '--n-trials', '100',           # Optuna trials per window
            '--cv-folds', '5',             # Cross-validation folds
            '--embargo-days', '5',         # Prevent label leakage
            '--task-type', 'classification',  # Predict win/loss
            '--optimization-metric', 'precision',
            '--precision-threshold', '0.60',  # Only take high-confidence trades
            '--use-ensemble'               # Use LightGBM + CatBoost + XGBoost
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10800  # 3 hour timeout (more complex than old system)
            )
            logger.info(f"  ✅ ML meta-labeling model trained: {model_path}")

            # Log key results
            summary_file = output_dir / f"{symbol}_{strategy_id}_ml_meta_labeling_executive_summary.txt"
            if summary_file.exists():
                logger.info(f"  📊 Summary: {summary_file}")

            return model_path

        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to train ML meta-labeling model: {e}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ ML meta-labeling training timed out (3 hours)")
            return None
        finally:
            # Clean up temporary CSV
            if expected_csv.exists():
                expected_csv.unlink()

    def validate_strategy(
        self,
        winner: Dict[str, Any],
        model_path: Path
    ) -> Dict[str, Any]:
        """
        Validate strategy with ML filter on 2022-2024.

        Args:
            winner: Winner dictionary from manifest
            model_path: Path to trained model

        Returns:
            Validation results dictionary
        """
        symbol = winner['symbol']
        strategy_id = winner['strategy_id']
        strategy_name = winner['strategy_name']

        validation_csv = self.validation_dir / f"{symbol}_{strategy_id}_validation.csv"

        logger.info(f"  Running validation (2022-2024)...")

        # Build command
        cmd = [
            'python3', 'research/backtest_runner.py',
            '--symbol', symbol,
            '--start', '2022-01-01',
            '--end', '2024-12-31',
            '--strategy', self.get_strategy_module_name(strategy_name),
            '--strategy-params', json.dumps(winner['params']),
            '--ml-model', str(model_path),
            '--output', str(validation_csv)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800  # 30 minute timeout
            )

            # Parse validation results from CSV
            if validation_csv.exists():
                # TODO: Extract key metrics from validation CSV
                # For now, return placeholder
                return {
                    'symbol': symbol,
                    'strategy_id': strategy_id,
                    'strategy_name': strategy_name,
                    'validation_csv': str(validation_csv),
                    'status': 'completed'
                }

            logger.info(f"  ✅ Validation complete: {validation_csv}")
            return {'status': 'completed'}

        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ Validation timed out (30 minutes)")
            return {'status': 'failed', 'error': 'timeout'}

    def process_winner(self, winner: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single winner through the complete ML pipeline.

        Args:
            winner: Winner dictionary from manifest

        Returns:
            Results dictionary
        """
        symbol = winner['symbol']
        strategy_id = winner['strategy_id']
        strategy_name = winner['strategy_name']

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PROCESSING: {symbol} - {strategy_name} (#{strategy_id})")
        logger.info("=" * 80)
        logger.info(f"Phase 1 Sharpe: {winner['sharpe_ratio']:.3f}")
        logger.info(f"Trades: {winner['total_trades']}")
        logger.info(f"Win Rate: {winner['win_rate']*100:.1f}%")
        logger.info("")

        result = {
            'symbol': symbol,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'phase1_sharpe': winner['sharpe_ratio'],
            'status': 'started',
            'started_at': datetime.now().isoformat()
        }

        try:
            # Step 1: Extract training data
            training_csv = self.extract_training_data(winner)
            if not training_csv:
                result['status'] = 'failed'
                result['error'] = 'training_data_extraction_failed'
                return result
            result['training_csv'] = str(training_csv)

            # Step 2: Train ML model
            model_path = self.train_ml_model(winner, training_csv)
            if not model_path:
                result['status'] = 'failed'
                result['error'] = 'ml_training_failed'
                return result
            result['model_path'] = str(model_path)

            # Step 3: Validate on 2022-2024
            validation_results = self.validate_strategy(winner, model_path)
            result.update(validation_results)

            result['status'] = 'completed'
            result['completed_at'] = datetime.now().isoformat()

            logger.info("")
            logger.info(f"✅ {symbol} - {strategy_name} COMPLETE")
            logger.info("")

        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def run(
        self,
        strategy_ids: List[int] = None,
        symbols: List[str] = None
    ):
        """
        Run ML pipeline for all (or filtered) winners.

        Args:
            strategy_ids: Optional filter by strategy IDs
            symbols: Optional filter by symbols
        """
        # Filter winners
        winners_to_process = self.winners

        if strategy_ids:
            winners_to_process = [
                w for w in winners_to_process
                if w['strategy_id'] in strategy_ids
            ]

        if symbols:
            winners_to_process = [
                w for w in winners_to_process
                if w['symbol'] in symbols
            ]

        logger.info("=" * 80)
        logger.info("ML PIPELINE AUTOMATION")
        logger.info("=" * 80)
        logger.info(f"Total winners in manifest: {len(self.winners)}")
        logger.info(f"Winners to process: {len(winners_to_process)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Workers: {self.workers}")
        logger.info("")

        # Process winners
        results = []
        for i, winner in enumerate(winners_to_process, 1):
            logger.info(f"Progress: {i}/{len(winners_to_process)}")
            result = self.process_winner(winner)
            results.append(result)

        # Save results
        results_file = self.reports_dir / f"ml_pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("")
        logger.info("=" * 80)
        logger.info("ML PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved: {results_file}")
        logger.info("")

        # Summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')

        logger.info(f"Summary:")
        logger.info(f"  ✅ Completed: {completed}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info("")

        if failed > 0:
            logger.info("Failed strategies:")
            for r in results:
                if r['status'] == 'failed':
                    logger.info(f"  - {r['symbol']} {r['strategy_name']}: {r.get('error', 'unknown')}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Automated ML pipeline for strategy factory winners'
    )
    parser.add_argument(
        '--manifest', type=str, required=True,
        help='Path to winners manifest JSON (from extract_winners.py)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='ml_pipeline',
        help='Output directory for all ML pipeline artifacts'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--strategies', nargs='+', type=int,
        help='Filter by strategy IDs (optional)'
    )
    parser.add_argument(
        '--symbols', nargs='+',
        help='Filter by symbols (optional)'
    )
    parser.add_argument(
        '--no-skip-existing', action='store_true',
        help='Reprocess even if outputs exist'
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run extract_winners.py first!")
        return 1

    output_dir = Path(args.output_dir)

    runner = MLPipelineRunner(
        manifest_path=manifest_path,
        output_dir=output_dir,
        workers=args.workers,
        skip_existing=not args.no_skip_existing
    )

    results = runner.run(
        strategy_ids=args.strategies,
        symbols=args.symbols
    )

    # Exit code based on results
    failed = sum(1 for r in results if r['status'] == 'failed')
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
