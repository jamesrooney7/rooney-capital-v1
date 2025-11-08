#!/usr/bin/env python3
"""
Automated Strategy Optimization Pipeline

Automates the complete strategy development workflow:
1. Skip: Strategy code generation (manual for now)
2. Backtest all symbols (2010-2025)
3. Extract training features
4. Train ML models (parallel)
5. Portfolio greedy optimization
6. Save to consolidated results

Usage:
    python research/auto_optimize_strategy.py --strategy breakout
    python research/auto_optimize_strategy.py --strategy breakout --symbols ES NQ CL
    python research/auto_optimize_strategy.py --strategy breakout --parallel-jobs 8
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default symbols (all 18 with ML models)
DEFAULT_SYMBOLS = [
    'ES', 'NQ', 'RTY', 'YM',  # Equities
    '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S',  # Currencies
    'CL', 'NG', 'GC', 'SI', 'HG', 'PL'  # Commodities
]

# Date ranges
BACKTEST_START = '2010-01-01'
BACKTEST_END = '2025-01-01'
FEATURE_SELECTION_END = '2020-12-31'
HOLDOUT_START = '2023-01-01'
TRAIN_START = '2023-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'


class StrategyPipeline:
    """Automated strategy optimization pipeline."""

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        parallel_jobs: int = 16,
        max_positions: int = 4,
        max_dd_limit: float = 5000.0,
        initial_capital: float = 150000.0,
        daily_stop_loss: float = 2500.0
    ):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.parallel_jobs = parallel_jobs
        self.max_positions = max_positions
        self.max_dd_limit = max_dd_limit
        self.initial_capital = initial_capital
        self.daily_stop_loss = daily_stop_loss

        # Paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.results_dir = self.project_root / 'results' / f'{strategy_name}_optimization'
        self.models_dir = self.project_root / 'src' / 'models'

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline initialized for strategy: {strategy_name}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Parallel jobs: {parallel_jobs}")

    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 80)
        logger.info(f"STARTING AUTOMATED PIPELINE: {self.strategy_name}")
        logger.info("=" * 80)
        logger.info("")

        start_time = datetime.now()

        try:
            # Step 2: Backtest
            self.step_2_backtest()

            # Step 3: Feature extraction
            self.step_3_feature_extraction()

            # Step 4: ML training (parallel)
            self.step_4_ml_training()

            # Step 5: Portfolio optimization
            results = self.step_5_portfolio_optimization()

            # Done!
            elapsed = datetime.now() - start_time
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"✅ PIPELINE COMPLETE: {self.strategy_name}")
            logger.info("=" * 80)
            logger.info(f"Elapsed time: {elapsed}")
            logger.info(f"Test Sharpe: {results['test_sharpe']:.3f}")
            logger.info(f"Optimal symbols: {', '.join(results['optimal_symbols'])}")
            logger.info(f"Max positions: {results['max_positions']}")
            logger.info(f"Results: results/all_optimizations.json")
            logger.info("")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def step_2_backtest(self):
        """Step 2: Run backtests for all symbols."""
        logger.info("=" * 80)
        logger.info("STEP 2: BACKTESTING")
        logger.info("=" * 80)
        logger.info(f"Running backtests for {len(self.symbols)} symbols...")
        logger.info(f"Date range: {BACKTEST_START} to {BACKTEST_END}")
        logger.info("")

        # Note: For now, we assume backtest_runner.py generates trade CSVs
        # In the future, this could be automated

        logger.info("⚠️  Manual step required:")
        logger.info(f"   Run backtests for strategy '{self.strategy_name}' on all symbols")
        logger.info(f"   Ensure trade CSVs are in: {self.results_dir}")
        logger.info("")
        logger.info("Expected files:")
        for symbol in self.symbols:
            logger.info(f"   - {self.results_dir}/{symbol}_rf_best_trades.csv")
        logger.info("")

        # Check if trade files exist
        missing = []
        for symbol in self.symbols:
            trade_file = self.results_dir / f"{symbol}_rf_best_trades.csv"
            if not trade_file.exists():
                missing.append(symbol)

        if missing:
            logger.error(f"Missing trade files for: {', '.join(missing)}")
            logger.error("Please run backtests first before continuing.")
            logger.info("")
            logger.info("Example commands:")
            for symbol in missing[:3]:  # Show first 3
                logger.info(f"  python research/backtest_runner.py --symbol {symbol} --start {BACKTEST_START}")
            sys.exit(1)

        logger.info(f"✓ All {len(self.symbols)} symbols have trade files")
        logger.info("")

    def step_3_feature_extraction(self):
        """Step 3: Extract training features for all symbols."""
        logger.info("=" * 80)
        logger.info("STEP 3: FEATURE EXTRACTION")
        logger.info("=" * 80)
        logger.info(f"Extracting features for {len(self.symbols)} symbols...")
        logger.info("")

        training_dir = self.data_dir / 'training'
        training_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0

        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Extracting features: {symbol}")

            # Check if already exists
            output_file = training_dir / f"{symbol}_transformed_features.csv"
            if output_file.exists():
                logger.info(f"  ✓ Features already exist: {output_file.name}")
                success_count += 1
                continue

            # Run extraction
            cmd = [
                'python', 'research/extract_training_data.py',
                '--symbol', symbol,
                '--output', str(output_file)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 min timeout
                )

                if result.returncode == 0:
                    logger.info(f"  ✓ Features extracted: {output_file.name}")
                    success_count += 1
                else:
                    logger.error(f"  ✗ Failed to extract features for {symbol}")
                    logger.error(f"    {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                logger.error(f"  ✗ Timeout extracting features for {symbol}")
            except Exception as e:
                logger.error(f"  ✗ Error extracting features for {symbol}: {e}")

        logger.info("")
        logger.info(f"✓ Feature extraction complete: {success_count}/{len(self.symbols)} successful")
        logger.info("")

        if success_count == 0:
            raise RuntimeError("No features extracted. Pipeline cannot continue.")

    def step_4_ml_training(self):
        """Step 4: Train ML models in parallel."""
        logger.info("=" * 80)
        logger.info("STEP 4: ML TRAINING (PARALLEL)")
        logger.info("=" * 80)
        logger.info(f"Training {len(self.symbols)} models with {self.parallel_jobs} parallel jobs...")
        logger.info("")

        # Use the existing parallel_optimization.sh script approach
        # Build list of CSV files
        training_dir = self.data_dir / 'training'
        csv_files = []

        for symbol in self.symbols:
            csv_file = training_dir / f"{symbol}_transformed_features.csv"
            if csv_file.exists():
                csv_files.append(str(csv_file))
            else:
                logger.warning(f"Missing features for {symbol}: {csv_file}")

        if not csv_files:
            raise RuntimeError("No training feature files found")

        logger.info(f"Found {len(csv_files)} feature files to train")
        logger.info("")

        # Run parallel training using xargs (same approach as parallel_optimization.sh)
        bash_script = f"""
#!/bin/bash
set -e

PROJECT_ROOT="{self.project_root}"
RESULTS_DIR="{self.results_dir}"
FEATURE_SELECTION_END="{FEATURE_SELECTION_END}"
HOLDOUT_START="{HOLDOUT_START}"
MAX_JOBS={self.parallel_jobs}

optimize_symbol() {{
    local csv_path=$1
    local symbol=$(basename "$csv_path" | sed 's/_transformed_features.csv//')
    local output_dir="$RESULTS_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training $symbol..."

    cd "$PROJECT_ROOT"

    PYTHONPATH="$PROJECT_ROOT/research:$PROJECT_ROOT/src:$PYTHONPATH" \\
    python3 research/rf_cpcv_random_then_bo.py \\
        --input "$csv_path" \\
        --outdir "$output_dir" \\
        --symbol "$symbol" \\
        --screen_method clustered \\
        --n_clusters 15 \\
        --features_per_cluster 2 \\
        --feature_selection_end "$FEATURE_SELECTION_END" \\
        --holdout_start "$HOLDOUT_START" \\
        --rs_trials 25 \\
        --bo_trials 65 \\
        --folds 5 \\
        --k_test 2 \\
        --embargo_days 2 \\
        2>&1 | tee "$output_dir/optimization_${{symbol}}.log"

    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed $symbol"

        # Copy models to src/models/
        if [ -f "$output_dir/${{symbol}}_best.json" ]; then
            cp "$output_dir/${{symbol}}_best.json" "$PROJECT_ROOT/src/models/"
            cp "$output_dir/${{symbol}}_rf_model.pkl" "$PROJECT_ROOT/src/models/"
            echo "  → Models copied to src/models/"
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed $symbol"
    fi
}}

export -f optimize_symbol
export PROJECT_ROOT RESULTS_DIR FEATURE_SELECTION_END HOLDOUT_START

# Run in parallel
printf '%s\\n' {' '.join([f'"{f}"' for f in csv_files])} | xargs -P "$MAX_JOBS" -I {{}} bash -c 'optimize_symbol "$@"' _ {{}}
"""

        # Write and execute bash script
        script_file = self.project_root / 'tmp_parallel_train.sh'
        script_file.write_text(bash_script)
        script_file.chmod(0o755)

        try:
            logger.info("Starting parallel ML training...")
            logger.info("")

            result = subprocess.run(
                ['bash', str(script_file)],
                cwd=self.project_root,
                timeout=3600 * 4  # 4 hour timeout
            )

            if result.returncode == 0:
                logger.info("")
                logger.info("✓ ML training complete")
                logger.info("")
            else:
                raise RuntimeError(f"ML training failed with exit code {result.returncode}")

        finally:
            # Cleanup
            if script_file.exists():
                script_file.unlink()

        # Verify models were created
        models_created = 0
        for symbol in self.symbols:
            model_file = self.models_dir / f"{symbol}_best.json"
            if model_file.exists():
                models_created += 1

        logger.info(f"Models created: {models_created}/{len(self.symbols)}")
        logger.info("")

        if models_created == 0:
            raise RuntimeError("No ML models created. Pipeline cannot continue.")

    def step_5_portfolio_optimization(self) -> Dict:
        """Step 5: Run portfolio greedy optimization."""
        logger.info("=" * 80)
        logger.info("STEP 5: PORTFOLIO GREEDY OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Train period: {TRAIN_START} to {TRAIN_END}")
        logger.info(f"Test period: {TEST_START} to {TEST_END}")
        logger.info(f"Constraint: Max DD < ${self.max_dd_limit:,.0f}")
        logger.info("")

        cmd = [
            'python', 'research/portfolio_optimizer_greedy_train_test.py',
            '--results-dir', str(self.results_dir),
            '--train-start', TRAIN_START,
            '--train-end', TRAIN_END,
            '--test-start', TEST_START,
            '--test-end', TEST_END,
            '--min-positions', '1',
            '--max-positions', str(self.max_positions),
            '--max-dd-limit', str(self.max_dd_limit),
            '--initial-capital', str(self.initial_capital),
            '--daily-stop-loss', str(self.daily_stop_loss),
            '--symbols'] + self.symbols + [
            '--output-suffix', self.strategy_name,
            '--strategy-name', self.strategy_name,
            '--output-dir', 'results'
        ]

        logger.info("Running greedy optimizer...")
        logger.info("")

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Portfolio optimization failed with exit code {result.returncode}")

        logger.info("")
        logger.info("✓ Portfolio optimization complete")
        logger.info("")

        # Load results from consolidated file
        consolidated_file = self.project_root / 'results' / 'all_optimizations.json'
        if not consolidated_file.exists():
            raise RuntimeError("Consolidated results file not found")

        with open(consolidated_file) as f:
            all_results = json.load(f)

        # Find our strategy
        our_result = None
        for r in all_results:
            if r['strategy_name'] == self.strategy_name:
                our_result = r
                break

        if not our_result:
            raise RuntimeError(f"Strategy '{self.strategy_name}' not found in consolidated results")

        return our_result


def main():
    parser = argparse.ArgumentParser(
        description='Automated strategy optimization pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize new strategy on all symbols
  python research/auto_optimize_strategy.py --strategy breakout

  # Optimize on specific symbols
  python research/auto_optimize_strategy.py --strategy breakout --symbols ES NQ CL

  # Use fewer parallel jobs (for smaller servers)
  python research/auto_optimize_strategy.py --strategy breakout --parallel-jobs 8
"""
    )

    parser.add_argument('--strategy', type=str, required=True,
                       help='Strategy name (e.g., breakout, mean_reversion)')
    parser.add_argument('--symbols', type=str, nargs='+', default=DEFAULT_SYMBOLS,
                       help='Symbols to test (default: all 18)')
    parser.add_argument('--parallel-jobs', type=int, default=16,
                       help='Number of parallel ML training jobs (default: 16)')
    parser.add_argument('--max-positions', type=int, default=4,
                       help='Max positions for greedy optimizer (default: 4)')
    parser.add_argument('--max-dd-limit', type=float, default=5000.0,
                       help='Max drawdown limit in dollars (default: 5000)')
    parser.add_argument('--initial-capital', type=float, default=150000.0,
                       help='Initial capital (default: 150000)')
    parser.add_argument('--daily-stop-loss', type=float, default=2500.0,
                       help='Daily stop loss (default: 2500)')

    args = parser.parse_args()

    # Validate strategy code exists
    strategy_file = Path('src/strategy') / f'{args.strategy}_strategy.py'
    if not strategy_file.exists():
        logger.error(f"Strategy code not found: {strategy_file}")
        logger.error(f"Please create the strategy class first before running pipeline.")
        sys.exit(1)

    # Create and run pipeline
    pipeline = StrategyPipeline(
        strategy_name=args.strategy,
        symbols=args.symbols,
        parallel_jobs=args.parallel_jobs,
        max_positions=args.max_positions,
        max_dd_limit=args.max_dd_limit,
        initial_capital=args.initial_capital,
        daily_stop_loss=args.daily_stop_loss
    )

    results = pipeline.run()

    # Print final summary
    print("\n" + "=" * 80)
    print(f"STRATEGY: {args.strategy}")
    print("=" * 80)
    print(f"Test Sharpe:      {results['test_sharpe']:.3f}")
    print(f"Test CAGR:        {results['test_cagr']*100:.2f}%")
    print(f"Test Max DD:      ${results['test_max_dd']:,.0f}")
    print(f"Generalization:   {results['generalization']*100:.1f}%")
    print(f"Optimal Symbols:  {', '.join(results['optimal_symbols'])}")
    print(f"Max Positions:    {results['max_positions']}")
    print("=" * 80)
    print(f"\nView all results: results/all_optimizations.json")
    print("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
