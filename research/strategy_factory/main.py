#!/usr/bin/env python3
"""
Strategy Factory - Main CLI Entry Point

Usage:
    # Run Phase 1: Test all 10 strategies on ES
    python research/strategy_factory/main.py phase1 \
        --symbol ES \
        --start 2010-01-01 \
        --end 2024-12-31 \
        --workers 16

    # Run specific strategies only
    python research/strategy_factory/main.py phase1 \
        --symbol ES \
        --strategies 21 1 36 \
        --workers 16

    # Run Phase 2: Multi-symbol validation
    python research/strategy_factory/main.py phase2 \
        --run-id <phase1_run_id> \
        --symbols ES NQ YM RTY \
        --workers 16
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from research.strategy_factory.engine import load_data
from research.strategy_factory.engine.optimizer import ParameterOptimizer, filter_results
from research.strategy_factory.engine.filters import (
    WalkForwardFilter,
    RegimeFilter,
    ParameterStabilityFilter,
    MonteCarloFilter,
    FDRFilter
)
from research.strategy_factory.database import DatabaseManager
from research.strategy_factory.strategies import STRATEGY_REGISTRY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_factory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StrategyFactoryCLI:
    """Main CLI for Strategy Factory."""

    def __init__(self):
        self.db = DatabaseManager()

    def run_phase1(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "15min",
        workers: int = 16,
        strategy_ids: Optional[List[int]] = None
    ):
        """
        Run Phase 1: Raw strategy screening.

        Args:
            symbol: Symbol to test (e.g., 'ES')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            workers: Number of parallel workers
            strategy_ids: Optional list of strategy IDs to test (None = all)
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: RAW STRATEGY SCREENING")
        logger.info("=" * 80)

        start_time = time.time()

        # Create run
        run_id = self.db.create_run(
            phase=1,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            workers=workers
        )

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info(f"Workers: {workers}")
        logger.info("")

        # Get strategies to test
        if strategy_ids:
            strategies_to_test = {
                sid: STRATEGY_REGISTRY[sid] for sid in strategy_ids
                if sid in STRATEGY_REGISTRY
            }
        else:
            strategies_to_test = STRATEGY_REGISTRY

        logger.info(f"Testing {len(strategies_to_test)} strategies:")
        for sid, strategy_class in strategies_to_test.items():
            logger.info(f"  - #{sid}: {strategy_class.__name__}")
        logger.info("")

        # Run optimization for each strategy
        all_results = []
        strategies_tested = 0
        total_backtests = 0

        for strategy_id, strategy_class in strategies_to_test.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimizing Strategy #{strategy_id}: {strategy_class.__name__}")
            logger.info(f"{'='*80}")

            optimizer = ParameterOptimizer(
                strategy_class=strategy_class,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                workers=workers
            )

            try:
                results = optimizer.optimize()
                all_results.extend(results)
                total_backtests += len(results)
                strategies_tested += 1

                # Save to database
                self.db.save_backtest_results_batch(run_id, results)

                logger.info(f"✓ Completed {len(results)} backtests for {strategy_class.__name__}")

            except Exception as e:
                logger.error(f"✗ Failed to optimize {strategy_class.__name__}: {e}")
                continue

        logger.info(f"\n{'='*80}")
        logger.info(f"RAW BACKTESTING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Backtests: {total_backtests}")
        logger.info(f"Strategies Tested: {strategies_tested}")
        logger.info("")

        # Apply Gate 1 filters
        logger.info("Applying Gate 1 Filters (Basic Criteria)...")
        gate1_survivors = filter_results(
            all_results,
            min_trades=10000,
            min_sharpe=0.2,
            min_profit_factor=1.15,
            max_drawdown_pct=0.30,
            min_win_rate=0.35
        )
        logger.info(f"Gate 1: {len(gate1_survivors)}/{total_backtests} passed")

        if not gate1_survivors:
            logger.warning("No strategies passed Gate 1. Exiting.")
            self.db.finalize_run(
                run_id, strategies_tested, total_backtests, 0,
                time.time() - start_time
            )
            return

        # Apply Walk-Forward validation
        logger.info("\nApplying Walk-Forward Validation...")
        walkforward_filter = WalkForwardFilter()
        walkforward_survivors = []

        for result in gate1_survivors:
            # Recreate strategy instance
            strategy_class = STRATEGY_REGISTRY[result.strategy_id]
            strategy = strategy_class(params=result.params)

            try:
                wf_result = walkforward_filter.apply(strategy, symbol, timeframe)
                if wf_result.passed:
                    walkforward_survivors.append(result)
                    logger.info(f"  ✓ {result.strategy_name} passed walk-forward")
                else:
                    logger.info(f"  ✗ {result.strategy_name} failed walk-forward")
            except Exception as e:
                logger.error(f"  ✗ {result.strategy_name} error: {e}")

        logger.info(f"Walk-Forward: {len(walkforward_survivors)}/{len(gate1_survivors)} passed")

        if not walkforward_survivors:
            logger.warning("No strategies passed walk-forward. Exiting.")
            self.db.finalize_run(
                run_id, strategies_tested, total_backtests, 0,
                time.time() - start_time
            )
            return

        # Apply Regime Analysis
        logger.info("\nApplying Regime Analysis...")
        regime_filter = RegimeFilter()
        regime_survivors = []

        for result in walkforward_survivors:
            strategy_class = STRATEGY_REGISTRY[result.strategy_id]
            strategy = strategy_class(params=result.params)

            try:
                regime_result = regime_filter.apply(strategy, symbol, timeframe)
                if regime_result.passed:
                    regime_survivors.append(result)
                    logger.info(f"  ✓ {result.strategy_name} passed regime analysis")
                else:
                    logger.info(f"  ✗ {result.strategy_name} failed regime analysis")
            except Exception as e:
                logger.error(f"  ✗ {result.strategy_name} error: {e}")

        logger.info(f"Regime: {len(regime_survivors)}/{len(walkforward_survivors)} passed")

        if not regime_survivors:
            logger.warning("No strategies passed regime analysis. Exiting.")
            self.db.finalize_run(
                run_id, strategies_tested, total_backtests, 0,
                time.time() - start_time
            )
            return

        # Apply Parameter Stability
        logger.info("\nApplying Parameter Stability Test...")
        stability_filter = ParameterStabilityFilter()
        stability_survivors = []

        for result in regime_survivors:
            strategy_class = STRATEGY_REGISTRY[result.strategy_id]
            strategy = strategy_class(params=result.params)

            try:
                stability_result = stability_filter.apply(
                    strategy, symbol, timeframe, start_date, end_date
                )
                if stability_result.passed:
                    stability_survivors.append(result)
                    logger.info(f"  ✓ {result.strategy_name} passed stability test")
                else:
                    logger.info(f"  ✗ {result.strategy_name} failed stability test")
            except Exception as e:
                logger.error(f"  ✗ {result.strategy_name} error: {e}")

        logger.info(f"Stability: {len(stability_survivors)}/{len(regime_survivors)} passed")

        if not stability_survivors:
            logger.warning("No strategies passed stability test.")
            self.db.finalize_run(
                run_id, strategies_tested, total_backtests, 0,
                time.time() - start_time
            )
            return

        # Apply Monte Carlo + FDR
        logger.info("\nApplying Monte Carlo Permutation Tests...")
        monte_carlo_filter = MonteCarloFilter(n_simulations=1000)
        monte_carlo_results = []

        for result in stability_survivors:
            try:
                mc_result = monte_carlo_filter.apply(result)
                monte_carlo_results.append((result, mc_result))
                if mc_result.passed:
                    logger.info(
                        f"  ✓ {result.strategy_name} p={mc_result.details['p_value']:.4f}"
                    )
                else:
                    logger.info(
                        f"  ✗ {result.strategy_name} p={mc_result.details['p_value']:.4f}"
                    )
            except Exception as e:
                logger.error(f"  ✗ {result.strategy_name} error: {e}")

        # Apply FDR correction
        logger.info("\nApplying False Discovery Rate Correction...")
        fdr_filter = FDRFilter(alpha=0.05)
        fdr_passed_indices = fdr_filter.apply(monte_carlo_results)

        final_winners = [
            monte_carlo_results[i][0] for i in fdr_passed_indices
        ]

        logger.info(f"FDR: {len(final_winners)}/{len(stability_survivors)} passed")

        # Summary
        elapsed = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1 COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Runtime: {elapsed/60:.1f} minutes")
        logger.info(f"Total Backtests: {total_backtests}")
        logger.info("")
        logger.info("Filter Results:")
        logger.info(f"  Gate 1 (Basic):        {len(gate1_survivors):3d} / {total_backtests}")
        logger.info(f"  Walk-Forward:          {len(walkforward_survivors):3d} / {len(gate1_survivors)}")
        logger.info(f"  Regime Consistency:    {len(regime_survivors):3d} / {len(walkforward_survivors)}")
        logger.info(f"  Parameter Stability:   {len(stability_survivors):3d} / {len(regime_survivors)}")
        logger.info(f"  Statistical (MC+FDR):  {len(final_winners):3d} / {len(stability_survivors)}")
        logger.info("")
        logger.info(f"FINAL WINNERS: {len(final_winners)} strategies")

        if final_winners:
            logger.info("\nWinning Strategies:")
            for i, result in enumerate(final_winners, 1):
                logger.info(
                    f"  {i}. {result.strategy_name} (#{result.strategy_id}): "
                    f"Sharpe={result.sharpe_ratio:.3f}, "
                    f"Trades={result.total_trades:,}, "
                    f"PF={result.profit_factor:.2f}"
                )
        else:
            logger.warning("\n⚠️  No strategies passed all filters!")
            logger.info("Recommendations:")
            logger.info("  - Review filter thresholds (may be too strict)")
            logger.info("  - Check data quality and date range")
            logger.info("  - Consider testing on different symbols")

        # Finalize run
        self.db.finalize_run(
            run_id, strategies_tested, total_backtests,
            len(final_winners), elapsed
        )

        logger.info(f"\nResults saved to database with run_id: {run_id}")
        logger.info(f"Query results: python -m strategy_factory.database.manager --run-id {run_id}")

    def run_phase2(self, run_id: str, symbols: List[str], workers: int = 16):
        """
        Run Phase 2: Multi-symbol validation.

        Args:
            run_id: Phase 1 run ID to continue from
            symbols: List of symbols to test
            workers: Number of parallel workers
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: MULTI-SYMBOL VALIDATION")
        logger.info("=" * 80)
        logger.info("Status: Not yet implemented")
        logger.info("This will be built after Phase 1 is validated")

    def run_phase3(self, run_id: str):
        """
        Run Phase 3: ML integration.

        Args:
            run_id: Phase 2 run ID to continue from
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: ML INTEGRATION")
        logger.info("=" * 80)
        logger.info("Status: Not yet implemented")
        logger.info("This will use existing extract_training_data.py and train_rf_cpcv_bo.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Strategy Factory - Systematic Strategy Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 on ES (all strategies)
  python -m research.strategy_factory.main phase1 --symbol ES --workers 16

  # Run Phase 1 with specific strategies
  python -m research.strategy_factory.main phase1 --symbol ES --strategies 21 1 36

  # Run Phase 2 (after Phase 1)
  python -m research.strategy_factory.main phase2 --run-id <run_id> --symbols ES NQ YM RTY
        """
    )

    subparsers = parser.add_subparsers(dest='phase', help='Phase to run')

    # Phase 1
    phase1_parser = subparsers.add_parser('phase1', help='Run Phase 1: Raw strategy screening')
    phase1_parser.add_argument('--symbol', required=True, help='Symbol to test (e.g., ES)')
    phase1_parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    phase1_parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    phase1_parser.add_argument('--timeframe', default='15min', help='Data timeframe')
    phase1_parser.add_argument('--workers', type=int, default=16, help='Number of workers')
    phase1_parser.add_argument(
        '--strategies', nargs='+', type=int,
        help='Specific strategy IDs to test (default: all)'
    )

    # Phase 2
    phase2_parser = subparsers.add_parser('phase2', help='Run Phase 2: Multi-symbol validation')
    phase2_parser.add_argument('--run-id', required=True, help='Phase 1 run ID')
    phase2_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to test')
    phase2_parser.add_argument('--workers', type=int, default=16, help='Number of workers')

    # Phase 3
    phase3_parser = subparsers.add_parser('phase3', help='Run Phase 3: ML integration')
    phase3_parser.add_argument('--run-id', required=True, help='Phase 2 run ID')

    args = parser.parse_args()

    if not args.phase:
        parser.print_help()
        sys.exit(1)

    cli = StrategyFactoryCLI()

    try:
        if args.phase == 'phase1':
            cli.run_phase1(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe,
                workers=args.workers,
                strategy_ids=args.strategies
            )
        elif args.phase == 'phase2':
            cli.run_phase2(
                run_id=args.run_id,
                symbols=args.symbols,
                workers=args.workers
            )
        elif args.phase == 'phase3':
            cli.run_phase3(run_id=args.run_id)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
