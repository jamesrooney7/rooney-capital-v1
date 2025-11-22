#!/usr/bin/env python3
"""
Portfolio Optimizer - Strategy Selection

Selects the optimal combination of ML-enhanced strategies to maximize Sharpe ratio
while respecting risk constraints:
- Max drawdown ≤ $6,000
- Max daily loss ≤ $3,000
- Equal weighting: 1 contract per strategy

Optimization period: 2022-2023
Test period: 2024 (out-of-sample validation)

Usage:
    # Read validation results from ML pipeline
    python research/portfolio_optimization/portfolio_optimizer.py \
        --validation-dir research/ml_meta_labeling/results \
        --output portfolio_manifest.json

    # Or specify specific strategies to consider
    python research/portfolio_optimization/portfolio_optimizer.py \
        --validation-dir research/ml_meta_labeling/results \
        --strategies ES_21 NQ_21 GC_42 \
        --output portfolio_manifest.json

Author: Rooney Capital
Date: 2025-01-22
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import combinations
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio selection to maximize Sharpe while respecting risk constraints.

    Uses combinatorial optimization with greedy search to find best subset of strategies.
    """

    def __init__(
        self,
        validation_dir: Path,
        max_drawdown: float = 6000.0,
        daily_loss_limit: float = 3000.0,
        optimization_start: str = "2022-01-01",
        optimization_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = "2024-12-31"
    ):
        self.validation_dir = Path(validation_dir)
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.optimization_start = optimization_start
        self.optimization_end = optimization_end
        self.test_start = test_start
        self.test_end = test_end

        self.strategies = {}
        self.optimization_results = {}

    def load_strategy_results(self, strategy_filter: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load validation results for all strategies.

        Args:
            strategy_filter: List of strategy IDs to include (e.g., ['ES_21', 'NQ_37'])
                           If None, loads all available strategies

        Returns:
            Dictionary mapping strategy_id -> equity curve DataFrame
        """
        logger.info("Loading strategy validation results...")

        # Find all validation result directories
        result_dirs = [d for d in self.validation_dir.iterdir() if d.is_dir()]

        for result_dir in result_dirs:
            strategy_id = result_dir.name

            # Apply filter if specified
            if strategy_filter and strategy_id not in strategy_filter:
                continue

            # Look for validation equity curve CSV
            # Expected format: {symbol}_{strategy_id}_validation_equity.csv
            equity_files = list(result_dir.glob('*_validation_equity.csv'))

            if not equity_files:
                # Try alternate format: {symbol}_{strategy_id}_equity_curve.csv
                equity_files = list(result_dir.glob('*_equity_curve.csv'))

            if not equity_files:
                logger.warning(f"No equity curve found for {strategy_id}")
                continue

            equity_file = equity_files[0]

            try:
                df = pd.read_csv(equity_file)

                # Ensure required columns exist
                if 'date' not in df.columns or 'equity' not in df.columns:
                    logger.warning(f"Missing required columns in {equity_file}")
                    continue

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # Use daily_pnl if it exists, otherwise calculate from equity
                if 'daily_pnl' not in df.columns:
                    df['daily_pnl'] = df['equity'].diff().fillna(0)

                self.strategies[strategy_id] = df
                logger.info(f"  Loaded {strategy_id}: {len(df)} days")

            except Exception as e:
                logger.error(f"Error loading {equity_file}: {e}")
                continue

        logger.info(f"Loaded {len(self.strategies)} strategies")
        return self.strategies

    def calculate_portfolio_metrics(
        self,
        strategy_ids: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics for a given combination of strategies.

        Args:
            strategy_ids: List of strategy IDs to include in portfolio
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Dictionary with portfolio metrics
        """
        if not strategy_ids:
            return None

        # Combine equity curves
        combined_equity = None

        for strategy_id in strategy_ids:
            if strategy_id not in self.strategies:
                logger.warning(f"Strategy {strategy_id} not loaded")
                return None

            df = self.strategies[strategy_id].copy()
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if combined_equity is None:
                combined_equity = df[['date', 'daily_pnl']].copy()
                combined_equity.columns = ['date', 'portfolio_pnl']
            else:
                # Merge on date and sum P&L
                combined_equity = combined_equity.merge(
                    df[['date', 'daily_pnl']],
                    on='date',
                    how='outer'
                )
                combined_equity['portfolio_pnl'] = (
                    combined_equity['portfolio_pnl'].fillna(0) +
                    combined_equity['daily_pnl'].fillna(0)
                )
                combined_equity = combined_equity[['date', 'portfolio_pnl']]

        combined_equity = combined_equity.sort_values('date')
        combined_equity['equity'] = combined_equity['portfolio_pnl'].cumsum()

        # Calculate metrics
        returns = combined_equity['portfolio_pnl'].values
        equity = combined_equity['equity'].values

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Max daily loss
        max_daily_loss = abs(returns.min()) if len(returns) > 0 else 0.0

        # Total return
        total_return = equity[-1] if len(equity) > 0 else 0.0

        # Win rate (days with positive P&L)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

        # Average daily P&L
        avg_daily_pnl = returns.mean() if len(returns) > 0 else 0.0

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_daily_loss': max_daily_loss,
            'total_return': total_return,
            'avg_daily_pnl': avg_daily_pnl,
            'win_rate': win_rate,
            'num_days': len(returns),
            'num_strategies': len(strategy_ids)
        }

    def check_constraints(self, metrics: Dict[str, float]) -> bool:
        """Check if portfolio meets risk constraints."""
        if metrics is None:
            return False

        return (
            metrics['max_drawdown'] <= self.max_drawdown and
            metrics['max_daily_loss'] <= self.daily_loss_limit
        )

    def optimize_greedy(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Greedy optimization: start with best strategy, keep adding if improves Sharpe
        while respecting constraints.

        Returns:
            Tuple of (selected_strategy_ids, portfolio_metrics)
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("GREEDY OPTIMIZATION (2022-2023)")
        logger.info("=" * 80)
        logger.info(f"Max Drawdown Constraint: ${self.max_drawdown:,.0f}")
        logger.info(f"Daily Loss Limit: ${self.daily_loss_limit:,.0f}")
        logger.info("")

        available_strategies = list(self.strategies.keys())
        selected_strategies = []
        best_sharpe = -np.inf

        # Find best single strategy to start
        logger.info("Finding best starting strategy...")
        for strategy_id in available_strategies:
            metrics = self.calculate_portfolio_metrics(
                [strategy_id],
                self.optimization_start,
                self.optimization_end
            )

            if metrics and self.check_constraints(metrics):
                logger.info(
                    f"  {strategy_id}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                    f"DD=${metrics['max_drawdown']:,.0f}, "
                    f"MaxDailyLoss=${metrics['max_daily_loss']:,.0f}"
                )
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    selected_strategies = [strategy_id]

        if not selected_strategies:
            logger.error("No strategies meet constraints individually!")
            return [], {}

        logger.info(f"\nStarting with: {selected_strategies[0]} (Sharpe={best_sharpe:.3f})")

        # Iteratively add strategies that improve Sharpe
        remaining = [s for s in available_strategies if s not in selected_strategies]
        iteration = 1

        while remaining:
            logger.info(f"\nIteration {iteration}: Testing {len(remaining)} candidates...")

            best_addition = None
            best_new_sharpe = best_sharpe
            best_new_metrics = None

            for candidate in remaining:
                test_portfolio = selected_strategies + [candidate]
                metrics = self.calculate_portfolio_metrics(
                    test_portfolio,
                    self.optimization_start,
                    self.optimization_end
                )

                if metrics and self.check_constraints(metrics):
                    if metrics['sharpe_ratio'] > best_new_sharpe:
                        best_new_sharpe = metrics['sharpe_ratio']
                        best_addition = candidate
                        best_new_metrics = metrics

            if best_addition:
                selected_strategies.append(best_addition)
                best_sharpe = best_new_sharpe
                remaining.remove(best_addition)

                logger.info(
                    f"  ✅ Added {best_addition}: "
                    f"Sharpe={best_new_metrics['sharpe_ratio']:.3f}, "
                    f"DD=${best_new_metrics['max_drawdown']:,.0f}, "
                    f"MaxDailyLoss=${best_new_metrics['max_daily_loss']:,.0f}"
                )

                iteration += 1
            else:
                logger.info("  No more strategies improve Sharpe while meeting constraints")
                break

        # Final metrics
        final_metrics = self.calculate_portfolio_metrics(
            selected_strategies,
            self.optimization_start,
            self.optimization_end
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Selected {len(selected_strategies)} strategies:")
        for s in selected_strategies:
            logger.info(f"  - {s}")
        logger.info(f"\nPortfolio Metrics (2022-2023):")
        logger.info(f"  Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: ${final_metrics['max_drawdown']:,.0f}")
        logger.info(f"  Max Daily Loss: ${final_metrics['max_daily_loss']:,.0f}")
        logger.info(f"  Total Return: ${final_metrics['total_return']:,.0f}")
        logger.info(f"  Avg Daily P&L: ${final_metrics['avg_daily_pnl']:,.0f}")
        logger.info(f"  Win Rate: {final_metrics['win_rate']:.1%}")

        return selected_strategies, final_metrics

    def validate_on_2024(self, selected_strategies: List[str]) -> Dict[str, float]:
        """
        Test optimized portfolio on 2024 (out-of-sample).

        Args:
            selected_strategies: List of strategy IDs from optimization

        Returns:
            Dictionary with 2024 performance metrics
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("OUT-OF-SAMPLE VALIDATION (2024)")
        logger.info("=" * 80)

        metrics = self.calculate_portfolio_metrics(
            selected_strategies,
            self.test_start,
            self.test_end
        )

        if metrics:
            logger.info(f"Portfolio Metrics (2024):")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  Max Drawdown: ${metrics['max_drawdown']:,.0f}")
            logger.info(f"  Max Daily Loss: ${metrics['max_daily_loss']:,.0f}")
            logger.info(f"  Total Return: ${metrics['total_return']:,.0f}")
            logger.info(f"  Avg Daily P&L: ${metrics['avg_daily_pnl']:,.0f}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")

            # Note if constraints would be violated (but don't fail)
            if metrics['max_drawdown'] > self.max_drawdown:
                logger.warning(
                    f"  ⚠️  Max drawdown ${metrics['max_drawdown']:,.0f} "
                    f"exceeds ${self.max_drawdown:,.0f} constraint"
                )
            if metrics['max_daily_loss'] > self.daily_loss_limit:
                logger.warning(
                    f"  ⚠️  Max daily loss ${metrics['max_daily_loss']:,.0f} "
                    f"exceeds ${self.daily_loss_limit:,.0f} limit"
                )
        else:
            logger.error("Failed to calculate 2024 metrics")

        return metrics

    def generate_deployment_manifest(
        self,
        selected_strategies: List[str],
        optimization_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        output_path: Path
    ):
        """
        Generate deployment manifest for live trading.

        Args:
            selected_strategies: List of strategy IDs
            optimization_metrics: Performance on 2022-2023
            test_metrics: Performance on 2024
            output_path: Path to save manifest JSON
        """
        # Parse strategy IDs to extract symbol and strategy_id
        # Format: {SYMBOL}_{STRATEGY_ID} (e.g., 'ES_21')
        strategy_details = []
        for strat_id in selected_strategies:
            parts = strat_id.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                strategy_num = parts[1]
                strategy_details.append({
                    'strategy_id': strat_id,
                    'symbol': symbol,
                    'strategy_number': int(strategy_num),
                    'position_size': 1  # Always 1 contract
                })

        manifest = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'optimization_config': {
                'max_drawdown': self.max_drawdown,
                'daily_loss_limit': self.daily_loss_limit,
                'optimization_period': f"{self.optimization_start} to {self.optimization_end}",
                'test_period': f"{self.test_start} to {self.test_end}"
            },
            'selected_strategies': strategy_details,
            'performance': {
                'optimization_period': optimization_metrics,
                'test_period': test_metrics
            },
            'risk_management': {
                'position_size': 1,
                'daily_loss_limit': self.daily_loss_limit,
                'shutdown_on_daily_loss': True,
                'max_drawdown_alert': self.max_drawdown
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info("")
        logger.info(f"✅ Deployment manifest saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio optimizer for strategy selection'
    )
    parser.add_argument(
        '--validation-dir', type=str,
        default='research/ml_meta_labeling/results',
        help='Directory containing strategy validation results'
    )
    parser.add_argument(
        '--strategies', nargs='+',
        help='Specific strategies to consider (e.g., ES_21 NQ_37). If not specified, uses all.'
    )
    parser.add_argument(
        '--max-drawdown', type=float, default=6000.0,
        help='Maximum portfolio drawdown in dollars (default: 6000)'
    )
    parser.add_argument(
        '--daily-loss-limit', type=float, default=3000.0,
        help='Daily loss limit in dollars (default: 3000)'
    )
    parser.add_argument(
        '--output', type=str, default='research/portfolio_optimization/portfolio_manifest.json',
        help='Output path for deployment manifest'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PORTFOLIO OPTIMIZER - STRATEGY SELECTION")
    logger.info("=" * 80)
    logger.info(f"Validation Directory: {args.validation_dir}")
    logger.info(f"Max Drawdown Constraint: ${args.max_drawdown:,.0f}")
    logger.info(f"Daily Loss Limit: ${args.daily_loss_limit:,.0f}")
    logger.info("")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        validation_dir=args.validation_dir,
        max_drawdown=args.max_drawdown,
        daily_loss_limit=args.daily_loss_limit
    )

    # Load strategy results
    optimizer.load_strategy_results(strategy_filter=args.strategies)

    if not optimizer.strategies:
        logger.error("No strategies loaded! Check validation directory.")
        return 1

    # Run optimization
    selected_strategies, optimization_metrics = optimizer.optimize_greedy()

    if not selected_strategies:
        logger.error("Optimization failed to find valid portfolio!")
        return 1

    # Validate on 2024
    test_metrics = optimizer.validate_on_2024(selected_strategies)

    # Generate deployment manifest
    optimizer.generate_deployment_manifest(
        selected_strategies,
        optimization_metrics,
        test_metrics,
        Path(args.output)
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PORTFOLIO OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info(f"1. Review manifest: {args.output}")
    logger.info("2. Deploy selected strategies to live trading")
    logger.info("3. Enable daily loss limit monitoring")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
