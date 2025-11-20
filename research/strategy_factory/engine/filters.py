"""
Statistical filters for Strategy Factory Phase 1.

Applies rigorous statistical tests to backtest results:
- Walk-forward validation (train/test split)
- Regime analysis (bull/bear/sideways)
- Parameter stability testing
- Monte Carlo permutation tests
- False Discovery Rate (FDR) corrections
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .backtester import BacktestResults, Backtester
from .data_loader import load_data
from ..strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result from a filter test."""
    passed: bool
    score: float  # 0-1 scale
    details: Dict[str, Any]


class WalkForwardFilter:
    """
    Walk-forward validation filter.

    Tests if strategy performance degrades on out-of-sample data.

    Split: 2010-2021 (train) vs 2022-2024 (test)
    Pass criteria:
    - Train Sharpe ≥ 0.3
    - Test Sharpe ≥ 0.15
    - Test/Train ratio ≥ 0.5 (max 50% degradation)
    """

    def __init__(
        self,
        train_start: str = "2010-01-01",
        train_end: str = "2021-12-31",
        test_start: str = "2022-01-01",
        test_end: str = "2024-12-31"
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def apply(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str = "15min"
    ) -> FilterResult:
        """
        Run walk-forward validation.

        Args:
            strategy: Strategy instance with parameters set
            symbol: Symbol to test
            timeframe: Data timeframe

        Returns:
            FilterResult with pass/fail and metrics
        """
        from .data_loader import load_data

        # Load train data
        train_data = load_data(symbol, timeframe, self.train_start, self.train_end)

        # Load test data
        test_data = load_data(symbol, timeframe, self.test_start, self.test_end)

        # Run backtests
        backtester = Backtester()
        train_results = backtester.run(strategy, train_data, symbol)
        test_results = backtester.run(strategy, test_data, symbol)

        # Calculate metrics
        train_sharpe = train_results.sharpe_ratio
        test_sharpe = test_results.sharpe_ratio

        # Avoid division by zero
        if train_sharpe <= 0:
            degradation_ratio = 0.0
        else:
            degradation_ratio = test_sharpe / train_sharpe

        # Pass criteria
        passed = (
            train_sharpe >= 0.3 and
            test_sharpe >= 0.15 and
            degradation_ratio >= 0.5 and
            test_sharpe > 0  # Must be profitable OOS
        )

        # Score (0-1): weighted combination
        score = 0.0
        if train_sharpe > 0:
            score += 0.3 * min(train_sharpe / 1.0, 1.0)  # Train performance
        if test_sharpe > 0:
            score += 0.4 * min(test_sharpe / 0.5, 1.0)   # Test performance
        score += 0.3 * min(degradation_ratio, 1.0)       # Consistency

        details = {
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'train_trades': train_results.total_trades,
            'test_trades': test_results.total_trades,
            'degradation_ratio': degradation_ratio,
            'degradation_pct': (1 - degradation_ratio) * 100
        }

        logger.info(
            f"Walk-forward: Train Sharpe={train_sharpe:.3f}, "
            f"Test Sharpe={test_sharpe:.3f}, Degradation={degradation_ratio:.1%}"
        )

        return FilterResult(passed=passed, score=score, details=details)


class RegimeFilter:
    """
    Regime analysis filter.

    Tests if strategy performs consistently across different market regimes.

    Regimes:
    - Bull: 2010-2019, 2023-2024 (low VIX, uptrend)
    - Bear: 2020 (COVID), 2022 (rate hikes)
    - Sideways: 2015-2016 (range-bound)

    Pass criteria:
    - Sharpe ≥ 0.2 in at least 2 of 3 regimes
    - No regime with Sharpe < -0.3 (avoid catastrophic failures)
    """

    def __init__(self):
        self.regimes = {
            'bull': [
                ('2010-01-01', '2019-12-31'),  # Long bull run
                ('2023-01-01', '2024-12-31')   # Post-2022 recovery
            ],
            'bear': [
                ('2020-02-01', '2020-06-30'),  # COVID crash
                ('2022-01-01', '2022-10-31')   # Rate hike selloff
            ],
            'sideways': [
                ('2015-01-01', '2016-12-31')   # Range-bound
            ]
        }

    def apply(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str = "15min"
    ) -> FilterResult:
        """
        Run regime analysis.

        Args:
            strategy: Strategy instance with parameters set
            symbol: Symbol to test
            timeframe: Data timeframe

        Returns:
            FilterResult with pass/fail and metrics
        """
        from .data_loader import load_data

        backtester = Backtester()
        regime_results = {}

        for regime_name, periods in self.regimes.items():
            regime_trades = []

            for start, end in periods:
                data = load_data(symbol, timeframe, start, end)
                results = backtester.run(strategy, data, symbol)
                regime_trades.extend(results.trades)

            # Calculate regime metrics
            if regime_trades:
                pnls = [t.pnl for t in regime_trades]
                sharpe = self._calculate_sharpe(pnls)
                win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
            else:
                sharpe = 0.0
                win_rate = 0.0

            regime_results[regime_name] = {
                'sharpe': sharpe,
                'trades': len(regime_trades),
                'win_rate': win_rate
            }

        # Extract Sharpe ratios
        sharpes = [r['sharpe'] for r in regime_results.values()]

        # Pass criteria
        positive_regimes = sum(1 for s in sharpes if s >= 0.2)
        min_sharpe = min(sharpes)

        passed = (
            positive_regimes >= 2 and  # At least 2 of 3 regimes positive
            min_sharpe >= -0.3         # No catastrophic regime
        )

        # Score: consistency across regimes
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        consistency = 1 - min(std_sharpe / (abs(mean_sharpe) + 0.1), 1.0)
        score = consistency * 0.7 + (positive_regimes / 3) * 0.3

        details = {
            'regime_results': regime_results,
            'positive_regimes': positive_regimes,
            'min_sharpe': min_sharpe,
            'mean_sharpe': mean_sharpe,
            'std_sharpe': std_sharpe,
            'consistency_score': consistency
        }

        logger.info(
            f"Regime: Bull={regime_results['bull']['sharpe']:.3f}, "
            f"Bear={regime_results['bear']['sharpe']:.3f}, "
            f"Sideways={regime_results['sideways']['sharpe']:.3f}"
        )

        return FilterResult(passed=passed, score=score, details=details)

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(returns) < 2:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return mean / std


class ParameterStabilityFilter:
    """
    Parameter stability filter.

    Tests if strategy is robust to small parameter changes.

    Method:
    - For each parameter, test ±10% variations
    - Calculate Sharpe range relative to mean
    - Pass if variation < 40%

    This prevents overfitted strategies that collapse with small tweaks.
    """

    def __init__(self, variation_pct: float = 0.10, max_variation: float = 0.40):
        self.variation_pct = variation_pct
        self.max_variation = max_variation

    def apply(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str = "15min",
        start_date: str = "2010-01-01",
        end_date: str = "2024-12-31"
    ) -> FilterResult:
        """
        Test parameter stability.

        Args:
            strategy: Strategy instance with parameters set
            symbol: Symbol to test
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date

        Returns:
            FilterResult with pass/fail and metrics
        """
        from .data_loader import load_data

        data = load_data(symbol, timeframe, start_date, end_date)
        backtester = Backtester()

        base_params = strategy.params.copy()
        param_results = {}

        for param_name, base_value in base_params.items():
            # Skip non-numeric parameters
            if not isinstance(base_value, (int, float)):
                continue

            # Skip exit parameters (already fixed in Phase 1)
            if param_name in ['stop_loss_atr', 'take_profit_atr', 'max_bars_held']:
                continue

            # Test variations
            variations = {
                'base': base_value,
                'minus_10': base_value * (1 - self.variation_pct),
                'plus_10': base_value * (1 + self.variation_pct)
            }

            sharpes = {}

            for var_name, var_value in variations.items():
                # Create strategy with varied parameter
                test_params = base_params.copy()
                test_params[param_name] = var_value

                # Re-instantiate strategy
                strategy_class = strategy.__class__
                test_strategy = strategy_class(params=test_params)

                # Backtest
                results = backtester.run(test_strategy, data, symbol)
                sharpes[var_name] = results.sharpe_ratio

            # Calculate stability metrics
            sharpe_values = list(sharpes.values())
            sharpe_range = max(sharpe_values) - min(sharpe_values)
            sharpe_mean = np.mean(sharpe_values)

            if sharpe_mean > 0:
                stability_ratio = sharpe_range / sharpe_mean
            else:
                stability_ratio = 999  # Very unstable if mean is negative/zero

            param_results[param_name] = {
                'sharpes': sharpes,
                'range': sharpe_range,
                'mean': sharpe_mean,
                'stability_ratio': stability_ratio,
                'stable': stability_ratio < self.max_variation
            }

        # Overall stability
        if not param_results:
            # No numeric parameters to test
            passed = True
            score = 1.0
        else:
            stability_ratios = [r['stability_ratio'] for r in param_results.values()]
            overall_stable = all(r['stable'] for r in param_results.values())
            mean_stability = np.mean(stability_ratios)

            passed = overall_stable
            score = max(0, 1 - (mean_stability / self.max_variation))

        details = {
            'param_results': param_results,
            'overall_stable': passed,
            'params_tested': len(param_results)
        }

        logger.info(
            f"Stability: {len([r for r in param_results.values() if r['stable']])}"
            f"/{len(param_results)} params stable"
        )

        return FilterResult(passed=passed, score=score, details=details)


class MonteCarloFilter:
    """
    Monte Carlo permutation test.

    Tests if Sharpe ratio is due to skill or luck.

    Method:
    1. Shuffle trade outcomes randomly (destroys temporal structure)
    2. Calculate Sharpe of shuffled returns
    3. Repeat 1000 times
    4. p-value = % of shuffled Sharpes ≥ actual Sharpe

    If p < 0.05: Edge is statistically significant
    """

    def __init__(self, n_simulations: int = 1000, alpha: float = 0.05):
        self.n_simulations = n_simulations
        self.alpha = alpha

    def apply(self, result: BacktestResults) -> FilterResult:
        """
        Run Monte Carlo permutation test.

        Args:
            result: BacktestResults to test

        Returns:
            FilterResult with pass/fail and p-value
        """
        if result.total_trades < 10:
            # Not enough trades for meaningful test
            return FilterResult(
                passed=False,
                score=0.0,
                details={'error': 'Insufficient trades for Monte Carlo test'}
            )

        # Get trade returns
        returns = np.array([t.pnl for t in result.trades])
        actual_sharpe = result.sharpe_ratio

        # Run simulations
        shuffled_sharpes = []
        for _ in range(self.n_simulations):
            shuffled = np.random.permutation(returns)
            shuffled_sharpe = self._calculate_sharpe(shuffled)
            shuffled_sharpes.append(shuffled_sharpe)

        # Calculate p-value
        p_value = np.mean(np.array(shuffled_sharpes) >= actual_sharpe)

        # Pass if p < alpha
        passed = p_value < self.alpha

        # Score: inverse of p-value (capped at 1.0)
        score = max(0, 1 - (p_value / self.alpha))

        details = {
            'actual_sharpe': actual_sharpe,
            'mean_shuffled_sharpe': np.mean(shuffled_sharpes),
            'std_shuffled_sharpe': np.std(shuffled_sharpes),
            'p_value': p_value,
            'n_simulations': self.n_simulations,
            'significant': passed
        }

        logger.info(
            f"Monte Carlo: Actual Sharpe={actual_sharpe:.3f}, "
            f"p-value={p_value:.4f}, Significant={passed}"
        )

        return FilterResult(passed=passed, score=score, details=details)

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return mean / std


class FDRFilter:
    """
    False Discovery Rate (FDR) correction.

    Controls for multiple testing problem.

    Method:
    - Apply Benjamini-Hochberg FDR correction to Monte Carlo p-values
    - Adjust p-values for multiple comparisons
    - Only keep strategies with corrected p < 0.05

    This prevents finding "winners" by pure chance when testing many strategies.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def apply(
        self,
        monte_carlo_results: List[Tuple[BacktestResults, FilterResult]]
    ) -> List[int]:
        """
        Apply FDR correction to multiple Monte Carlo results.

        Args:
            monte_carlo_results: List of (BacktestResults, MonteCarloFilterResult) tuples

        Returns:
            List of indices that passed FDR correction
        """
        # Extract p-values
        p_values = []
        for result, mc_filter in monte_carlo_results:
            if 'p_value' in mc_filter.details:
                p_values.append(mc_filter.details['p_value'])
            else:
                p_values.append(1.0)  # Failed tests get p=1.0

        if not p_values:
            return []

        # Apply FDR correction
        reject, corrected_p, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method='fdr_bh'  # Benjamini-Hochberg
        )

        # Get indices that passed
        passed_indices = [i for i, r in enumerate(reject) if r]

        logger.info(
            f"FDR: {len(passed_indices)}/{len(p_values)} strategies passed "
            f"after correction"
        )

        return passed_indices


if __name__ == "__main__":
    """Test filters on a sample strategy."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from strategy_factory.strategies import RSI2MeanReversion
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing Statistical Filters")
    print("=" * 80)
    print()

    # Create strategy
    strategy = RSI2MeanReversion(params={
        'rsi_length': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 65
    })

    print("Note: This is a dry run test. Actual data loading requires your server.")
    print("Filters are ready to use once data is available.")
