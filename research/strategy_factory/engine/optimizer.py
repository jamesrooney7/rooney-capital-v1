"""
Parameter grid generation and parallel optimization engine.

Generates all parameter combinations and distributes backtests across
multiple CPU cores for fast execution.
"""

from itertools import product
from typing import Dict, List, Any, Callable, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from tqdm import tqdm
import logging

from ..strategies.base import BaseStrategy
from .backtester import Backtester, BacktestResults
from .data_loader import load_data

logger = logging.getLogger(__name__)


def generate_parameter_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations from parameter grid.

    Args:
        param_grid: Dictionary mapping param names to lists of values
                    Example: {'rsi_length': [2, 3], 'threshold': [10, 20]}

    Returns:
        List of parameter dictionaries
        Example: [
            {'rsi_length': 2, 'threshold': 10},
            {'rsi_length': 2, 'threshold': 20},
            {'rsi_length': 3, 'threshold': 10},
            {'rsi_length': 3, 'threshold': 20}
        ]
    """
    # Get parameter names and values
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Generate cartesian product
    combinations = []
    for value_combo in product(*values):
        param_dict = dict(zip(keys, value_combo))
        combinations.append(param_dict)

    logger.info(f"Generated {len(combinations)} parameter combinations")

    return combinations


def _run_single_backtest(
    args: tuple,
    strategy_class: type,
    data_path: tuple,
    fixed_params: Dict[str, Any]
) -> BacktestResults:
    """
    Run a single backtest (helper for parallel execution).

    Args:
        args: Tuple of (param_dict, backtest_kwargs)
        strategy_class: Strategy class to instantiate
        data_path: Tuple of (symbol, timeframe, start_date, end_date)
        fixed_params: Fixed parameters (e.g., stop_loss_atr, take_profit_atr)

    Returns:
        BacktestResults object
    """
    param_dict = args

    # Merge with fixed params
    full_params = {**param_dict, **fixed_params}

    # Load data
    symbol, timeframe, start_date, end_date = data_path
    data = load_data(symbol, timeframe, start_date, end_date)

    # Create strategy instance
    strategy = strategy_class(params=full_params)

    # Run backtest
    backtester = Backtester(
        initial_capital=100000,
        commission_per_side=1.00,
        slippage_ticks=1.0  # 1 tick slippage per side
    )

    results = backtester.run(strategy, data, symbol)

    return results


class ParameterOptimizer:
    """
    Parallel parameter optimizer for strategy research.

    Generates parameter combinations and runs backtests in parallel
    across multiple CPU cores.
    """

    def __init__(
        self,
        strategy_class: type,
        symbol: str,
        timeframe: str = "15min",
        start_date: str = "2010-01-01",
        end_date: str = "2024-12-31",
        workers: Optional[int] = None
    ):
        """
        Initialize optimizer.

        Args:
            strategy_class: Strategy class to optimize
            symbol: Symbol to backtest
            timeframe: Data timeframe
            start_date: Start date for backtest
            end_date: End date for backtest
            workers: Number of parallel workers (default: CPU count)
        """
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.workers = workers or cpu_count()

        logger.info(
            f"Initialized optimizer for {strategy_class.__name__} "
            f"on {symbol} with {self.workers} workers"
        )

    def optimize(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> List[BacktestResults]:
        """
        Run optimization across all parameter combinations.

        Args:
            param_grid: Parameter grid (if None, uses strategy's param_grid)
            fixed_params: Fixed parameters (e.g., stop_loss_atr=1.0)

        Returns:
            List of BacktestResults, one per parameter combination
        """
        # Get parameter grid
        if param_grid is None:
            # Create temporary instance to get param_grid
            temp_strategy = self.strategy_class(params={})
            param_grid = temp_strategy.param_grid

        # Generate combinations
        combinations = generate_parameter_combinations(param_grid)

        logger.info(f"Starting optimization: {len(combinations)} backtests")

        # Fixed parameters (Phase 1: fixed exits)
        if fixed_params is None:
            fixed_params = {
                'stop_loss_atr': 1.0,
                'take_profit_atr': 1.0,
                'max_bars_held': 20,
                'auto_close_time': '16:00'
            }

        # Prepare data path
        data_path = (self.symbol, self.timeframe, self.start_date, self.end_date)

        # Parallel execution
        partial_func = partial(
            _run_single_backtest,
            strategy_class=self.strategy_class,
            data_path=data_path,
            fixed_params=fixed_params
        )

        results = []

        if self.workers == 1:
            # Single-threaded (for debugging)
            for params in tqdm(combinations, desc="Backtesting"):
                result = partial_func(params)
                results.append(result)
        else:
            # Multi-threaded
            with Pool(processes=self.workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(partial_func, combinations),
                        total=len(combinations),
                        desc=f"Backtesting {self.strategy_class.__name__}"
                    )
                )

        logger.info(f"Optimization complete: {len(results)} results")

        return results

    def optimize_multiple_strategies(
        self,
        strategies: List[tuple],  # List of (strategy_class, param_grid) tuples
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[BacktestResults]]:
        """
        Optimize multiple strategies in sequence.

        Args:
            strategies: List of (strategy_class, param_grid) tuples
            fixed_params: Fixed parameters for all strategies

        Returns:
            Dictionary mapping strategy name -> list of results
        """
        all_results = {}

        for strategy_class, param_grid in strategies:
            logger.info(f"Optimizing {strategy_class.__name__}...")

            results = self.optimize(param_grid, fixed_params)
            all_results[strategy_class.__name__] = results

        return all_results


def results_to_dataframe(results: List[BacktestResults]) -> pd.DataFrame:
    """
    Convert list of BacktestResults to DataFrame.

    Args:
        results: List of backtest results

    Returns:
        DataFrame with one row per result
    """
    records = []

    for result in results:
        record = result.to_dict()
        records.append(record)

    df = pd.DataFrame(records)

    # Sort by Sharpe ratio (descending)
    df = df.sort_values('sharpe_ratio', ascending=False)

    return df


def filter_results(
    results: List[BacktestResults],
    min_trades: int = 100,
    min_sharpe: float = 0.2,
    min_profit_factor: float = 1.15,
    max_drawdown_pct: float = 0.30,
    min_win_rate: float = 0.35
) -> List[BacktestResults]:
    """
    Apply Phase 1 filters to backtest results.

    Args:
        results: List of backtest results
        min_trades: Minimum number of trades
        min_sharpe: Minimum Sharpe ratio
        min_profit_factor: Minimum profit factor
        max_drawdown_pct: Maximum drawdown percentage
        min_win_rate: Minimum win rate

    Returns:
        Filtered list of results
    """
    filtered = []

    for result in results:
        # Apply filters
        if (result.total_trades >= min_trades and
            result.sharpe_ratio >= min_sharpe and
            result.profit_factor >= min_profit_factor and
            result.max_drawdown_pct <= max_drawdown_pct and
            result.win_rate >= min_win_rate):
            filtered.append(result)

    logger.info(
        f"Filtered results: {len(filtered)}/{len(results)} passed "
        f"(trades>={min_trades}, sharpe>={min_sharpe}, pf>={min_profit_factor})"
    )

    return filtered


if __name__ == "__main__":
    """
    Test parameter optimizer.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from strategy_factory.strategies.rsi2_mean_reversion import RSI2MeanReversion
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing Parameter Optimizer")
    print("=" * 80)
    print()

    # Test 1: Generate parameter combinations
    print("Test 1: Parameter Grid Generation")
    param_grid = {
        'rsi_length': [2, 3, 4],
        'rsi_oversold': [5, 10, 15],
        'rsi_overbought': [60, 65, 70, 75]
    }
    combinations = generate_parameter_combinations(param_grid)
    print(f"Generated {len(combinations)} combinations")
    print(f"First 3: {combinations[:3]}")
    print()

    # Test 2: Run optimization (comment out if no data available)
    # print("Test 2: Run Optimization (RSI(2) on ES)")
    # optimizer = ParameterOptimizer(
    #     strategy_class=RSI2MeanReversion,
    #     symbol="ES",
    #     timeframe="15min",
    #     start_date="2023-01-01",
    #     end_date="2023-12-31",
    #     workers=4
    # )
    # results = optimizer.optimize()
    # print(f"Completed {len(results)} backtests")
    # print(f"Best Sharpe: {max(r.sharpe_ratio for r in results):.3f}")
