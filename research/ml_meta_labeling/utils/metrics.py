"""
Metrics utilities for ML meta-labeling system.
"""

import numpy as np
import pandas as pd
from typing import Dict


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns
        periods_per_year: Annualization factor (252 for daily)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation).

    Args:
        returns: Array of returns
        periods_per_year: Annualization factor

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Cumulative equity curve

    Returns:
        Maximum drawdown (as positive fraction)
    """
    if len(equity_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    return drawdown.max()


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Annualization factor

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate annualized return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Calculate max drawdown
    equity = (1 + returns).cumprod()
    mdd = max_drawdown(equity)

    if mdd == 0:
        return 0.0

    return annualized_return / mdd


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Array of returns

    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return 0.0

    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())

    if gross_losses == 0:
        return np.inf if gross_profits > 0 else 0.0

    return gross_profits / gross_losses


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (fraction of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0

    return (returns > 0).mean()


def calculate_performance_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Array of returns
        periods_per_year: Annualization factor

    Returns:
        Dictionary of metrics
    """
    equity = (1 + returns).cumprod()

    metrics = {
        'sharpe_ratio': sharpe_ratio(returns, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown(equity),
        'profit_factor': profit_factor(returns),
        'win_rate': win_rate(returns),
        'total_return': equity[-1] - 1 if len(equity) > 0 else 0,
        'n_trades': len(returns),
        'mean_return': returns.mean(),
        'std_return': returns.std()
    }

    return metrics
