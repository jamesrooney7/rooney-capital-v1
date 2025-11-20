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


# Regression metrics for ML models

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score (1.0 = perfect, 0.0 = as good as mean baseline, negative = worse than mean)
    """
    if len(y_true) == 0:
        return 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE
    """
    if len(y_true) == 0:
        return 0.0

    return np.abs(y_true - y_pred).mean()


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE
    """
    if len(y_true) == 0:
        return 0.0

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE (as percentage, 0-100)
    """
    if len(y_true) == 0:
        return 0.0

    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mean_true': y_true.mean(),
        'mean_pred': y_pred.mean(),
        'std_true': y_true.std(),
        'std_pred': y_pred.std(),
        'n_samples': len(y_true)
    }

    return metrics
