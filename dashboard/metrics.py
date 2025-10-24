"""Performance metrics calculator for trading dashboard.

Calculates advanced metrics like Sharpe Ratio, Sortino Ratio, and Profit Factor.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np


def calculate_sharpe_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe Ratio.

    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        daily_returns: List of daily returns (as decimals, not percentages)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe Ratio
    """
    if not daily_returns or len(daily_returns) < 2:
        return 0.0

    returns_array = np.array(daily_returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    daily_rf = risk_free_rate / periods_per_year
    sharpe = (mean_return - daily_rf) / std_return * math.sqrt(periods_per_year)

    return float(sharpe)


def calculate_sortino_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino Ratio.

    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation

    Only considers downside volatility (negative returns).

    Args:
        daily_returns: List of daily returns (as decimals)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sortino Ratio
    """
    if not daily_returns or len(daily_returns) < 2:
        return 0.0

    returns_array = np.array(daily_returns)
    mean_return = np.mean(returns_array)

    # Calculate downside deviation (only negative returns)
    negative_returns = returns_array[returns_array < 0]

    if len(negative_returns) == 0:
        # No losing days - infinite Sortino
        return float("inf") if mean_return > 0 else 0.0

    downside_dev = np.std(negative_returns, ddof=1)

    if downside_dev == 0:
        return 0.0

    # Annualize
    daily_rf = risk_free_rate / periods_per_year
    sortino = (mean_return - daily_rf) / downside_dev * math.sqrt(periods_per_year)

    return float(sortino)


def calculate_profit_factor(trades: list[dict[str, Any]]) -> float:
    """Calculate Profit Factor.

    Profit Factor = Gross Profit / Gross Loss

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Profit factor (>1 is profitable, <1 is losing)
    """
    if not trades:
        return 0.0

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_max_drawdown(daily_pnl: dict[str, float]) -> tuple[float, float]:
    """Calculate maximum drawdown from daily P&L.

    Args:
        daily_pnl: Dictionary mapping date strings to daily P&L

    Returns:
        Tuple of (max_drawdown_dollars, max_drawdown_percent)
    """
    if not daily_pnl:
        return 0.0, 0.0

    # Calculate cumulative P&L
    sorted_dates = sorted(daily_pnl.keys())
    cumulative_pnl = []
    running_total = 0.0

    for date in sorted_dates:
        running_total += daily_pnl[date]
        cumulative_pnl.append(running_total)

    if not cumulative_pnl:
        return 0.0, 0.0

    # Find maximum drawdown
    peak = cumulative_pnl[0]
    max_dd = 0.0

    for value in cumulative_pnl:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown

    # Calculate percentage drawdown
    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0

    return max_dd, max_dd_pct


def calculate_win_rate(trades: list[dict[str, Any]]) -> float:
    """Calculate win rate percentage.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Win rate as percentage (0-100)
    """
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t["pnl"] > 0)
    return (wins / len(trades)) * 100


def calculate_average_trade_duration(trades: list[dict[str, Any]]) -> float:
    """Calculate average trade duration in hours.

    Args:
        trades: List of trade dictionaries with 'entry_time' and 'exit_time'

    Returns:
        Average duration in hours
    """
    if not trades:
        return 0.0

    durations = []
    for trade in trades:
        try:
            entry = datetime.fromisoformat(trade["entry_time"])
            exit = datetime.fromisoformat(trade["exit_time"])
            duration = (exit - entry).total_seconds() / 3600  # Convert to hours
            durations.append(duration)
        except (KeyError, ValueError):
            continue

    return sum(durations) / len(durations) if durations else 0.0


def get_daily_returns_from_pnl(
    daily_pnl: dict[str, float], starting_capital: float = 100000.0
) -> list[float]:
    """Convert daily P&L to daily returns.

    Args:
        daily_pnl: Dictionary mapping date strings to daily P&L
        starting_capital: Starting account balance

    Returns:
        List of daily returns as decimals (e.g., 0.02 for 2%)
    """
    if not daily_pnl:
        return []

    sorted_dates = sorted(daily_pnl.keys())
    returns = []
    current_capital = starting_capital

    for date in sorted_dates:
        pnl = daily_pnl[date]
        daily_return = pnl / current_capital if current_capital > 0 else 0.0
        returns.append(daily_return)
        current_capital += pnl

    return returns


def calculate_portfolio_metrics(
    trades: list[dict[str, Any]],
    daily_pnl: dict[str, float],
    starting_capital: float = 100000.0,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Calculate comprehensive portfolio metrics.

    Args:
        trades: List of completed trades
        daily_pnl: Dictionary of daily P&L by date
        starting_capital: Starting account balance
        risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%)

    Returns:
        Dictionary of portfolio metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_trade_duration_hours": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    # Calculate daily returns
    daily_returns = get_daily_returns_from_pnl(daily_pnl, starting_capital)

    # Calculate max drawdown
    max_dd, max_dd_pct = calculate_max_drawdown(daily_pnl)

    return {
        "total_trades": len(trades),
        "total_pnl": sum(t["pnl"] for t in trades),
        "win_rate": calculate_win_rate(trades),
        "profit_factor": calculate_profit_factor(trades),
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(daily_returns, risk_free_rate),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "avg_trade_duration_hours": calculate_average_trade_duration(trades),
        "best_trade": max((t["pnl"] for t in trades), default=0.0),
        "worst_trade": min((t["pnl"] for t in trades), default=0.0),
    }


def calculate_instrument_metrics(
    trades: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate metrics per instrument.

    Args:
        trades: List of completed trades

    Returns:
        Dictionary mapping symbol to metrics dictionary
    """
    # Group trades by symbol
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        symbol = trade.get("symbol", "UNKNOWN")
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(trade)

    # Calculate metrics for each symbol
    results = {}
    for symbol, symbol_trades in by_symbol.items():
        results[symbol] = {
            "total_trades": len(symbol_trades),
            "total_pnl": sum(t["pnl"] for t in symbol_trades),
            "win_rate": calculate_win_rate(symbol_trades),
            "profit_factor": calculate_profit_factor(symbol_trades),
            "avg_trade_duration_hours": calculate_average_trade_duration(symbol_trades),
            "best_trade": max((t["pnl"] for t in symbol_trades), default=0.0),
            "worst_trade": min((t["pnl"] for t in symbol_trades), default=0.0),
        }

    return results
