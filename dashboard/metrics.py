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


def calculate_expectancy(trades: list[dict[str, Any]]) -> float:
    """Calculate expectancy (average profit per trade).

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Average profit/loss per trade in dollars
    """
    if not trades:
        return 0.0
    return sum(t["pnl"] for t in trades) / len(trades)


def calculate_calmar_ratio(
    total_pnl: float, max_drawdown: float, trading_days: int
) -> float:
    """Calculate Calmar Ratio (annualized return / max drawdown).

    Args:
        total_pnl: Total profit/loss
        max_drawdown: Maximum drawdown in dollars
        trading_days: Number of days with trades

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0 or trading_days == 0:
        return 0.0

    # Annualize the return (assume 252 trading days per year)
    annualized_return = (total_pnl / trading_days) * 252 if trading_days > 0 else 0
    return annualized_return / max_drawdown if max_drawdown > 0 else 0.0


def calculate_recovery_factor(total_pnl: float, max_drawdown: float) -> float:
    """Calculate Recovery Factor (net profit / max drawdown).

    Args:
        total_pnl: Total profit/loss
        max_drawdown: Maximum drawdown in dollars

    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return float('inf') if total_pnl > 0 else 0.0
    return total_pnl / max_drawdown


def calculate_consecutive_stats(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate consecutive win/loss streaks.

    Args:
        trades: List of trades sorted by exit time

    Returns:
        Dictionary with current_streak, max_win_streak, max_loss_streak
    """
    if not trades:
        return {"current_streak": 0, "max_win_streak": 0, "max_loss_streak": 0}

    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for trade in trades:
        if trade["pnl"] > 0:
            current_win_streak += 1
            current_loss_streak = 0
            current_streak = current_win_streak
            max_win_streak = max(max_win_streak, current_win_streak)
        elif trade["pnl"] < 0:
            current_loss_streak += 1
            current_win_streak = 0
            current_streak = -current_loss_streak
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            # Break even trade
            current_win_streak = 0
            current_loss_streak = 0

    return {
        "current_streak": current_streak,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def calculate_daily_stats(daily_pnl: dict[str, float]) -> dict[str, Any]:
    """Calculate daily-level statistics.

    Args:
        daily_pnl: Dictionary mapping date strings to daily P&L

    Returns:
        Dictionary with daily statistics
    """
    if not daily_pnl:
        return {
            "total_trading_days": 0,
            "profitable_days": 0,
            "profitable_days_pct": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "avg_day": 0.0,
        }

    pnls = list(daily_pnl.values())
    profitable_days = sum(1 for p in pnls if p > 0)

    return {
        "total_trading_days": len(pnls),
        "profitable_days": profitable_days,
        "profitable_days_pct": (profitable_days / len(pnls) * 100) if pnls else 0.0,
        "best_day": max(pnls) if pnls else 0.0,
        "worst_day": min(pnls) if pnls else 0.0,
        "avg_day": sum(pnls) / len(pnls) if pnls else 0.0,
    }


def calculate_win_loss_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate win/loss analysis.

    Args:
        trades: List of trades

    Returns:
        Dictionary with avg_win, avg_loss, win_loss_ratio
    """
    if not trades:
        return {"avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0}

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
    }


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
            "expectancy": 0.0,
            "calmar_ratio": 0.0,
            "recovery_factor": 0.0,
            "current_streak": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "total_trading_days": 0,
            "profitable_days_pct": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "trades_per_day": 0.0,
        }

    # Calculate daily returns
    daily_returns = get_daily_returns_from_pnl(daily_pnl, starting_capital)

    # Calculate max drawdown
    max_dd, max_dd_pct = calculate_max_drawdown(daily_pnl)

    # Calculate total P&L
    total_pnl = sum(t["pnl"] for t in trades)

    # Get additional stats
    win_loss = calculate_win_loss_stats(trades)
    consecutive = calculate_consecutive_stats(trades)
    daily_stats = calculate_daily_stats(daily_pnl)

    # Trades per day
    trades_per_day = len(trades) / daily_stats["total_trading_days"] if daily_stats["total_trading_days"] > 0 else 0.0

    return {
        # Basic metrics
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "win_rate": calculate_win_rate(trades),
        "profit_factor": calculate_profit_factor(trades),

        # Risk-adjusted metrics
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(daily_returns, risk_free_rate),
        "calmar_ratio": calculate_calmar_ratio(total_pnl, max_dd, daily_stats["total_trading_days"]),
        "recovery_factor": calculate_recovery_factor(total_pnl, max_dd),

        # Drawdown
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,

        # Per-trade metrics
        "avg_trade_duration_hours": calculate_average_trade_duration(trades),
        "best_trade": max((t["pnl"] for t in trades), default=0.0),
        "worst_trade": min((t["pnl"] for t in trades), default=0.0),
        "expectancy": calculate_expectancy(trades),

        # Win/Loss analysis
        "avg_win": win_loss["avg_win"],
        "avg_loss": win_loss["avg_loss"],
        "win_loss_ratio": win_loss["win_loss_ratio"],

        # Streaks
        "current_streak": consecutive["current_streak"],
        "max_win_streak": consecutive["max_win_streak"],
        "max_loss_streak": consecutive["max_loss_streak"],

        # Daily stats
        "total_trading_days": daily_stats["total_trading_days"],
        "profitable_days_pct": daily_stats["profitable_days_pct"],
        "best_day": daily_stats["best_day"],
        "worst_day": daily_stats["worst_day"],
        "trades_per_day": trades_per_day,
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
