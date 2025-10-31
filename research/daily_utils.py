#!/usr/bin/env python3
"""Daily utilities for time-series aggregation."""

import pandas as pd


def ensure_daily_index(series: pd.Series, dates: pd.Series = None) -> pd.Series:
    """
    Ensure series has a proper DatetimeIndex.

    If series is already daily-aggregated (index is dates), just ensure proper datetime format.
    If series is trade-level, aggregate to daily returns.

    Args:
        series: Series of returns (can be trade-level or already daily-aggregated)
        dates: Series of dates corresponding to trades (optional, only needed if series is trade-level)

    Returns:
        Series with daily returns and DatetimeIndex
    """
    if series.empty:
        return pd.Series(dtype=float)

    # Check if series already has a DatetimeIndex (already aggregated)
    if isinstance(series.index, pd.DatetimeIndex):
        # Already daily-aggregated, just ensure proper format
        result = series.copy()
        result.index = pd.to_datetime(result.index)
        return result

    # Trade-level data - need to aggregate to daily
    if dates is None:
        raise ValueError("dates parameter required for trade-level data aggregation")

    # Ensure series and dates have matching indices
    # Use series' index to align dates
    aligned_dates = dates.loc[series.index]

    # Create DataFrame with aligned dates and values
    df = pd.DataFrame({
        'date': pd.to_datetime(aligned_dates),
        'value': series.values
    }, index=series.index)

    # Group by date and sum returns
    daily = df.groupby('date')['value'].sum()
    daily.index = pd.to_datetime(daily.index)

    return daily
