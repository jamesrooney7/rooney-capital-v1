#!/usr/bin/env python3
"""Daily utilities for time-series aggregation."""

import pandas as pd


def ensure_daily_index(series: pd.Series, dates: pd.Series) -> pd.Series:
    """
    Convert a series of trade-level returns to daily returns with DatetimeIndex.

    Args:
        series: Series of returns (trade-level)
        dates: Series of dates corresponding to trades

    Returns:
        Series with daily returns and DatetimeIndex
    """
    if series.empty:
        return pd.Series(dtype=float)

    # Create DataFrame with dates and values
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'value': series.values
    })

    # Group by date and sum returns
    daily = df.groupby('date')['value'].sum()
    daily.index = pd.to_datetime(daily.index)

    return daily
