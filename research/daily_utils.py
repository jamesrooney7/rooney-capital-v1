#!/usr/bin/env python3
"""Daily utilities for time-series aggregation."""

import pandas as pd


def ensure_daily_index(series: pd.Series, dates: pd.Series) -> pd.Series:
    """
    Convert a series of trade-level returns to daily returns with DatetimeIndex.

    Args:
        series: Series of returns (trade-level)
        dates: Series of dates corresponding to trades (must have same index as series)

    Returns:
        Series with daily returns and DatetimeIndex
    """
    if series.empty:
        return pd.Series(dtype=float)

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
