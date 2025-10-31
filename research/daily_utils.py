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

    # Check if series already has a DatetimeIndex or datetime-like index (already aggregated)
    is_datetime_index = isinstance(series.index, pd.DatetimeIndex)
    is_datetime_like = False

    if not is_datetime_index and len(series.index) > 0:
        # Check if index contains datetime-like objects
        try:
            pd.to_datetime(series.index)
            is_datetime_like = True
        except (ValueError, TypeError):
            is_datetime_like = False

    if is_datetime_index or is_datetime_like:
        # Already daily-aggregated, just ensure proper format
        result = series.copy()
        result.index = pd.to_datetime(result.index)
        return result

    # Trade-level data - need to aggregate to daily
    if dates is None:
        raise ValueError("dates parameter required for trade-level data aggregation")

    # Ensure series and dates have matching indices
    # Use series' index to align dates
    try:
        aligned_dates = dates.loc[series.index]
    except KeyError:
        # If indices don't match, try reindexing dates to match series
        aligned_dates = dates.reindex(series.index)
        if aligned_dates.isna().all():
            raise ValueError("Cannot align dates with series index")

    # Create DataFrame with aligned dates and values
    df = pd.DataFrame({
        'date': pd.to_datetime(aligned_dates),
        'value': series.values
    }, index=series.index)

    # Group by date and sum returns
    daily = df.groupby('date')['value'].sum()
    daily.index = pd.to_datetime(daily.index)

    return daily
