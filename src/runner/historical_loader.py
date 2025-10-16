"""Utilities for loading historical Databento data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import databento as db

logger = logging.getLogger(__name__)


def load_historical_data(
    api_key: str,
    dataset: str,
    symbols: Iterable[str],
    days: int = 252,
) -> dict[str, Any]:
    """Load historical L1 data for indicator warmup."""

    client = db.Historical(api_key)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=max(days, 1))

    historical_data: dict[str, Any] = {}

    for symbol in symbols:
        logger.debug(
            "Requesting historical data for %s from %s to %s", symbol, start, end
        )
        data = client.timeseries.get_range(
            dataset=dataset,
            schema="mbp-1",  # L1 top-of-book
            symbols=[symbol],
            start=start,
            end=end,
            stype_in="parent",
        )
        historical_data[str(symbol)] = data

    logger.info(
        "Loaded historical data for %d symbols spanning %d days", len(historical_data), days
    )
    return historical_data
