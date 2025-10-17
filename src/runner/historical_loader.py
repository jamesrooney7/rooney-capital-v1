"""Utilities for loading historical Databento data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import databento as db

from runner.contract_map import ContractMap

logger = logging.getLogger(__name__)


def load_historical_data(
    api_key: str,
    dataset: str,
    symbols: Iterable[str],
    days: int = 252,
    contract_map: Optional[ContractMap] = None,
) -> dict[str, Any]:
    """Load historical L1 data for indicator warmup."""

    client = db.Historical(api_key)

    end = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    start = end - timedelta(days=max(days, 1))

    historical_data: dict[str, Any] = {}

    for symbol in symbols:
        logger.debug(
            "Requesting historical data for %s from %s to %s", symbol, start, end
        )
        product_id: Optional[str] = None
        if contract_map is not None:
            try:
                contract = contract_map.active_contract(symbol)
            except KeyError:
                contract = None
            if contract is not None:
                product_id = contract.databento.product_id

        request_symbols = [product_id] if product_id else [f"{symbol}.FUT"]
        data = client.timeseries.get_range(
            dataset=dataset,
            schema="mbp-1",  # L1 top-of-book
            symbols=request_symbols,
            start=start,
            end=end,
            stype_in="parent",
        )
        historical_data[str(symbol)] = data

    logger.info(
        "Loaded historical data for %d symbols spanning %d days", len(historical_data), days
    )
    return historical_data
