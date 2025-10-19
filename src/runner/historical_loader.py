"""Utilities for loading historical Databento data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

import databento as db

from runner.contract_map import ContractMap

logger = logging.getLogger(__name__)


def load_historical_data(
    api_key: str,
    dataset: str,
    symbols: Iterable[str],
    *,
    stype_in: str = "parent",
    days: int = 252,
    contract_map: Optional[ContractMap] = None,
    on_symbol_loaded: Optional[Callable[[str, Any], None]] = None,
) -> Optional[dict[str, Any]]:
    """Load historical L1 data for indicator warmup.

    When ``on_symbol_loaded`` is provided, symbols are streamed sequentially to
    the callback to keep peak memory usage low. Otherwise, the function returns
    a mapping of symbol to the retrieved historical data.
    """

    client = db.Historical(api_key)

    # Use a larger buffer to avoid requesting beyond the last available data,
    # especially over weekends when the latest data may be from Friday's close.
    end = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    start = end - timedelta(days=max(days, 1))

    historical_data: dict[str, Any] = {}
    processed_symbols = 0

    schema_preferences: tuple[str, ...] = (
        "ohlcv-1min",
        "ohlcv-1s",
        "mbp-1",
    )

    for symbol in symbols:
        logger.debug(
            "Requesting historical data for %s from %s to %s", symbol, start, end
        )
        request_symbols: list[str] = []
        if contract_map is not None:
            subscription = contract_map.subscription_for(symbol)
            if subscription and subscription.codes:
                request_symbols.extend(subscription.codes)
            elif stype_in == "product_id":
                try:
                    contract = contract_map.active_contract(symbol)
                except KeyError:
                    contract = None
                if contract and contract.databento.product_id:
                    request_symbols.append(contract.databento.product_id)

        if not request_symbols:
            if stype_in == "parent":
                request_symbols = [f"{symbol}.FUT"]
            else:
                request_symbols = [symbol]

        data = None
        errors: list[Exception] = []
        for schema in schema_preferences:
            try:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema=schema,
                    symbols=request_symbols,
                    start=start,
                    end=end,
                    stype_in=stype_in,
                )
                if data is not None:
                    if schema != "mbp-1":
                        logger.debug(
                            "Using lightweight schema %s for %s", schema, symbol
                        )
                    break
            except Exception as exc:  # pragma: no cover - depends on SDK errors
                errors.append(exc)
                logger.debug(
                    "Historical request for %s failed with schema %s: %s",
                    symbol,
                    schema,
                    exc,
                )
                data = None

        if data is None and errors:
            raise errors[-1]

        if on_symbol_loaded is not None:
            on_symbol_loaded(str(symbol), data)
            processed_symbols += 1
            del data
        else:
            historical_data[str(symbol)] = data

    if on_symbol_loaded is not None:
        logger.info(
            "Processed historical data for %d symbols spanning %d days",
            processed_symbols,
            days,
        )
        return None

    logger.info(
        "Loaded historical data for %d symbols spanning %d days", len(historical_data), days
    )
    return historical_data
