"""Client helpers for posting execution events to TradersPost webhooks.

This module centralises the formatting and delivery of execution events so
`IbsStrategy` notifications can be pushed to TradersPost without replicating
payload shaping logic throughout the runner.  Payloads contain the strategy
root ticker symbol, action (buy/sell/exit), quantity, executed price, and a
``thresholds`` section which is populated with the ML score and filter snapshot
that triggered the order.

HTTP requests are retried with an exponential backoff when recoverable errors
occur (connection errors or ``5xx`` responses).  ``4xx`` responses are treated
as permanent failures so issues such as authentication errors are surfaced
immediately.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping, Optional

import backtrader as bt
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "TradersPostClient",
    "TradersPostError",
    "order_notification_to_message",
    "trade_notification_to_message",
]


class TradersPostError(RuntimeError):
    """Raised when TradersPost webhook delivery fails."""


class TradersPostClient:
    """Minimal HTTP client for posting execution events to TradersPost."""

    def __init__(
        self,
        webhook_url: Optional[str],
        *,
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: float = 10.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_factor <= 0:
            raise ValueError("backoff_factor must be > 0")

        self.webhook_url = webhook_url
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_order(self, payload: Mapping[str, Any]) -> requests.Response:
        """Post an order execution payload to the webhook."""

        return self._post({"event": "order", **dict(payload)})

    def post_trade(self, payload: Mapping[str, Any]) -> requests.Response:
        """Post a trade completion payload to the webhook."""

        return self._post({"event": "trade", **dict(payload)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_webhook_ready(self) -> None:
        if not self.webhook_url:
            raise TradersPostError("TradersPost webhook URL not configured")

    def _post(self, json_payload: Mapping[str, Any]) -> requests.Response:
        self._ensure_webhook_ready()
        attempt = 0
        backoff = self.backoff_factor
        last_error: Optional[BaseException] = None

        retry_after_hint: Optional[float] = None

        while attempt <= self.max_retries:
            try:
                response = self.session.post(
                    self.webhook_url,
                    json=json_payload,
                    timeout=self.timeout,
                )
                status = response.status_code
                if status >= 400:
                    if status == 429:
                        raise requests.HTTPError(response=response)
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        if 400 <= status < 500:
                            raise TradersPostError(
                                f"Webhook rejected payload with status {status}: {response.text}"
                            ) from exc
                        raise
                logger.debug(
                    "Posted TradersPost payload event=%s status=%s",
                    json_payload.get("event"),
                    response.status_code,
                )
                return response
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 429:
                    retry_after_hint = _retry_after_seconds(getattr(exc.response, "headers", {}))
                elif status is not None and 400 <= status < 500:
                    raise TradersPostError(
                        f"Webhook rejected payload with status {status}: {getattr(exc.response, 'text', '')}"
                    ) from exc
                last_error = exc
            except requests.RequestException as exc:
                retry_after_hint = None
                last_error = exc

            attempt += 1
            if attempt > self.max_retries:
                break

            sleep_for = retry_after_hint if retry_after_hint is not None else backoff
            backoff *= 2
            logger.warning(
                "TradersPost webhook post failed (attempt %s/%s); retrying in %.2fs",
                attempt,
                self.max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)
            retry_after_hint = None

        raise TradersPostError("Failed to deliver payload to TradersPost webhook") from last_error

# ---------------------------------------------------------------------------
# Notification payload helpers
# ---------------------------------------------------------------------------


def _safe_upper_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    sym = symbol.strip()
    return sym.upper() if sym else None


def _convert_contract_to_full_year(contract_symbol: str) -> str:
    """Convert Databento contract format (CLZ5) to TradersPost format (CLZ2025).

    Args:
        contract_symbol: Contract symbol from Databento like "CLZ5", "6NZ5", "ESH6"

    Returns:
        Contract symbol with full year like "CLZ2025", "6NZ2025", "ESH2026"
    """
    if not contract_symbol or len(contract_symbol) < 2:
        return contract_symbol

    # Extract the last character (year digit)
    year_digit = contract_symbol[-1]

    # Check if it's actually a year digit (0-9)
    if not year_digit.isdigit():
        return contract_symbol

    # Convert single digit to full year (assuming 2020s decade)
    # 0-9 maps to 2020-2029
    full_year = f"202{year_digit}"

    # Replace the single year digit with full year
    # e.g., "CLZ5" -> "CLZ2025", "ESH6" -> "ESH2026"
    return contract_symbol[:-1] + full_year


def _extract_symbol(strategy: Any, data: Any, queue_manager: Optional[Any] = None) -> str:
    """Extract the symbol from strategy/data, preferring contract-specific symbols.

    Args:
        strategy: The trading strategy
        data: The data feed
        queue_manager: Optional QueueFanout instance to look up contract symbols

    Returns:
        The symbol, preferring contract-specific symbols like "6NZ2025" over roots like "6N"
    """
    data_name = getattr(data, "_name", None)
    root_symbol = None
    if isinstance(data_name, str) and data_name:
        root_symbol = data_name.split("_")[0].upper()
    else:
        root_symbol = _safe_upper_symbol(getattr(getattr(strategy, "p", None), "symbol", None))

    if not root_symbol:
        return ""

    # Try to get the contract-specific symbol from queue_manager
    if queue_manager is not None and hasattr(queue_manager, "get_current_contract_symbol"):
        contract_symbol = queue_manager.get_current_contract_symbol(root_symbol)
        if contract_symbol:
            # Convert from Databento format (CLZ5) to TradersPost format (CLZ2025)
            full_year_contract = _convert_contract_to_full_year(contract_symbol.upper())
            logger.info("_extract_symbol: root=%s contract=%s full_year=%s",
                       root_symbol, contract_symbol, full_year_contract)
            return full_year_contract

    # Fallback to root symbol if contract lookup fails
    logger.info("_extract_symbol: using root symbol=%s (no contract found)", root_symbol)
    return root_symbol


def _ensure_ml_snapshot(snapshot: Mapping[str, Any] | None, strategy: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if snapshot:
        result.update(snapshot)
    ml_score = result.get("ml_score")
    if ml_score is None:
        ml_score = getattr(strategy, "_ml_last_score", None)
        if ml_score is not None:
            result["ml_score"] = ml_score
    return result


def _compact(mapping: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in mapping if value is not None}


def _retry_after_seconds(headers: Mapping[str, Any] | None) -> Optional[float]:
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            seconds = float(stripped)
        except ValueError:
            try:
                retry_dt = parsedate_to_datetime(stripped)
            except (TypeError, ValueError, IndexError):
                return None
            if retry_dt is None:
                return None
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            seconds = (retry_dt - datetime.now(timezone.utc)).total_seconds()
        if not math.isfinite(seconds):
            return None
        return max(0.0, seconds)
    return None
def _format_dt(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def order_notification_to_message(
    strategy: Any, order: Any, queue_manager: Optional[Any] = None
) -> Optional[dict[str, Any]]:
    """Translate an ``IbsStrategy.notify_order`` callback into a payload.

    Args:
        strategy: The trading strategy
        order: The order object
        queue_manager: Optional QueueFanout instance to look up contract symbols

    Returns:
        A dictionary payload for TradersPost, or None if the order is not ready
    """
    if getattr(order, "status", None) != bt.Order.Completed:
        return None

    executed = getattr(order, "executed", None)
    if executed is None:
        return None

    size = getattr(executed, "size", None)
    price = getattr(executed, "price", None)
    if size is None:
        return None

    info = getattr(order, "info", {}) or {}
    snapshot = info.get("filter_snapshot")
    thresholds = _ensure_ml_snapshot(snapshot if isinstance(snapshot, Mapping) else None, strategy)

    metadata = _compact(
        (
            ("created", _format_dt(info.get("created") or info.get("created_dt"))),
            ("ibs_value", info.get("ibs")),
            ("signal", info.get("exit_reason") or getattr(strategy, "current_signal", None)),
            ("order_ref", getattr(order, "ref", None)),
        )
    )

    payload = {
        "ticker": _extract_symbol(strategy, getattr(order, "data", None), queue_manager),
        "action": "buy" if getattr(order, "isbuy", lambda: False)() else "sell",
        "quantity": abs(size),  # TradersPost expects positive quantity, action determines direction
        "price": price,
        "timeInForce": "day",  # Active until market close, then auto-cancels
        "thresholds": thresholds,
        "metadata": metadata,
    }

    return payload


def trade_notification_to_message(
    strategy: Any,
    trade: Any,
    exit_snapshot: Optional[Mapping[str, Any]] = None,
    queue_manager: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Translate an ``IbsStrategy.notify_trade`` callback into a payload.

    Args:
        strategy: The trading strategy
        trade: The trade object
        exit_snapshot: Optional snapshot of exit conditions
        queue_manager: Optional QueueFanout instance to look up contract symbols

    Returns:
        A dictionary payload for TradersPost, or None if the trade is not ready
    """
    if not getattr(trade, "isclosed", False):
        return None

    snapshot = dict(exit_snapshot or {})
    thresholds_payload = _ensure_ml_snapshot(
        snapshot.get("filter_snapshot")
        if isinstance(snapshot.get("filter_snapshot"), Mapping)
        else None,
        strategy,
    )

    thresholds_payload.update(
        _compact(
            (
                ("ibs_value", snapshot.get("ibs_value")),
                ("sma200", snapshot.get("sma200")),
                ("tlt_sma20", snapshot.get("tlt_sma20")),
            )
        )
    )

    size_hint = snapshot.get("size")
    if size_hint is None:
        size_hint = getattr(trade, "size", None)

    metadata = _compact(
        (
            ("exit_reason", snapshot.get("exit_reason")),
            ("pnl", getattr(trade, "pnlcomm", getattr(trade, "pnl", None))),
            (
                "closed_at",
                _format_dt(
                    getattr(trade, "close_dt", None)
                    or getattr(trade, "dtclose", None)
                    or snapshot.get("dt")
                ),
            ),
        )
    )

    payload = {
        "ticker": _extract_symbol(strategy, getattr(trade, "data", None), queue_manager),
        "action": "exit",  # Trade close uses "exit" action per TradersPost API
        "quantity": abs(size_hint) if size_hint else None,
        "price": snapshot.get("price", getattr(trade, "price", None)),
        "thresholds": thresholds_payload,
        "metadata": metadata,
    }

    return payload
