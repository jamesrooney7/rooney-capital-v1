"""Client helpers for posting execution events to TradersPost webhooks.

This module centralises the formatting and delivery of execution events so
`IbsStrategy` notifications can be pushed to TradersPost without replicating
payload shaping logic throughout the runner.  Payloads contain the strategy
root symbol, side, size, executed price, and a ``thresholds`` section which is
populated with the ML score and filter snapshot that triggered the order.

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
from decimal import Decimal, InvalidOperation
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
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
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
        self.api_base_url = api_base_url.rstrip("/") if api_base_url else None
        self.api_key = api_key

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

    # ------------------------------------------------------------------
    # REST API helpers
    # ------------------------------------------------------------------

    def _ensure_api_ready(self) -> None:
        if not self.api_base_url:
            raise TradersPostError("TradersPost API base URL not configured")

    def _api_headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if extra:
            headers.update(extra)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers

    def _api_request(
        self,
        method: str,
        path: str,
        *,
        allow_404: bool = False,
        **kwargs: Any,
    ) -> Optional[requests.Response]:
        self._ensure_api_ready()
        url = f"{self.api_base_url}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        if "json" in kwargs:
            headers.setdefault("Content-Type", "application/json")
        request_headers = self._api_headers(headers)
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                headers=request_headers,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise TradersPostError(
                f"TradersPost API request failed for {method} {url}: {exc}"
            ) from exc
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 400:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise TradersPostError(
                    f"TradersPost API request failed for {method} {url} with status "
                    f"{response.status_code}: {response.text}"
                ) from exc
        return response

    # ------------------------------------------------------------------
    # Portfolio management helpers
    # ------------------------------------------------------------------

    def get_open_positions(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the open position for *symbol* if one exists."""

        response = self._api_request("GET", f"positions/{symbol}", allow_404=True)
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise TradersPostError("Invalid JSON returned from TradersPost get_open_positions") from exc
        if not isinstance(payload, Mapping):
            return None
        size = payload.get("size")
        if _position_is_flat(size):
            return None
        return dict(payload)

    def get_pending_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Return pending orders for *symbol*."""

        response = self._api_request("GET", f"orders", params={"status": "pending", "symbol": symbol})
        try:
            payload = response.json() if response else []
        except ValueError as exc:
            raise TradersPostError("Invalid JSON returned from TradersPost get_pending_orders") from exc
        if not isinstance(payload, list):
            return []
        result: list[dict[str, Any]] = []
        for entry in payload:
            if isinstance(entry, Mapping):
                result.append(dict(entry))
        return result

    def close_position(self, symbol: str, reason: str = "Restart cleanup") -> bool:
        """Submit a market order to flatten the position for *symbol*."""

        payload = {"symbol": symbol, "reason": reason}
        response = self._api_request("POST", f"positions/{symbol}/close", json=payload)
        if not response:
            return False
        return response.status_code < 400

    def cancel_order(self, order_id: str) -> bool:
        """Cancel the pending order identified by *order_id*."""

        response = self._api_request("DELETE", f"orders/{order_id}", allow_404=True)
        if response is None:
            return False
        return response.status_code < 400


# ---------------------------------------------------------------------------
# Notification payload helpers
# ---------------------------------------------------------------------------


def _safe_upper_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    sym = symbol.strip()
    return sym.upper() if sym else None


def _extract_symbol(strategy: Any, data: Any) -> str:
    data_name = getattr(data, "_name", None)
    if isinstance(data_name, str) and data_name:
        return data_name.split("_")[0].upper()
    strat_symbol = _safe_upper_symbol(getattr(getattr(strategy, "p", None), "symbol", None))
    if strat_symbol:
        return strat_symbol
    return ""


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


def _position_is_flat(size: Any) -> bool:
    if size is None:
        return True
    candidate = size
    if isinstance(size, str):
        candidate = size.strip()
        if not candidate:
            return True
    try:
        return Decimal(str(candidate)) == 0
    except (InvalidOperation, ValueError, TypeError):
        return False


def _format_dt(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def order_notification_to_message(strategy: Any, order: Any) -> Optional[dict[str, Any]]:
    """Translate an ``IbsStrategy.notify_order`` callback into a payload."""

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
        "symbol": _extract_symbol(strategy, getattr(order, "data", None)),
        "side": "buy" if getattr(order, "isbuy", lambda: False)() else "sell",
        "size": size,
        "price": price,
        "thresholds": thresholds,
        "metadata": metadata,
    }

    return payload


def trade_notification_to_message(
    strategy: Any,
    trade: Any,
    exit_snapshot: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Translate an ``IbsStrategy.notify_trade`` callback into a payload."""

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
        "symbol": _extract_symbol(strategy, getattr(trade, "data", None)),
        "side": "sell" if (size_hint or 0) < 0 else "buy",
        "size": size_hint,
        "price": snapshot.get("price", getattr(trade, "price", None)),
        "thresholds": thresholds_payload,
        "metadata": metadata,
    }

    return payload
