"""Databento live data bridge utilities.

This module provides three building blocks used by the live runtime:

* :class:`DatabentoSubscriber` connects to the Databento live gateway,
  subscribes to the configured dataset/product pairs and aggregates tick
  trades into one-minute bars.
* :class:`QueueFanout` exposes per-symbol queues that the subscriber uses
  to fan out newly completed bars.  Backtrader data feeds read from those
  queues concurrently, so the fan-out container is thread safe and keeps
  track of instrument roll mappings learned from Databento metadata.
* :class:`DatabentoLiveData` adapts the queues into Backtrader's
  :class:`~backtrader.feeds.DataBase` interface and gracefully handles
  reconnects or temporary data gaps.

The implementation is intentionally defensive.  Network hiccups or API
schema changes shouldn't take the runner down; instead we log and retry
while continuing to service the queues with the data that is available.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import logging
import queue
import threading
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import backtrader as bt
from databento import Live, SymbolMappingMsg, TradeMsg

logger = logging.getLogger(__name__)


def _utcnow() -> dt.datetime:
    """Return a timezone-aware UTC timestamp."""

    return dt.datetime.now(dt.timezone.utc)


def _format_datetime(value: Optional[dt.datetime]) -> Optional[str]:
    """Render ``value`` as an ISO-8601 string if present."""

    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return (
        value.astimezone(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _as_float(value: Any) -> Optional[float]:
    """Attempt to coerce ``value`` into ``float`` returning ``None`` on failure."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclasses.dataclass(frozen=True)
class Bar:
    """Simple container for a completed bar."""

    symbol: str
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument_id: Optional[int] = None
    contract_symbol: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class QueueSignal:
    """A control message sent through a :class:`QueueFanout` queue."""

    kind: str
    symbol: Optional[str] = None

    RESET = "reset"
    SHUTDOWN = "shutdown"


class QueueFanout:
    """Manage per-symbol queues and roll mappings.

    Parameters
    ----------
    product_to_root:
        Mapping of Databento product codes (e.g. ``ESH4``) to the
        continuous root symbol (``ES``) used elsewhere in the runtime.
    maxsize:
        Maximum number of entries kept in each queue.  A bounded queue
        prevents runaway growth when a consumer stalls.
    """

    def __init__(self, product_to_root: Mapping[str, str], maxsize: int = 2048) -> None:
        self._product_to_root: Dict[str, str] = dict(product_to_root)
        self._instrument_to_product: Dict[int, str] = {}
        self._instrument_to_raw_symbol: Dict[int, str] = {}
        self._root_to_contract_symbol: Dict[str, str] = {}
        self._queues: Dict[str, "queue.Queue[Bar | QueueSignal]"] = {}
        self._maxsize = maxsize
        self._lock = threading.Lock()

        # Proactively create queues for all known roots so readers can grab
        # them before the first trade arrives.
        for root in sorted(set(self._product_to_root.values())):
            self._ensure_queue(root)

    # ------------------------------------------------------------------
    # Queue access helpers
    # ------------------------------------------------------------------
    def _ensure_queue(self, symbol: str) -> queue.Queue:
        with self._lock:
            if symbol not in self._queues:
                logger.debug("Creating queue for symbol %s", symbol)
                self._queues[symbol] = queue.Queue(self._maxsize)
            return self._queues[symbol]

    def get_queue(self, symbol: str) -> "queue.Queue[Bar | QueueSignal]":
        """Return the queue for ``symbol`` creating it on-demand."""

        return self._ensure_queue(symbol)

    def publish_bar(self, bar: Bar) -> None:
        """Push a completed bar into the per-symbol queue."""

        q = self._ensure_queue(bar.symbol)

        # Track the latest contract_symbol for this root
        if bar.contract_symbol:
            with self._lock:
                self._root_to_contract_symbol[bar.symbol] = bar.contract_symbol

        try:
            q.put_nowait(bar)
        except queue.Full:
            logger.warning(
                "Queue for symbol %s is full (%d/%d); dropping bar at %s",
                bar.symbol,
                q.qsize(),
                self._maxsize,
                bar.timestamp,
            )

    def publish_reset(self, symbol: str) -> None:
        """Notify consumers that an instrument has been reset."""

        q = self._ensure_queue(symbol)
        try:
            q.put_nowait(QueueSignal(QueueSignal.RESET, symbol))
        except queue.Full:
            logger.warning("Queue for symbol %s full while pushing reset", symbol)

    def broadcast_shutdown(self) -> None:
        """Signal every consumer that the producer is shutting down."""

        with self._lock:
            queues = list(self._queues.items())
        for symbol, q in queues:
            try:
                q.put_nowait(QueueSignal(QueueSignal.SHUTDOWN, symbol))
            except queue.Full:
                logger.debug("Queue for %s full when sending shutdown", symbol)

    # ------------------------------------------------------------------
    # Roll mapping helpers
    # ------------------------------------------------------------------
    def update_mapping(self, instrument_id: int, symbol: Optional[str]) -> None:
        """Record the mapping between ``instrument_id`` and ``symbol``.

        The Databento feed publishes ``SymbolMapping`` messages when a
        contract rolls to a new instrument id.  Those messages contain the
        symbology both in the inbound format (what we subscribed with) and
        in the outbound format (usually the continuous symbol).  We focus
        on the inbound value to resolve which root queue should receive
        the bar.
        """

        if not symbol:
            return

        root = self._product_to_root.get(symbol)
        if not root:
            # Fall back to stripping digits which mirrors how CME front
            # month codes are written, e.g. ESZ3 -> ES.
            root = "".join(ch for ch in symbol if not ch.isdigit())

        if not root:
            logger.debug(
                "Unable to resolve root for instrument %s (symbol %s)",
                instrument_id,
                symbol,
            )
            return

        self._ensure_queue(root)
        self._instrument_to_product[instrument_id] = symbol
        logger.info(
            "Mapped instrument %s (%s) to root %s", instrument_id, symbol, root
        )

    def resolve_root(self, instrument_id: int) -> Optional[str]:
        symbol = self._instrument_to_product.get(instrument_id)
        if not symbol:
            return None
        return self._product_to_root.get(symbol) or "".join(
            ch for ch in symbol if not ch.isdigit()
        )

    def update_raw_symbol(self, instrument_id: int, raw_symbol: Optional[str]) -> None:
        """Record the raw_symbol (contract-specific symbol) for an instrument_id.

        The raw_symbol contains the actual contract code like "6NZ2025" instead of
        the generic "6N.FUT" that comes from SymbolMappingMsg.

        Only accepts simple contract symbols, rejecting spreads and complex symbols.
        """
        if not raw_symbol:
            return

        # Filter out spread symbols and complex instruments
        # Valid contract symbols: CLZ5, 6NZ5, ESH6 (root + month + year)
        # Invalid: CL:C1 RB-CL Z5, CLZ5-CLH6, etc. (contain :, -, or spaces)
        if any(char in raw_symbol for char in [':', '-', ' ', '(', ')']):
            logger.debug("Rejecting complex symbol %s for instrument %s (likely a spread)",
                        raw_symbol, instrument_id)
            return

        with self._lock:
            self._instrument_to_raw_symbol[instrument_id] = raw_symbol
            logger.debug(
                "Recorded raw_symbol %s for instrument %s", raw_symbol, instrument_id
            )

    def resolve_contract_symbol(self, instrument_id: int) -> Optional[str]:
        """Resolve the specific contract symbol for an instrument_id.

        Returns the raw_symbol (e.g., "6NZ2025") if available, otherwise None.
        """
        with self._lock:
            return self._instrument_to_raw_symbol.get(instrument_id)

    def get_current_contract_symbol(self, root: str) -> Optional[str]:
        """Get the latest contract symbol for a root symbol.

        Returns the most recent contract symbol (e.g., "6NZ2025") for the given
        root (e.g., "6N"), or None if not available.
        """
        with self._lock:
            return self._root_to_contract_symbol.get(root.upper())

    def known_symbols(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._queues.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return queue depth and mapping telemetry for monitoring."""

        with self._lock:
            queue_depths = {symbol: q.qsize() for symbol, q in self._queues.items()}
            mapped = len(self._instrument_to_product)
        return {
            "known_symbols": sorted(queue_depths.keys()),
            "queue_depths": queue_depths,
            "mapped_instruments": mapped,
        }


class DatabentoSubscriber:
    """Subscribe to Databento live trades and build minute bars."""

    def __init__(
        self,
        dataset: str,
        product_codes: Sequence[str],
        queue_manager: QueueFanout,
        api_key: Optional[str] = None,
        schema: str = "trades",
        stype_in: str = "product_id",
        start: Optional[dt.datetime] = None,
        heartbeat_interval: Optional[int] = None,
        reconnect_backoff: Sequence[float] = (5.0, 10.0, 30.0),
    ) -> None:
        if not api_key:
            raise ValueError("Databento API key must be provided")

        self.dataset = dataset
        self.product_codes = tuple(product_codes)
        self.queue_manager = queue_manager
        self.api_key = api_key
        self.schema = schema
        self.stype_in = stype_in
        self.start_time = start
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_backoff = tuple(reconnect_backoff)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[Live] = None
        self._current_bars: Dict[str, MutableMapping[str, object]] = {}
        self._lock = threading.Lock()
        self._last_emitted_minute: Dict[str, dt.datetime] = {}
        self._last_close: Dict[str, float] = {}
        self._last_trade_ts: Dict[str, dt.datetime] = {}
        self._last_error: Optional[str] = None
        self._last_connect_time: Optional[dt.datetime] = None
        self._last_disconnect_time: Optional[dt.datetime] = None
        self._instrument_roots: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.debug("DatabentoSubscriber already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="databento-subscriber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        client = self._client
        if client is not None:
            try:
                client.stop()
            except Exception:  # pragma: no cover - best effort shutdown
                logger.debug("Error while stopping Databento client", exc_info=True)
        if self._thread:
            self._thread.join(timeout=5)
        self.flush()
        self.queue_manager.broadcast_shutdown()
        self._last_disconnect_time = _utcnow()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self) -> None:
        backoffs = collections.deque(self.reconnect_backoff)
        while not self._stop_event.is_set():
            try:
                self._last_error = None
                self._connect_and_stream()
            except Exception as exc:  # pragma: no cover - network/runtime errors
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Databento stream error: %s", exc)
            finally:
                self._last_disconnect_time = _utcnow()
                if self._stop_event.is_set():
                    break

            delay = backoffs[0] if backoffs else self.reconnect_backoff[-1]
            if len(backoffs) > 1:
                backoffs.rotate(-1)
            logger.info("Reconnecting to Databento after %.1fs", delay)
            time.sleep(delay)

    def _connect_and_stream(self) -> None:
        start_display = self.start_time.isoformat() if self.start_time else "default"
        logger.info(
            "Connecting to Databento dataset=%s schema=%s products=%s start=%s",
            self.dataset,
            self.schema,
            ",".join(self.product_codes),
            start_display,
        )
        client = Live(
            key=self.api_key,
            reconnect_policy="none",
            heartbeat_interval_s=self.heartbeat_interval,
        )
        self._client = client
        self._last_connect_time = _utcnow()

        client.subscribe(
            dataset=self.dataset,
            schema=self.schema,
            symbols=self.product_codes,
            stype_in=self.stype_in,
            start=self.start_time,
        )

        # Pull metadata (if present) to seed the instrument roll map.
        try:
            metadata = client.metadata
        except Exception:  # pragma: no cover - metadata access best effort
            metadata = None

        if metadata is not None:
            for mapping in getattr(metadata, "mappings", []) or []:
                instrument_id = getattr(mapping, "instrument_id", None)
                symbol = getattr(mapping, "symbol", None) or getattr(
                    mapping, "stype_in_symbol", None
                )
                raw_symbol = getattr(mapping, "raw_symbol", None)
                if instrument_id is None:
                    continue
                self.queue_manager.update_mapping(instrument_id, symbol)
                logger.info("Metadata mapping: instrument_id=%s symbol=%s raw_symbol=%s", instrument_id, symbol, raw_symbol)
                if raw_symbol:
                    self.queue_manager.update_raw_symbol(instrument_id, raw_symbol)
                    logger.info("Recorded raw_symbol %s for instrument %s", raw_symbol, instrument_id)
                root = self.queue_manager.resolve_root(instrument_id)
                if root:
                    self._instrument_roots[instrument_id] = root

            for entry in getattr(metadata, "symbols", []) or []:
                instrument_id = getattr(entry, "instrument_id", None)
                symbol = getattr(entry, "symbol", None) or getattr(
                    entry, "stype_in_symbol", None
                )
                raw_symbol = getattr(entry, "raw_symbol", None)
                if instrument_id is None:
                    continue
                self.queue_manager.update_mapping(instrument_id, symbol)
                logger.info("Metadata symbol: instrument_id=%s symbol=%s raw_symbol=%s", instrument_id, symbol, raw_symbol)
                if raw_symbol:
                    self.queue_manager.update_raw_symbol(instrument_id, raw_symbol)
                    logger.info("Recorded raw_symbol %s for instrument %s", raw_symbol, instrument_id)
                root = self.queue_manager.resolve_root(instrument_id)
                if root:
                    self._instrument_roots[instrument_id] = root

        sym_map = getattr(client, "symbology_map", None)
        if isinstance(sym_map, dict):
            for instrument_id, symbol in sym_map.items():
                try:
                    inst = int(instrument_id)
                except (TypeError, ValueError):
                    continue
                self.queue_manager.update_mapping(inst, str(symbol))
                root = self.queue_manager.resolve_root(inst)
                if root:
                    self._instrument_roots[inst] = root

        logger.info("Databento subscription ready; entering stream loop")

        try:
            for record in client:
                if self._stop_event.is_set():
                    break
                self._handle_record(record)
        finally:
            try:
                client.stop()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Error stopping Databento client", exc_info=True)
            self._client = None

        self.flush()
        logger.info("Databento stream closed")

    # ------------------------------------------------------------------
    # Record handling
    # ------------------------------------------------------------------
    def _handle_record(self, record) -> None:
        if isinstance(record, SymbolMappingMsg):
            instrument_id = getattr(record, "instrument_id", None)
            if instrument_id is None:
                return
            symbol = getattr(record, "stype_in_symbol", None) or getattr(
                record, "stype_out_symbol", None
            )
            raw_symbol = getattr(record, "raw_symbol", None)

            # Try to extract contract symbol from Databento client's internal mappings
            if self._client and hasattr(self._client, '_session'):
                try:
                    # The Databento client maintains symbology mappings internally
                    # Try to resolve the instrument_id to its contract symbol
                    client_symbol = self._client.symbology_map.get(instrument_id)
                    if client_symbol and not raw_symbol:
                        raw_symbol = client_symbol
                        logger.info("Extracted contract symbol %s from Databento client mapping for instrument %s",
                                   raw_symbol, instrument_id)
                except Exception as e:
                    logger.debug("Could not access Databento client symbology: %s", e)

            previous_root = self._instrument_roots.get(instrument_id)
            self.queue_manager.update_mapping(instrument_id, symbol)
            logger.info("SymbolMappingMsg: instrument_id=%s symbol=%s raw_symbol=%s", instrument_id, symbol, raw_symbol)
            if raw_symbol:
                self.queue_manager.update_raw_symbol(instrument_id, raw_symbol)
                logger.info("Recorded raw_symbol %s for instrument %s from SymbolMappingMsg", raw_symbol, instrument_id)
            root = self.queue_manager.resolve_root(instrument_id)
            if not root:
                self._instrument_roots.pop(instrument_id, None)
                return

            if previous_root is None:
                logger.debug(
                    "Recorded initial mapping for instrument %s to root %s",
                    instrument_id,
                    root,
                )
                self._instrument_roots[instrument_id] = root
                return

            if previous_root == root:
                logger.debug(
                    "Ignoring duplicate symbol mapping for instrument %s (root %s)",
                    instrument_id,
                    root,
                )
                self._instrument_roots[instrument_id] = root
                return

            self._instrument_roots[instrument_id] = root
            logger.info(
                "Received symbol mapping update for instrument %s; resetting state for %s",
                instrument_id,
                root,
            )
            with self._lock:
                self._current_bars.pop(root, None)
                self._last_emitted_minute.pop(root, None)
                self._last_close.pop(root, None)
                self._last_trade_ts.pop(root, None)
            self.queue_manager.publish_reset(root)
            return

        if isinstance(record, TradeMsg):
            root = self.queue_manager.resolve_root(record.instrument_id)
            if not root:
                # We haven't seen the mapping yet; skip until the next mapping
                # message arrives.
                logger.debug(
                    "Skipping trade for unknown instrument %s", record.instrument_id
                )
                return

            # Try to get contract symbol from Databento client's internal symbology
            if self._client:
                try:
                    client_symbol = self._client.symbology_map.get(record.instrument_id)
                    if client_symbol:
                        self.queue_manager.update_raw_symbol(record.instrument_id, client_symbol)
                        logger.info("Extracted contract symbol %s from client for instrument %s (root: %s)",
                                   client_symbol, record.instrument_id, root)
                except Exception as e:
                    logger.debug("Could not access client symbology for trade: %s", e)

            self._apply_trade(root, record, record.instrument_id)
            return

        # Other message types (e.g., heartbeats) are ignored but logged at
        # debug so we can inspect when necessary.
        logger.debug("Ignoring Databento record type %s", getattr(record, "rtype", "?"))

    def _apply_trade(self, root: str, trade: TradeMsg, instrument_id: int) -> None:
        price_raw = trade.pretty_price
        size = float(trade.size or 0)
        if price_raw is None:
            logger.debug("Trade without price for %s", root)
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Unparsable trade price %s for %s", price_raw, root)
            return

        # Log price details for debugging
        logger.debug("Trade price for %s: pretty_price=%s, price=%s, size=%s",
                    root, price_raw, price, size)

        ts = self._normalize_timestamp(trade.ts_event)
        minute = ts.replace(second=0, microsecond=0)

        with self._lock:
            self._last_trade_ts[root] = ts
            bar = self._current_bars.get(root)
            if not bar or bar["minute"] != minute:
                if bar:
                    emitted = self._emit_bar(root, bar)
                    if emitted:
                        self._fill_quiet_minutes(root, emitted.close, minute)
                bar = {
                    "minute": minute,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": size,
                    "last_ts": ts,
                    "instrument_id": instrument_id,
                }
                self._current_bars[root] = bar
            else:
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] = float(bar["volume"]) + size
                bar["last_ts"] = ts
                bar["instrument_id"] = instrument_id

    def flush(self) -> None:
        """Force emission of the current bars.

        This is typically called during shutdown to ensure that any
        partially built bars are delivered downstream before we tear down
        the subscriber.
        """

        with self._lock:
            bars = list(self._current_bars.items())
            self._current_bars.clear()

        for root, payload in bars:
            self._emit_bar(root, payload)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _emit_bar(self, root: str, payload: MutableMapping[str, object]) -> Optional[Bar]:
        instrument_id = payload.get("instrument_id")
        contract_symbol = None
        if isinstance(instrument_id, int):
            contract_symbol = self.queue_manager.resolve_contract_symbol(instrument_id)
            logger.info("Resolving contract for root=%s instrument_id=%s -> contract_symbol=%s", root, instrument_id, contract_symbol)

        bar = Bar(
            symbol=root,
            timestamp=payload["minute"],
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
            instrument_id=instrument_id if isinstance(instrument_id, int) else None,
            contract_symbol=contract_symbol,
        )
        logger.info("Emitting bar %s %s (contract: %s) O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
                   root, payload["minute"], contract_symbol or "unknown",
                   bar.open, bar.high, bar.low, bar.close, bar.volume)
        self.queue_manager.publish_bar(bar)
        self._last_emitted_minute[root] = bar.timestamp
        self._last_close[root] = bar.close
        return bar

    def _fill_quiet_minutes(
        self, root: str, last_close: float, next_minute: dt.datetime
    ) -> None:
        last_minute = self._last_emitted_minute.get(root)
        if last_minute is None:
            return
        tzinfo = last_minute.tzinfo
        if tzinfo is None and next_minute.tzinfo is not None:
            tzinfo = next_minute.tzinfo
        current = last_minute + dt.timedelta(minutes=1)
        if tzinfo is not None and current.tzinfo is None:
            current = current.replace(tzinfo=tzinfo)
        while current < next_minute:
            bar = Bar(
                symbol=root,
                timestamp=current,
                open=last_close,
                high=last_close,
                low=last_close,
                close=last_close,
                volume=0.0,
            )
            logger.debug("Backfilling quiet minute %s %s", root, current)
            self.queue_manager.publish_bar(bar)
            self._last_emitted_minute[root] = current
            self._last_close[root] = last_close
            current = current + dt.timedelta(minutes=1)
            if tzinfo is not None and current.tzinfo is None:
                current = current.replace(tzinfo=tzinfo)

    @staticmethod
    def _normalize_timestamp(ts_event: int) -> dt.datetime:
        # Databento timestamps are in nanoseconds.
        return dt.datetime.fromtimestamp(ts_event / 1_000_000_000, tz=dt.timezone.utc)

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------
    def status_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for heartbeat reporting."""

        thread_alive = bool(self._thread and self._thread.is_alive())
        client_connected = self._client is not None

        with self._lock:
            current_bars = {}
            for root, payload in self._current_bars.items():
                current_bars[root] = {
                    "minute": _format_datetime(payload.get("minute")),
                    "last_trade_ts": _format_datetime(payload.get("last_ts")),
                    "close": _as_float(payload.get("close")),
                    "volume": _as_float(payload.get("volume")),
                }
            last_emitted = {
                root: _format_datetime(ts) for root, ts in self._last_emitted_minute.items()
            }
            last_trade = {
                root: _format_datetime(ts) for root, ts in self._last_trade_ts.items()
            }

        return {
            "dataset": self.dataset,
            "product_codes": list(self.product_codes),
            "thread_alive": thread_alive,
            "client_connected": client_connected,
            "last_connect_time": _format_datetime(self._last_connect_time),
            "last_disconnect_time": _format_datetime(self._last_disconnect_time),
            "last_error": self._last_error,
            "last_emitted_minute": last_emitted,
            "building_bars": current_bars,
            "last_trade": last_trade,
        }


class DatabentoLiveData(bt.feeds.DataBase):
    """Backtrader live feed powered by :class:`QueueFanout` bars."""

    params = (
        ("symbol", None),
        ("queue_manager", None),
        ("backfill", True),
        ("qcheck", 0.5),
    )

    def __init__(self, **kwargs) -> None:
        timeframe = kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
        compression = kwargs.setdefault("compression", 1)
        super().__init__(**kwargs)
        try:
            self.p.timeframe = timeframe
            self.p.compression = compression
        except Exception:  # pragma: no cover - Params may be immutable in older bt
            pass
        self._timeframe = timeframe
        self._compression = compression
        if not self.p.symbol:
            raise ValueError("symbol parameter is required")
        if not self.p.queue_manager:
            raise ValueError("queue_manager parameter is required")

        self._queue: queue.Queue = self.p.queue_manager.get_queue(self.p.symbol)
        self._latest_dt: Optional[dt.datetime] = None
        self._stopped = False
        self._warmup_bars: "collections.deque[Bar]" = collections.deque()
        self._current_contract_symbol: Optional[str] = None  # Actual contract (e.g., "ESZ4")

    def start(self) -> None:
        super().start()
        logger.info("Starting DatabentoLiveData for %s", self.p.symbol)

    def stop(self) -> None:
        logger.info("Stopping DatabentoLiveData for %s", self.p.symbol)
        super().stop()
        self._stopped = True

    def _load(self) -> Optional[bool]:
        if self._stopped:
            return False

        if self._warmup_bars:
            # When draining warmup data, bypass the qcheck delay so Backtrader
            # can consume historical bars as fast as it can process them.  This
            # mirrors Cerebro's ``do_qcheck`` logic by manipulating the private
            # ``_qcheck`` attribute which controls how long the engine sleeps
            # between load attempts.
            self._qcheck = 0.0
            payload: Bar | QueueSignal = self._warmup_bars.popleft()
        else:
            if not self._qcheck:
                # Restore the configured qcheck once warmup bars are exhausted.
                self._qcheck = self.p.qcheck or 0.5

            timeout = self._qcheck or self.p.qcheck or 0.5
            try:
                if timeout <= 0:
                    payload = self._queue.get_nowait()
                else:
                    payload = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None

        if isinstance(payload, QueueSignal):
            if payload.kind == QueueSignal.SHUTDOWN:
                logger.info("Queue shutdown signalled for %s", self.p.symbol)
                self._stopped = True
                return False
            if payload.kind == QueueSignal.RESET:
                logger.info("Resetting feed state for %s", self.p.symbol)
                self._latest_dt = None
                return None
            return None

        if not isinstance(payload, Bar):  # pragma: no cover - defensive guard
            logger.debug("Unknown payload %s", type(payload))
            return None

        if self._latest_dt and payload.timestamp <= self._latest_dt:
            # Duplicate or out-of-order bar; ignore unless backfill is enabled.
            if not self.p.backfill:
                logger.debug(
                    "Skipping stale bar for %s @ %s", self.p.symbol, payload.timestamp
                )
                return None

        self.lines.datetime[0] = bt.date2num(payload.timestamp)
        self.lines.open[0] = payload.open
        self.lines.high[0] = payload.high
        self.lines.low[0] = payload.low
        self.lines.close[0] = payload.close
        self.lines.volume[0] = payload.volume

        self._latest_dt = payload.timestamp
        # Track the actual contract symbol from Databento (e.g., "ESZ4", "HGX2025")
        if payload.contract_symbol:
            self._current_contract_symbol = payload.contract_symbol
        return True

    # ------------------------------------------------------------------
    # Warmup helpers
    # ------------------------------------------------------------------
    def extend_warmup(self, bars: Iterable[Bar]) -> int:
        """Append historical ``bars`` to be consumed before live data."""

        appended = 0
        for bar in bars:
            if not isinstance(bar, Bar):  # pragma: no cover - defensive guard
                logger.debug("Ignoring warmup payload of type %s", type(bar))
                continue
            self._warmup_bars.append(bar)
            appended += 1
        if appended:
            logger.debug(
                "Buffered %d warmup bars for %s", appended, self.p.symbol
            )
        return appended

    def warmup_backlog_size(self) -> int:
        """Return the number of buffered warmup bars awaiting consumption."""

        return len(self._warmup_bars)


class ResampledLiveData(bt.feeds.DataBase):
    """Hybrid feed that aggregates minute bars into daily/hourly with warmup support.

    This feed operates in two modes:
    1. Warmup mode: Consumes pre-aggregated daily/hourly bars from warmup queue
    2. Live mode: Aggregates minute bars from source feed into daily/hourly bars

    This design allows us to:
    - Load 253 pre-aggregated daily bars during warmup (fast)
    - Seamlessly switch to aggregating live minute bars (correct)
    - Preserve full OHLC data for indicators
    """

    params = (
        ("symbol", None),
        ("source_feed", None),  # The minute-resolution feed to resample from
        ("session_end_hour", 23),  # Hour when day closes (UTC)
        ("session_end_minute", 0),
        ("bar_interval_minutes", 1440),  # 1440=daily, 60=hourly
        ("qcheck", 0.5),
    )

    def __init__(self, **kwargs) -> None:
        # Set timeframe based on interval
        interval = kwargs.get("bar_interval_minutes", 1440)
        if interval >= 1440:
            kwargs.setdefault("timeframe", bt.TimeFrame.Days)
            kwargs.setdefault("compression", 1)
        else:
            kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
            kwargs.setdefault("compression", interval)

        super().__init__(**kwargs)

        self._warmup_bars: "collections.deque[Bar]" = collections.deque()
        self._current_bar: Optional[dict[str, Any]] = None
        self._last_bar_timestamp: Optional[dt.datetime] = None
        self._stopped = False

    def start(self) -> None:
        super().start()
        logger.info("Starting ResampledLiveData for %s (%d-min bars)",
                    self.p.symbol, self.p.bar_interval_minutes)

    def stop(self) -> None:
        logger.info("Stopping ResampledLiveData for %s", self.p.symbol)
        super().stop()
        self._stopped = True

    def _load(self) -> Optional[bool]:
        """Load next bar (from warmup queue or by aggregating from source)."""
        if self._stopped:
            return False

        # Mode 1: Warmup - drain pre-aggregated bars from queue
        if self._warmup_bars:
            self._qcheck = 0.0  # Fast warmup
            bar = self._warmup_bars.popleft()
            return self._populate_lines_from_bar(bar)

        # Mode 2: Live - aggregate minute bars from source feed
        if not self._qcheck:
            self._qcheck = self.p.qcheck or 0.5  # Restore normal qcheck

        if self.p.source_feed is None:
            return None  # No source feed configured

        # Try to aggregate minute bars into our timeframe
        return self._aggregate_from_source()

    def _aggregate_from_source(self) -> Optional[bool]:
        """Aggregate minute bars from source feed into daily/hourly bars."""
        source = self.p.source_feed

        # Check if source has data available
        if len(source) == 0:
            return None

        # Get current minute bar from source
        minute_timestamp = bt.num2date(source.datetime[0])

        # Determine if this minute bar belongs to a new aggregation period
        if self._should_start_new_bar(minute_timestamp):
            # Close and emit the previous bar if it exists
            if self._current_bar is not None:
                result = self._populate_lines_from_dict(self._current_bar)
                self._current_bar = None
                self._last_bar_timestamp = minute_timestamp
                return result
            self._last_bar_timestamp = minute_timestamp

        # Aggregate this minute bar into current bar
        self._aggregate_minute_bar(
            timestamp=minute_timestamp,
            open_price=source.open[0],
            high_price=source.high[0],
            low_price=source.low[0],
            close_price=source.close[0],
            volume=source.volume[0]
        )

        # Check if we should close this bar (end of period)
        if self._is_bar_complete(minute_timestamp):
            result = self._populate_lines_from_dict(self._current_bar)
            self._current_bar = None
            return result

        return None  # Bar still building, not ready to emit

    def _should_start_new_bar(self, timestamp: dt.datetime) -> bool:
        """Check if this timestamp starts a new aggregation period."""
        if self._current_bar is None:
            return True  # No current bar, start fresh

        current_start = self._current_bar['timestamp']

        if self.p.bar_interval_minutes >= 1440:
            # Daily bars: new bar if different day
            return timestamp.date() != current_start.date()
        else:
            # Hourly bars: new bar if crossed hour boundary
            hours_diff = (timestamp - current_start).total_seconds() / 3600
            return hours_diff >= (self.p.bar_interval_minutes / 60)

    def _is_bar_complete(self, timestamp: dt.datetime) -> bool:
        """Check if current bar is complete and should be emitted.

        For hourly bars, we rely on _should_start_new_bar to emit when
        entering a new hour. This method is primarily for daily bars
        which need to emit at session end time.
        """
        if self.p.bar_interval_minutes >= 1440:
            # Daily bar: complete at session end
            return (timestamp.hour == self.p.session_end_hour and
                    timestamp.minute == self.p.session_end_minute)
        else:
            # Hourly bars: emit via _should_start_new_bar when hour changes
            # Don't emit here to avoid double-emission at :00 minute
            return False

    def _aggregate_minute_bar(
        self,
        timestamp: dt.datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float
    ) -> None:
        """Add minute bar data to current aggregated bar."""
        if self._current_bar is None:
            # Start new bar
            self._current_bar = {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
        else:
            # Update existing bar
            self._current_bar['high'] = max(self._current_bar['high'], high_price)
            self._current_bar['low'] = min(self._current_bar['low'], low_price)
            self._current_bar['close'] = close_price
            self._current_bar['volume'] += volume
            self._current_bar['timestamp'] = timestamp  # Use latest timestamp

    def _populate_lines_from_bar(self, bar: Bar) -> bool:
        """Populate feed lines from a Bar object."""
        self.lines.datetime[0] = bt.date2num(bar.timestamp)
        self.lines.open[0] = bar.open
        self.lines.high[0] = bar.high
        self.lines.low[0] = bar.low
        self.lines.close[0] = bar.close
        self.lines.volume[0] = bar.volume
        return True

    def _populate_lines_from_dict(self, bar_dict: dict) -> bool:
        """Populate feed lines from a dictionary."""
        self.lines.datetime[0] = bt.date2num(bar_dict['timestamp'])
        self.lines.open[0] = bar_dict['open']
        self.lines.high[0] = bar_dict['high']
        self.lines.low[0] = bar_dict['low']
        self.lines.close[0] = bar_dict['close']
        self.lines.volume[0] = bar_dict['volume']
        return True

    def extend_warmup(self, bars: Iterable[Bar]) -> int:
        """Add pre-aggregated bars for warmup (called before live trading)."""
        appended = 0
        for bar in bars:
            if not isinstance(bar, Bar):
                logger.debug("Ignoring warmup payload of type %s", type(bar))
                continue
            self._warmup_bars.append(bar)
            appended += 1
        if appended:
            logger.debug("Buffered %d warmup bars for %s", appended, self.p.symbol)
        return appended

    def warmup_backlog_size(self) -> int:
        """Return number of buffered warmup bars awaiting consumption."""
        return len(self._warmup_bars)


class DailyResampledLiveData(ResampledLiveData):
    """Daily resolution feed with warmup support and minute bar aggregation."""

    params = (
        ("symbol", None),
        ("source_feed", None),
        ("session_end_hour", 23),
        ("session_end_minute", 0),
        ("bar_interval_minutes", 1440),  # 1 day = 1440 minutes
        ("qcheck", 0.5),
    )


class HourlyResampledLiveData(ResampledLiveData):
    """Hourly resolution feed with warmup support and minute bar aggregation."""

    params = (
        ("symbol", None),
        ("source_feed", None),
        ("session_end_hour", 23),
        ("session_end_minute", 0),
        ("bar_interval_minutes", 60),  # 1 hour = 60 minutes
        ("qcheck", 0.5),
    )
