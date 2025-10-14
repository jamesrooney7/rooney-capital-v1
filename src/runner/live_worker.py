"""Backtrader live runtime orchestrator.

This module wires together the Databento subscriber bridge with the
:class:`backtrader.Cerebro` runtime so production workers can hydrate the ML
bundles, attach live data feeds, and run :class:`~strategy.ibs_strategy.IbsStrategy`
continuously.  The implementation favours clarity and defensive guards so the
worker can be used both for documentation and for integration tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import backtrader as bt

from models import load_model_bundle, strategy_kwargs_from_bundle
from runner.databento_bridge import DatabentoLiveData, DatabentoSubscriber, QueueFanout
from runner.traderspost_client import (
    TradersPostClient,
    TradersPostError,
    order_notification_to_message,
    trade_notification_to_message,
)
from strategy.ibs_strategy import IbsStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "ContractMetadata",
    "InstrumentRuntimeConfig",
    "RuntimeConfig",
    "LiveWorker",
    "load_contract_metadata",
    "load_runtime_config",
]


# ---------------------------------------------------------------------------
# Strategy wrapper adding external notification callbacks
# ---------------------------------------------------------------------------


class NotifyingIbsStrategy(IbsStrategy):
    """`IbsStrategy` wrapper that forwards order/trade notifications."""

    def __init__(
        self,
        *args: Any,
        order_callbacks: Optional[list[Callable[["NotifyingIbsStrategy", Any], None]]] = None,
        trade_callbacks: Optional[
            list[Callable[["NotifyingIbsStrategy", Any, Optional[Mapping[str, Any]]], None]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        self._external_order_callbacks = list(order_callbacks or [])
        self._external_trade_callbacks = list(trade_callbacks or [])
        super().__init__(*args, **kwargs)

    def notify_order(self, order: Any) -> None:
        super().notify_order(order)
        if not self._external_order_callbacks:
            return
        for callback in list(self._external_order_callbacks):
            try:
                callback(self, order)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Order notification callback failed")

    def notify_trade(self, trade: Any) -> None:
        exit_snapshot: Optional[Mapping[str, Any]] = None
        if getattr(trade, "isclosed", False):
            pending_exit = getattr(self, "pending_exit", None)
            if isinstance(pending_exit, Mapping):
                exit_snapshot = dict(pending_exit)
        super().notify_trade(trade)
        if not self._external_trade_callbacks:
            return
        for callback in list(self._external_trade_callbacks):
            try:
                callback(self, trade, exit_snapshot)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Trade notification callback failed")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractMetadata:
    """Metadata describing how to subscribe to a Databento feed for a symbol."""

    symbol: str
    dataset: str
    feed_symbol: str
    product_id: Optional[str] = None

    @property
    def subscription_codes(self) -> tuple[str, ...]:
        """Return the subscription codes (product_id/feed_symbol) for Databento."""

        candidates: list[str] = []
        if self.product_id:
            candidates.append(self.product_id)
        if self.feed_symbol and self.feed_symbol != self.product_id:
            candidates.append(self.feed_symbol)
        if not candidates:
            candidates.append(self.symbol)
        # Preserve order while deduplicating
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate)
        return tuple(ordered)


@dataclass(frozen=True)
class InstrumentRuntimeConfig:
    """Runtime overrides controlling strategy sizing and commission."""

    symbol: str
    size: int = 1
    commission: float = 0.0
    margin: Optional[float] = None
    multiplier: Optional[float] = None
    strategy_overrides: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, symbol: str, payload: Mapping[str, Any]) -> "InstrumentRuntimeConfig":
        size = int(payload.get("size", 1) or 1)
        commission_raw = payload.get("commission", payload.get("commission_per_contract", 0.0))
        try:
            commission = float(commission_raw or 0.0)
        except (TypeError, ValueError):
            commission = 0.0
        margin_raw = payload.get("margin")
        multiplier_raw = payload.get("multiplier")
        try:
            margin = float(margin_raw) if margin_raw is not None else None
        except (TypeError, ValueError):
            margin = None
        try:
            multiplier = float(multiplier_raw) if multiplier_raw is not None else None
        except (TypeError, ValueError):
            multiplier = None
        overrides = dict(payload.get("strategy_overrides") or {})
        return cls(
            symbol=symbol,
            size=size,
            commission=commission,
            margin=margin,
            multiplier=multiplier,
            strategy_overrides=overrides,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level runtime configuration for the live worker."""

    databento_api_key: Optional[str]
    contract_map_path: Path
    models_path: Optional[Path]
    symbols: tuple[str, ...]
    starting_cash: float = 0.0
    backfill: bool = True
    queue_maxsize: int = 2048
    heartbeat_interval: Optional[int] = None
    poll_interval: float = 1.0
    traderspost_webhook: Optional[str] = None
    instruments: Mapping[str, InstrumentRuntimeConfig] = field(default_factory=dict)

    def instrument(self, symbol: str) -> InstrumentRuntimeConfig:
        cfg = self.instruments.get(symbol)
        if cfg:
            return cfg
        return InstrumentRuntimeConfig(symbol=symbol)


# ---------------------------------------------------------------------------
# Configuration loaders
# ---------------------------------------------------------------------------


def _load_json_or_yaml(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError(f"Unable to parse configuration file {path}: {exc}") from exc


def load_contract_metadata(path: str | Path) -> Mapping[str, ContractMetadata]:
    """Load Databento contract metadata from the deployment contract map."""

    contract_path = Path(path)
    payload = _load_json_or_yaml(contract_path)
    contracts: MutableMapping[str, ContractMetadata] = {}
    for entry in payload.get("contracts", []) or []:
        symbol = str(entry.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        databento = entry.get("databento") or {}
        dataset = str(databento.get("dataset") or "").strip()
        feed_symbol = str(databento.get("feed_symbol") or symbol).strip()
        product_id_raw = databento.get("product_id")
        product_id = str(product_id_raw).strip() if product_id_raw else None
        contracts[symbol] = ContractMetadata(
            symbol=symbol,
            dataset=dataset,
            feed_symbol=feed_symbol,
            product_id=product_id,
        )
    return contracts


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load worker runtime configuration from disk or environment."""

    config_path: Optional[Path]
    if path is None:
        env_path = os.environ.get("PINE_RUNTIME_CONFIG")
        config_path = Path(env_path).expanduser() if env_path else None
    else:
        config_path = Path(path)

    if not config_path:
        raise FileNotFoundError(
            "Runtime configuration path was not provided. Set PINE_RUNTIME_CONFIG or pass a path explicitly."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Runtime configuration file does not exist: {config_path}")

    payload = _load_json_or_yaml(config_path)

    contract_map_path = payload.get("contract_map") or payload.get("contract_map_path")
    if not contract_map_path:
        contract_map_path = Path("Data/Databento_contract_map.yml")
    contract_map_path = Path(contract_map_path).expanduser()

    models_path_raw = payload.get("models_path") or payload.get("models_dir")
    models_path = Path(models_path_raw).expanduser() if models_path_raw else None

    symbols_payload = payload.get("symbols") or payload.get("roots")
    if symbols_payload:
        symbols = tuple(str(sym).strip().upper() for sym in symbols_payload if sym)
    else:
        # Backfill from instrument configs if explicit symbols not provided.
        symbols = tuple(str(sym).strip().upper() for sym in (payload.get("contracts") or {}).keys())

    starting_cash = payload.get("starting_cash", payload.get("cash", 0.0))
    try:
        starting_cash_val = float(starting_cash or 0.0)
    except (TypeError, ValueError):
        starting_cash_val = 0.0

    backfill = bool(payload.get("backfill", True))
    queue_maxsize = int(payload.get("queue_maxsize", payload.get("queue_size", 2048)) or 2048)
    heartbeat_interval_raw = payload.get("heartbeat_interval")
    try:
        heartbeat_interval = int(heartbeat_interval_raw) if heartbeat_interval_raw is not None else None
    except (TypeError, ValueError):
        heartbeat_interval = None
    poll_interval_raw = payload.get("poll_interval", 1.0)
    try:
        poll_interval = float(poll_interval_raw)
    except (TypeError, ValueError):
        poll_interval = 1.0

    traderspost_payload = payload.get("traderspost") or {}
    webhook_url = (
        payload.get("traderspost_webhook")
        or traderspost_payload.get("webhook")
        or os.environ.get("TRADERSPOST_WEBHOOK_URL")
    )

    contracts_payload = payload.get("contracts") or {}
    instruments: Dict[str, InstrumentRuntimeConfig] = {}
    for symbol, cfg_payload in contracts_payload.items():
        sym = str(symbol).strip().upper()
        if not sym:
            continue
        instruments[sym] = InstrumentRuntimeConfig.from_mapping(sym, cfg_payload or {})

    return RuntimeConfig(
        databento_api_key=payload.get("databento_api_key") or os.environ.get("DATABENTO_API_KEY"),
        contract_map_path=contract_map_path,
        models_path=models_path,
        symbols=symbols,
        starting_cash=starting_cash_val,
        backfill=backfill,
        queue_maxsize=queue_maxsize,
        heartbeat_interval=heartbeat_interval,
        poll_interval=poll_interval,
        traderspost_webhook=webhook_url,
        instruments=instruments,
    )


# ---------------------------------------------------------------------------
# Live worker orchestration
# ---------------------------------------------------------------------------


class LiveWorker:
    """Coordinate Databento ingestion and Backtrader strategy execution."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.contracts = load_contract_metadata(config.contract_map_path)
        if not self.contracts:
            raise RuntimeError("No contracts loaded from contract map")

        # Limit to the configured symbol universe.
        if config.symbols:
            self.symbols = tuple(sym for sym in config.symbols if sym in self.contracts)
        else:
            self.symbols = tuple(sorted(self.contracts.keys()))
        if not self.symbols:
            raise RuntimeError("Runtime configuration did not reference any known symbols")

        product_to_root: Dict[str, str] = {}
        dataset_map: Dict[str, set[str]] = {}
        for symbol in self.symbols:
            meta = self.contracts[symbol]
            for code in meta.subscription_codes:
                product_to_root[code] = symbol
            dataset_map.setdefault(meta.dataset, set()).update(meta.subscription_codes)

        self.queue_manager = QueueFanout(product_to_root=product_to_root, maxsize=config.queue_maxsize)
        self.subscribers = [
            DatabentoSubscriber(
                dataset=dataset,
                product_codes=sorted(codes),
                queue_manager=self.queue_manager,
                api_key=config.databento_api_key,
                heartbeat_interval=config.heartbeat_interval,
            )
            for dataset, codes in dataset_map.items()
        ]

        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(config.starting_cash)
        self._cerebro_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop_callbacks: list[Callable[[], None]] = []

        self.traderspost_client: Optional[TradersPostClient]
        if config.traderspost_webhook:
            try:
                self.traderspost_client = TradersPostClient(config.traderspost_webhook)
                logger.info("TradersPost client initialised")
            except Exception:
                logger.exception("Failed to initialise TradersPost client")
                self.traderspost_client = None
        else:
            self.traderspost_client = None

        self._setup_data_and_strategies()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_data_and_strategies(self) -> None:
        for symbol in self.symbols:
            meta = self.contracts[symbol]
            instrument_cfg = self.config.instrument(symbol)

            data = DatabentoLiveData(symbol=symbol, queue_manager=self.queue_manager, backfill=self.config.backfill)
            self.cerebro.adddata(data, name=symbol)

            try:
                bundle = load_model_bundle(symbol, base_dir=self.config.models_path)
                bundle_kwargs = strategy_kwargs_from_bundle(bundle)
                logger.info("Loaded ML bundle for %s with features=%s", symbol, bundle.features)
            except FileNotFoundError:
                logger.warning("No ML bundle found for %s; running without ML filter", symbol)
                bundle_kwargs = {}
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to load ML bundle for %s: %s", symbol, exc)
                bundle_kwargs = {}

            strategy_kwargs: Dict[str, Any] = {"symbol": symbol, "size": instrument_cfg.size}
            strategy_kwargs.update(bundle_kwargs)
            strategy_kwargs.update(instrument_cfg.strategy_overrides)

            order_callbacks: list[Callable[[NotifyingIbsStrategy, Any], None]] = []
            trade_callbacks: list[
                Callable[[NotifyingIbsStrategy, Any, Optional[Mapping[str, Any]]], None]
            ] = []
            if self.traderspost_client:
                order_callbacks.append(self._traderspost_order_callback)
                trade_callbacks.append(self._traderspost_trade_callback)

            self.cerebro.addstrategy(
                NotifyingIbsStrategy,
                order_callbacks=order_callbacks,
                trade_callbacks=trade_callbacks,
                **strategy_kwargs,
            )

            commission_args: Dict[str, Any] = {"name": symbol, "commission": instrument_cfg.commission}
            if instrument_cfg.margin is not None:
                commission_args["margin"] = instrument_cfg.margin
            if instrument_cfg.multiplier is not None:
                commission_args["mult"] = instrument_cfg.multiplier
            try:
                self.cerebro.broker.setcommission(**commission_args)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Failed to apply commission for %s", symbol)

    def _traderspost_order_callback(self, strategy: NotifyingIbsStrategy, order: Any) -> None:
        if not self.traderspost_client:
            return
        payload = order_notification_to_message(strategy, order)
        if not payload:
            return
        try:
            self.traderspost_client.post_order(payload)
        except TradersPostError as exc:
            logger.error(
                "TradersPost order post failed for %s %s size=%s: %s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
                exc,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Unexpected TradersPost order post failure")
        else:
            logger.info(
                "Posted order event to TradersPost: %s %s size=%s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
            )

    def _traderspost_trade_callback(
        self,
        strategy: NotifyingIbsStrategy,
        trade: Any,
        exit_snapshot: Optional[Mapping[str, Any]],
    ) -> None:
        if not self.traderspost_client:
            return
        payload = trade_notification_to_message(strategy, trade, exit_snapshot)
        if not payload:
            return
        try:
            self.traderspost_client.post_trade(payload)
        except TradersPostError as exc:
            logger.error(
                "TradersPost trade post failed for %s %s size=%s: %s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
                exc,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Unexpected TradersPost trade post failure")
        else:
            logger.info(
                "Posted trade event to TradersPost: %s %s size=%s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
            )

    def _run_cerebro(self) -> None:
        logger.info("Starting Backtrader runtime for symbols: %s", ", ".join(self.symbols))
        try:
            self.cerebro.run(runonce=False, stdstats=False, maxcpus=1)
        except Exception as exc:  # pragma: no cover - unexpected runtime error
            logger.exception("Cerebro execution failed: %s", exc)
        finally:
            self._stop_event.set()
            logger.info("Cerebro runtime stopped")

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._cerebro_thread and self._cerebro_thread.is_alive():
            logger.debug("LiveWorker already running")
            return
        logger.info("Starting Databento subscribers")
        for subscriber in self.subscribers:
            subscriber.start()
        self._stop_event.clear()
        self._cerebro_thread = threading.Thread(target=self._run_cerebro, name="cerebro-runner", daemon=True)
        self._cerebro_thread.start()

    def stop(self) -> None:
        logger.info("Stopping LiveWorker")
        for subscriber in self.subscribers:
            subscriber.stop()
        self._stop_event.set()
        if self._cerebro_thread:
            self._cerebro_thread.join(timeout=10)
        logger.info("LiveWorker stopped")

    def request_stop(self) -> None:
        logger.info("Stop requested")
        self._stop_event.set()
        for callback in list(self._loop_callbacks):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Loop callback failed")

    # ------------------------------------------------------------------
    # Event loop helpers
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        """Run the worker under asyncio, handling signals and shutdown."""

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _handle_signal(sig: signal.Signals) -> None:
            logger.info("Received signal %s; shutting down", sig.name)
            stop_event.set()
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal, sig)
            except NotImplementedError:  # pragma: no cover - Windows
                signal.signal(sig, lambda *_: _handle_signal(sig))

        self.start()

        cerebro_future = loop.run_in_executor(None, self._wait_for_cerebro)

        try:
            while not self._stop_event.is_set():
                if stop_event.is_set():
                    break
                await asyncio.sleep(self.config.poll_interval)
        finally:
            self.stop()
            await cerebro_future

    def _wait_for_cerebro(self) -> None:
        if self._cerebro_thread:
            self._cerebro_thread.join()

    def run(self) -> None:
        """Synchronous helper that drives :meth:`run_async`."""

        asyncio.run(self.run_async())

    def add_loop_callback(self, callback: Callable[[], None]) -> None:
        self._loop_callbacks.append(callback)
