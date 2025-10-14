"""Backtrader live runtime orchestrator.

This module wires together the Databento subscriber bridge with the
:class:`backtrader.Cerebro` runtime so production workers can hydrate the ML
bundles, attach live data feeds, and run :class:`~strategy.ibs_strategy.IbsStrategy`
continuously.  The implementation favours clarity and defensive guards so the
worker can be used both for documentation and for integration tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
import signal
import socket
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import backtrader as bt

import requests
from requests import exceptions as requests_exceptions

from config import PAIR_MAP, REQUIRED_REFERENCE_FEEDS
from models import load_model_bundle, strategy_kwargs_from_bundle
from runner.contract_map import ContractMap, ContractMapError, load_contract_map
from runner.databento_bridge import DatabentoLiveData, DatabentoSubscriber, QueueFanout
from runner.traderspost_client import (
    TradersPostClient,
    TradersPostError,
    order_notification_to_message,
    trade_notification_to_message,
)
from strategy.contract_specs import CONTRACT_SPECS
from strategy.ibs_strategy import IbsStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "InstrumentRuntimeConfig",
    "PreflightConfig",
    "RuntimeConfig",
    "LiveWorker",
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
class PreflightConfig:
    """Configuration flags controlling pre-flight validation."""

    enabled: bool = True
    skip_ml_validation: bool = False
    skip_connection_checks: bool = False
    fail_fast: bool = True


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
    heartbeat_file: Optional[Path] = None
    heartbeat_write_interval: float = 30.0
    poll_interval: float = 1.0
    traderspost_webhook: Optional[str] = None
    traderspost_api_base: Optional[str] = None
    traderspost_api_key: Optional[str] = None
    reconciliation_enabled: bool = True
    reconciliation_wait_seconds: float = 3.0
    reconciliation_max_retries: int = 2
    instruments: Mapping[str, InstrumentRuntimeConfig] = field(default_factory=dict)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)

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


def _coerce_bool(value: Any, default: bool) -> bool:
    """Return a best-effort boolean coercion for configuration flags."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1", "yes", "on"}:
            return True
        if normalised in {"false", "0", "no", "off"}:
            return False
        return default
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


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
    heartbeat_file_raw = (
        payload.get("heartbeat_file")
        or os.environ.get("PINE_HEARTBEAT_FILE")
    )
    heartbeat_file = Path(heartbeat_file_raw).expanduser() if heartbeat_file_raw else None
    heartbeat_write_interval_raw = (
        payload.get("heartbeat_write_interval")
        or os.environ.get("PINE_HEARTBEAT_WRITE_INTERVAL")
    )
    try:
        heartbeat_write_interval = (
            float(heartbeat_write_interval_raw)
            if heartbeat_write_interval_raw is not None
            else 30.0
        )
    except (TypeError, ValueError):
        heartbeat_write_interval = 30.0
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
    api_base = (
        traderspost_payload.get("api_base_url")
        or traderspost_payload.get("api_base")
        or os.environ.get("TRADERSPOST_API_BASE_URL")
    )
    api_key = traderspost_payload.get("api_key") or os.environ.get("TRADERSPOST_API_KEY")

    reconciliation_payload = payload.get("reconciliation") or {}
    reconciliation_enabled = bool(reconciliation_payload.get("enabled", True))
    wait_seconds_raw = reconciliation_payload.get("verification_wait_seconds", 3)
    try:
        reconciliation_wait_seconds = float(wait_seconds_raw)
    except (TypeError, ValueError):
        reconciliation_wait_seconds = 3.0
    max_retries_raw = reconciliation_payload.get("max_retries", 2)
    try:
        reconciliation_max_retries = int(max_retries_raw)
    except (TypeError, ValueError):
        reconciliation_max_retries = 2

    contracts_payload = payload.get("contracts") or {}
    instruments: Dict[str, InstrumentRuntimeConfig] = {}
    for symbol, cfg_payload in contracts_payload.items():
        sym = str(symbol).strip().upper()
        if not sym:
            continue
        instruments[sym] = InstrumentRuntimeConfig.from_mapping(sym, cfg_payload or {})

    preflight_payload = payload.get("preflight") or {}
    preflight_config = PreflightConfig(
        enabled=_coerce_bool(preflight_payload.get("enabled"), True),
        skip_ml_validation=_coerce_bool(preflight_payload.get("skip_ml_validation"), False),
        skip_connection_checks=_coerce_bool(
            preflight_payload.get("skip_connection_checks"), False
        ),
        fail_fast=_coerce_bool(preflight_payload.get("fail_fast"), True),
    )

    return RuntimeConfig(
        databento_api_key=payload.get("databento_api_key") or os.environ.get("DATABENTO_API_KEY"),
        contract_map_path=contract_map_path,
        models_path=models_path,
        symbols=symbols,
        starting_cash=starting_cash_val,
        backfill=backfill,
        queue_maxsize=queue_maxsize,
        heartbeat_interval=heartbeat_interval,
        heartbeat_file=heartbeat_file,
        heartbeat_write_interval=heartbeat_write_interval,
        poll_interval=poll_interval,
        traderspost_webhook=webhook_url,
        traderspost_api_base=api_base,
        traderspost_api_key=api_key,
        reconciliation_enabled=reconciliation_enabled,
        reconciliation_wait_seconds=reconciliation_wait_seconds,
        reconciliation_max_retries=reconciliation_max_retries,
        instruments=instruments,
        preflight=preflight_config,
    )


# ---------------------------------------------------------------------------
# Live worker orchestration
# ---------------------------------------------------------------------------


class LiveWorker:
    """Coordinate Databento ingestion and Backtrader strategy execution."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        try:
            self.contract_map: ContractMap = load_contract_map(config.contract_map_path)
        except ContractMapError as exc:
            raise RuntimeError(f"Failed to load contract metadata: {exc}") from exc

        if not self.contract_map.symbols():
            raise RuntimeError("No contracts loaded from contract map")

        # Limit to the configured symbol universe.
        if config.symbols:
            self.symbols = tuple(sym for sym in config.symbols if sym in self.contract_map)
        else:
            self.symbols = self.contract_map.symbols()
        if not self.symbols:
            raise RuntimeError("Runtime configuration did not reference any known symbols")

        pair_symbols = {
            sym
            for sym in set(PAIR_MAP.keys()) | set(PAIR_MAP.values())
            if sym in self.contract_map
        }
        contract_symbols = tuple(sorted(set(self.symbols) | pair_symbols))

        self.reference_symbols = self.contract_map.reference_symbols()

        product_to_root: Dict[str, str] = {}
        for code, symbol in self.contract_map.product_to_root(contract_symbols).items():
            product_to_root[code] = symbol
        for code, symbol in self.contract_map.reference_product_to_root(
            self.reference_symbols
        ).items():
            product_to_root.setdefault(code, symbol)

        dataset_groups = self.contract_map.dataset_groups(contract_symbols)
        for key, codes in self.contract_map.reference_dataset_groups(self.reference_symbols).items():
            existing = set(dataset_groups.get(key, ()))
            existing.update(codes)
            dataset_groups[key] = tuple(sorted(existing))

        self.data_symbols = tuple(sorted(set(contract_symbols) | set(self.reference_symbols)))

        self.queue_manager = QueueFanout(product_to_root=product_to_root, maxsize=config.queue_maxsize)
        self._data_feeds: Dict[str, bt.feeds.DataBase] = {}
        self.subscribers = []
        for (dataset, stype_in), codes in dataset_groups.items():
            self.subscribers.append(
                DatabentoSubscriber(
                    dataset=dataset,
                    product_codes=sorted(codes),
                    queue_manager=self.queue_manager,
                    api_key=config.databento_api_key,
                    heartbeat_interval=config.heartbeat_interval,
                    stype_in=stype_in,
                )
            )

        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(config.starting_cash)
        self._cerebro_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop_callbacks: list[Callable[[], None]] = []
        self._last_heartbeat_update: float = 0.0

        self.traderspost_client: Optional[TradersPostClient]
        if config.traderspost_webhook or config.traderspost_api_base:
            try:
                self.traderspost_client = TradersPostClient(
                    config.traderspost_webhook,
                    api_base_url=config.traderspost_api_base,
                    api_key=config.traderspost_api_key,
                )
                logger.info("TradersPost client initialised")
            except Exception:
                logger.exception("Failed to initialise TradersPost client")
                self.traderspost_client = None
        else:
            self.traderspost_client = None

        self._setup_data_and_strategies()

        if self.config.heartbeat_file:
            logger.info("Heartbeat file configured at %s", self.config.heartbeat_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_data_and_strategies(self) -> None:
        for symbol in self.data_symbols:
            if symbol in self._data_feeds:
                continue
            data = DatabentoLiveData(
                symbol=symbol,
                queue_manager=self.queue_manager,
                backfill=self.config.backfill,
            )
            self.cerebro.adddata(data, name=symbol)
            self._data_feeds[symbol] = data
            self.cerebro.resampledata(
                data,
                timeframe=bt.TimeFrame.Minutes,
                compression=60,
                name=f"{symbol}_hour",
            )
            self.cerebro.resampledata(
                data,
                timeframe=bt.TimeFrame.Days,
                compression=1,
                name=f"{symbol}_day",
            )

        for symbol in self.symbols:
            instrument_cfg = self.config.instrument(symbol)

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
        metadata = self._metadata_for_symbol(payload.get("symbol"))
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
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
        metadata = self._metadata_for_symbol(payload.get("symbol"))
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
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

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def reconcile_on_startup(self) -> None:
        """Flatten existing positions and cancel pending orders before trading."""

        if not self.config.reconciliation_enabled:
            logger.info("Position reconciliation skipped (disabled)")
            return
        if not self.traderspost_client:
            logger.info("Position reconciliation skipped - TradersPost client unavailable")
            return

        symbols = tuple(self.symbols)
        logger.info("Starting position reconciliation for %s symbols", len(symbols))

        wait_seconds = max(float(self.config.reconciliation_wait_seconds), 0.0)
        max_retries = max(int(self.config.reconciliation_max_retries), 0)
        errors_detected = False

        for symbol in symbols:
            try:
                position = self.traderspost_client.get_open_positions(symbol)
            except Exception:
                errors_detected = True
                logger.exception("Failed to query open position for %s", symbol)
                position = None

            last_check: Optional[Mapping[str, Any]] = position if isinstance(position, Mapping) else None

            if position:
                size = position.get("size") if isinstance(position, Mapping) else None
                entry_price = position.get("entry_price") if isinstance(position, Mapping) else None
                logger.warning("⚠️ Found existing position: %s size=%s @ %s", symbol, size, entry_price)

                closed = False
                attempts = max_retries + 1
                for _ in range(max(1, attempts)):
                    try:
                        logger.info("Closing position: %s", symbol)
                        success = self.traderspost_client.close_position(symbol)
                        if not success:
                            errors_detected = True
                    except Exception:
                        errors_detected = True
                        logger.exception("Failed to submit close order for %s", symbol)
                    if wait_seconds:
                        time.sleep(wait_seconds)
                    try:
                        check_position = self.traderspost_client.get_open_positions(symbol)
                    except Exception:
                        errors_detected = True
                        logger.exception("Failed to verify position close for %s", symbol)
                        check_position = None
                    if not check_position:
                        logger.info("✓ Position closed: %s", symbol)
                        closed = True
                        last_check = None
                        break
                    last_check = check_position if isinstance(check_position, Mapping) else None
                if not closed:
                    remaining = last_check.get("size") if isinstance(last_check, Mapping) else None
                    logger.error("❌ Failed to close position: %s - still shows size=%s", symbol, remaining)
                    errors_detected = True

            try:
                pending_orders = self.traderspost_client.get_pending_orders(symbol)
            except Exception:
                errors_detected = True
                logger.exception("Failed to query pending orders for %s", symbol)
                pending_orders = []

            for order in pending_orders:
                order_id_raw = None
                if isinstance(order, Mapping):
                    order_id_raw = order.get("id") or order.get("order_id")
                order_id = str(order_id_raw).strip() if order_id_raw is not None else ""
                if not order_id:
                    continue
                logger.info("Found pending order: %s for %s", order_id, symbol)
                logger.info("Canceling order: %s", order_id)
                try:
                    cancelled = self.traderspost_client.cancel_order(order_id)
                except Exception:
                    errors_detected = True
                    logger.exception("Failed to cancel order %s", order_id)
                    continue
                if cancelled:
                    logger.info("✓ Order canceled: %s", order_id)
                else:
                    logger.error("❌ Failed to cancel order: %s", order_id)
                    errors_detected = True

            try:
                final_position = self.traderspost_client.get_open_positions(symbol)
            except Exception:
                errors_detected = True
                logger.exception("Failed to perform final reconciliation check for %s", symbol)
                continue

            if final_position:
                errors_detected = True
                size = final_position.get("size") if isinstance(final_position, Mapping) else None
                entry_price = final_position.get("entry_price") if isinstance(final_position, Mapping) else None
                logger.warning("⚠️ Found existing position: %s size=%s @ %s", symbol, size, entry_price)

        if errors_detected:
            logger.warning("⚠️ Reconciliation completed with errors - manual verification recommended")
        else:
            logger.info("Position reconciliation complete - all symbols flat")

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
        self._update_heartbeat(status="running", force=True)

    def stop(self) -> None:
        logger.info("Stopping LiveWorker")
        for subscriber in self.subscribers:
            subscriber.stop()
        self._stop_event.set()
        if self._cerebro_thread:
            self._cerebro_thread.join(timeout=10)
        logger.info("LiveWorker stopped")
        self._update_heartbeat(status="stopped", force=True)

    def request_stop(self) -> None:
        logger.info("Stop requested")
        self._stop_event.set()
        for callback in list(self._loop_callbacks):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Loop callback failed")
        self._update_heartbeat(status="stopping", force=True)

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

        if not self.run_preflight_checks():
            logger.critical("❌ STARTUP ABORTED: Pre-flight checks failed")
            self._update_heartbeat(status="failed", force=True, details={"stage": "preflight"})
            raise RuntimeError("Pre-flight validation failed")

        try:
            self.reconcile_on_startup()
        except Exception:
            logger.exception("Unexpected error during startup reconciliation")
            self._update_heartbeat(status="failed", force=True, details={"stage": "reconciliation"})
            raise

        self._update_heartbeat(status="starting", force=True)
        self.start()

        cerebro_future = loop.run_in_executor(None, self._wait_for_cerebro)

        try:
            while not self._stop_event.is_set():
                if stop_event.is_set():
                    break
                self._update_heartbeat(status="running")
                await asyncio.sleep(self.config.poll_interval)
        finally:
            self._update_heartbeat(status="stopping", force=True)
            self.stop()
            await cerebro_future

    def _wait_for_cerebro(self) -> None:
        if self._cerebro_thread:
            self._cerebro_thread.join()

    def _metadata_for_symbol(self, symbol: Any) -> dict[str, Any]:
        if not isinstance(symbol, str):
            return {}
        sym = symbol.strip().upper()
        if not sym or sym not in self.contract_map:
            return {}
        return self.contract_map.traderspost_metadata(sym)

    def _update_heartbeat(
        self,
        status: str,
        *,
        force: bool = False,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        path = self.config.heartbeat_file
        if path is None:
            return

        interval = max(float(self.config.heartbeat_write_interval), 0.0)
        now = time.monotonic()
        if not force and interval and now - self._last_heartbeat_update < interval:
            return

        payload: dict[str, Any] = {
            "status": str(status),
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "symbols": list(self.symbols),
        }
        if details:
            try:
                payload["details"] = dict(details)
            except Exception:  # pragma: no cover - defensive guard
                payload["details"] = {"message": str(details)}

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
            self._last_heartbeat_update = now
        except Exception:  # pragma: no cover - best effort monitoring
            logger.exception("Failed to write heartbeat file at %s", path)

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------

    def validate_ml_models(self) -> bool:
        total = len(self.symbols)
        logger.info("Validating ML models for %s symbols...", total)
        if not total:
            logger.info("✓ ML models validated for %s symbols", total)
            return True

        failed = False
        for symbol in self.symbols:
            try:
                bundle = load_model_bundle(symbol, base_dir=self.config.models_path)
            except Exception as exc:
                logger.error("❌ %s: Failed to load model bundle - %s", symbol, exc)
                failed = True
                continue

            try:
                bundle_kwargs = strategy_kwargs_from_bundle(bundle)
            except Exception as exc:
                logger.error("❌ %s: Failed to prepare model bundle - %s", symbol, exc)
                failed = True
                continue

            model = bundle_kwargs.get("ml_model")
            ml_features = bundle_kwargs.get("ml_features")
            ml_threshold = bundle_kwargs.get("ml_threshold")

            if not hasattr(model, "predict_proba"):
                logger.error("❌ %s: Model missing predict_proba method", symbol)
                failed = True
                continue

            if not ml_features or len(ml_features) == 0:
                logger.error("❌ %s: Model has no features defined", symbol)
                failed = True
                continue

            if ml_threshold is None:
                logger.error("❌ %s: Model has no threshold defined", symbol)
                failed = True
                continue

            dummy_vector = [0.0] * len(ml_features)
            try:
                model.predict_proba([dummy_vector])
            except Exception as exc:
                logger.error("❌ %s: Model failed test prediction - %s", symbol, exc)
                failed = True
                continue

            logger.info(
                "✓ %s: Model loaded, %s features, threshold=%s",
                symbol,
                len(ml_features),
                ml_threshold,
            )

        if failed:
            return False

        logger.info("✓ ML models validated for %s symbols", total)
        return True

    def validate_traderspost_connection(self) -> bool:
        logger.info("Validating TradersPost connection...")
        url = (self.config.traderspost_webhook or "").strip()
        if not url:
            logger.error("❌ TradersPost webhook URL not configured")
            return False

        payload = {"type": "health_check", "timestamp": datetime.utcnow().isoformat()}
        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests_exceptions.Timeout:
            logger.error("❌ TradersPost connection timeout after 10s")
            return False
        except requests_exceptions.ConnectionError:
            logger.error("❌ TradersPost connection refused - check URL: %s", url)
            return False
        except requests_exceptions.RequestException as exc:
            logger.error("❌ TradersPost validation failed: %s", exc)
            return False

        status = response.status_code
        text = (response.text or "").strip()
        if 200 <= status < 300:
            logger.info("✓ TradersPost webhook reachable at %s", url)
            return True

        error_text = text or "(no response body)"
        if 400 <= status < 500:
            logger.error("❌ TradersPost returned error: %s - %s", status, error_text)
        elif 500 <= status < 600:
            logger.error("❌ TradersPost returned error: %s - %s", status, error_text)
        else:
            logger.error("❌ TradersPost validation failed: HTTP %s - %s", status, error_text)
        return False

    def validate_databento_connection(self) -> bool:
        logger.info("Validating Databento connection...")
        api_key = (self.config.databento_api_key or "").strip()
        if not api_key:
            logger.error("❌ Databento API key not configured")
            return False

        try:
            import databento
        except Exception as exc:
            logger.error("❌ Databento validation failed: %s", exc)
            return False

        try:
            client = databento.Live(key=api_key)
        except Exception as exc:
            logger.error("❌ Databento API key invalid or expired")
            logger.debug("Databento Live initialisation failed", exc_info=exc)
            return False

        metadata_client = getattr(client, "metadata", None)
        if metadata_client is None:
            logger.error("❌ Databento validation failed: metadata interface unavailable")
            return False

        probe_called = False
        try:
            if hasattr(metadata_client, "list_datasets"):
                probe_called = True
                try:
                    metadata_client.list_datasets(timeout=10)
                except TypeError:
                    metadata_client.list_datasets()
            elif hasattr(metadata_client, "list_schemas"):
                probe_called = True
                try:
                    metadata_client.list_schemas(timeout=10)
                except TypeError:
                    metadata_client.list_schemas()
            else:
                raise RuntimeError("metadata probe unavailable")
        except (requests_exceptions.Timeout, TimeoutError, socket.timeout):
            logger.error("❌ Databento connection timeout after 10s")
            return False
        except Exception as exc:
            message = str(exc).lower()
            if "unauthorized" in message or "invalid" in message or "forbidden" in message:
                logger.error("❌ Databento API key invalid or expired")
            else:
                logger.error("❌ Databento validation failed: %s", exc)
            return False

        if not probe_called:
            logger.error("❌ Databento validation failed: metadata probe unavailable")
            return False

        logger.info("✓ Databento API connection verified")
        return True

    def validate_reference_data(self) -> bool:
        logger.info("Validating reference data...")

        try:
            reference_symbols = set(self.contract_map.reference_symbols())
        except Exception as exc:
            logger.error("❌ Failed to load contract map: %s", exc)
            return False

        failed = False

        for feed_name in REQUIRED_REFERENCE_FEEDS:
            feed = str(feed_name or "").strip()
            if not feed:
                continue
            base = feed
            if base.endswith("_day"):
                base = base[: -len("_day")]
            elif base.endswith("_hour"):
                base = base[: -len("_hour")]
            if base.upper() not in reference_symbols:
                logger.error("❌ Required reference feed missing: %s", feed)
                failed = True

        for symbol in self.symbols:
            if symbol not in CONTRACT_SPECS:
                logger.error("❌ Contract specs missing for: %s", symbol)
                failed = True

        for symbol in self.symbols:
            pair_symbol = PAIR_MAP.get(symbol)
            if not pair_symbol:
                logger.warning("⚠️ Pair symbol not configured for: %s (pair trading disabled)", symbol)
                continue
            if pair_symbol not in self.symbols:
                logger.warning(
                    "⚠️ Pair symbol not configured for: %s (pair trading disabled)", symbol
                )

        if failed:
            return False

        logger.info("✓ Reference data validated for %s symbols", len(self.symbols))
        return True

    def validate_data_feeds(self) -> bool:
        logger.info("Validating data feed configuration...")

        available_names = {
            getattr(data, "_name", None)
            for data in getattr(self.cerebro, "datas", [])
        }
        available_names = {
            name for name in available_names if isinstance(name, str) and name
        }

        failed = False

        for symbol in self.symbols:
            hour_name = f"{symbol}_hour"
            day_name = f"{symbol}_day"
            if hour_name not in available_names:
                logger.error("❌ Hourly feed not configured: %s", hour_name)
                failed = True
            if day_name not in available_names:
                logger.error("❌ Daily feed not configured: %s", day_name)
                failed = True

        for feed_name in REQUIRED_REFERENCE_FEEDS:
            feed = str(feed_name or "").strip()
            if not feed:
                continue
            if feed not in available_names:
                logger.error("❌ Reference feed not configured: %s", feed)
                failed = True

        if failed:
            return False

        logger.info("✓ Data feeds validated for %s symbols", len(self.symbols))
        return True

    def run_preflight_checks(self) -> bool:
        separator = "=" * 60
        logger.info(separator)
        logger.info("STARTING PRE-FLIGHT VALIDATION")
        logger.info(separator)

        preflight_cfg = getattr(self.config, "preflight", PreflightConfig())

        if not preflight_cfg.enabled:
            logger.info("")
            logger.info("Pre-flight validation disabled via configuration; skipping checks.")
            logger.info("")
            logger.info(separator)
            logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
            logger.info(separator)
            return True

        failed_checks: list[str] = []
        checks: list[tuple[str, Callable[[], bool]]] = []

        if preflight_cfg.skip_ml_validation:
            logger.info("")
            logger.info("Skipping ML model validation (disabled via configuration)")
        else:
            checks.append(("ML Models", self.validate_ml_models))

        if preflight_cfg.skip_connection_checks:
            logger.info("")
            logger.info("Skipping connectivity checks (disabled via configuration)")
        else:
            checks.append(("TradersPost Connection", self.validate_traderspost_connection))
            checks.append(("Databento Connection", self.validate_databento_connection))

        checks.append(("Reference Data", self.validate_reference_data))
        checks.append(("Data Feeds", self.validate_data_feeds))

        for name, check in checks:
            logger.info("")
            result = False
            try:
                result = check()
            except Exception as exc:
                logger.exception("Unexpected error while running %s check: %s", name, exc)
            if not result:
                failed_checks.append(name)
                if preflight_cfg.fail_fast:
                    break

        logger.info("")
        logger.info(separator)
        if failed_checks:
            logger.error("❌ PRE-FLIGHT CHECKS FAILED")
            logger.error("Failed checks: %s", ", ".join(failed_checks))
            logger.info(separator)
            return False

        logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
        logger.info(separator)
        return True

    def run(self) -> None:
        """Synchronous helper that drives :meth:`run_async`."""

        asyncio.run(self.run_async())

    def add_loop_callback(self, callback: Callable[[], None]) -> None:
        self._loop_callbacks.append(callback)
