"""Backtrader live runtime orchestrator.

This module wires together the Databento subscriber bridge with the
:class:`backtrader.Cerebro` runtime so production workers can hydrate the ML
bundles, attach live data feeds, and run :class:`~strategy.ibs_strategy.IbsStrategy`
continuously.  The implementation favours clarity and defensive guards so the
worker can be used both for documentation and for integration tests.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import json
import logging
import os
import re
import signal
import socket
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import weakref

import backtrader as bt

import pandas as pd
import requests
from requests import exceptions as requests_exceptions

from config import PAIR_MAP, REQUIRED_REFERENCE_FEEDS
from models import (
    load_model_bundle,
    strategy_kwargs_from_bundle,
    load_factory_model_bundle,
    factory_strategy_kwargs_from_bundle,
)
from runner.contract_map import ContractMap, ContractMapError, load_contract_map
from runner.databento_bridge import (
    Bar,
    DailyResampledLiveData,
    DatabentoLiveData,
    DatabentoSubscriber,
    HourlyResampledLiveData,
    QueueFanout,
)
from runner.historical_loader import load_historical_data
from runner.ml_feature_tracker import MlFeatureTracker
from runner.portfolio_coordinator import PortfolioCoordinator
from runner.traderspost_client import (
    TradersPostClient,
    TradersPostError,
    order_notification_to_message,
    trade_notification_to_message,
)
from strategy.contract_specs import CONTRACT_SPECS
from strategy.ibs_strategy import IbsStrategy
from strategy.factory_adapter import NotifyingFactoryAdapter, VALIDATED_STRATEGIES
from runner.contract_selector import ContractSelector, ContractSelection
from utils.discord_notifier import DiscordNotifier

logger = logging.getLogger(__name__)


LIVE_BACKFILL_MAX_DAYS = 4

_VALID_WARMUP_COMPRESSIONS = {"1min", "1h", "1d"}
_WARMUP_COMPRESSION_ALIASES = {
    "minute": "1min",
    "minutes": "1min",
    "1minute": "1min",
    "1m": "1min",
    "min": "1min",
    "mins": "1min",
    "hour": "1h",
    "hours": "1h",
    "1hour": "1h",
    "1hr": "1h",
    "hourly": "1h",
    "60m": "1h",
    "day": "1d",
    "1day": "1d",
    "daily": "1d",
}
_MINUTE_FEATURE_PATTERN = re.compile(r"(?<!\d)\d{2,3}m\b|\b(?:min|mins|minute|minutes)\b")


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
        live_worker_ref = kwargs.pop("live_worker_ref", None)
        self._live_worker_ref: Optional["weakref.ReferenceType[LiveWorker]"]
        if isinstance(live_worker_ref, weakref.ReferenceType):
            self._live_worker_ref = live_worker_ref
        elif live_worker_ref is None:
            self._live_worker_ref = None
        else:  # pragma: no cover - defensive guard
            try:
                self._live_worker_ref = weakref.ref(live_worker_ref)  # type: ignore[arg-type]
            except TypeError:
                self._live_worker_ref = None
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

    def next(self):
        super().next()
        worker_ref = getattr(self, "_live_worker_ref", None)
        if not worker_ref:
            return
        worker = worker_ref()
        if worker is None:
            return
        try:
            worker._on_strategy_feature_snapshot(self)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to process ML warmup snapshot")


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
    backfill_days: Optional[int] = None
    backfill_lookback: int = 0
    load_historical_warmup: bool = False
    historical_lookback_days: int = 252  # Daily bars lookback
    historical_hourly_lookback_days: int = 15  # Hourly bars lookback (enough for 252 hourly bars)
    historical_warmup_batch_size: int = 5000
    historical_warmup_queue_soft_limit: int = 20000
    historical_warmup_compression: str = "1min"  # Deprecated: use separate daily/hourly loading
    queue_maxsize: int = 2048
    heartbeat_interval: Optional[int] = None
    heartbeat_file: Optional[Path] = None
    heartbeat_write_interval: float = 30.0
    poll_interval: float = 1.0
    traderspost_webhook: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    resample_session_start: Optional[dt.time] = dt.time(23, 0)
    instruments: Mapping[str, InstrumentRuntimeConfig] = field(default_factory=dict)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    killswitch: bool = False
    portfolio_max_positions: Optional[int] = None
    portfolio_daily_stop_loss: Optional[float] = None
    portfolio_instruments: Optional[tuple[str, ...]] = None  # Instruments to actually trade
    use_factory_strategies: bool = False  # Use Strategy Factory adapter instead of IbsStrategy
    factory_strategy_mapping: Mapping[str, str] = field(default_factory=dict)  # symbol -> strategy name

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


def _expand_env_placeholders(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively expand ``${VAR}`` placeholders for string values."""

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, Mapping):
            return {key: _expand(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [_expand(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_expand(item) for item in value)
        return value

    return {key: _expand(value) for key, value in payload.items()}


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


def _parse_time_of_day(value: Any) -> Optional[dt.time]:
    """Parse ``value`` into a :class:`datetime.time` if possible."""

    if value is None:
        return None
    if isinstance(value, dt.time):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return dt.datetime.strptime(raw, fmt).time()
            except ValueError:
                continue
        logger.warning("Invalid session start time %r; expected HH:MM or HH:MM:SS", value)
        return None
    if isinstance(value, Sequence):
        try:
            parts = [int(float(part)) for part in value[:3]]
        except (TypeError, ValueError):
            logger.warning(
                "Invalid session start sequence %r; expected [hour, minute, second]",
                value,
            )
            return None
        while len(parts) < 3:
            parts.append(0)
        hour, minute, second = parts[:3]
        try:
            return dt.time(hour=hour, minute=minute, second=second)
        except ValueError:
            logger.warning("Invalid session start sequence %r; ignoring", value)
            return None
    return None


def _normalise_warmup_compression(value: Any) -> str:
    """Coerce ``value`` into one of the supported warmup compressions."""

    if value is None:
        return "1min"
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw:
            return "1min"
        mapped = _WARMUP_COMPRESSION_ALIASES.get(raw, raw)
        return mapped if mapped in _VALID_WARMUP_COMPRESSIONS else "1min"
    if isinstance(value, (int, float)):
        # Treat common numeric aliases (e.g. 60 -> 1h, 1440 -> 1d).
        numeric = int(value)
        if numeric == 60:
            return "1h"
        if numeric == 1440:
            return "1d"
        return "1min"
    return "1min"


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

    payload = _expand_env_placeholders(_load_json_or_yaml(config_path))

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
    backfill_lookback_minutes = 0
    backfill_days_val: Optional[int] = None
    backfill_days_raw = payload.get("backfill_days")
    if backfill_days_raw is not None:
        try:
            candidate_days = int(float(backfill_days_raw))
        except (TypeError, ValueError):
            logger.warning("Invalid backfill days value %r; ignoring", backfill_days_raw)
        else:
            candidate_days = max(0, candidate_days)
            backfill_days_val = min(candidate_days, LIVE_BACKFILL_MAX_DAYS)
            backfill_lookback_minutes = backfill_days_val * 24 * 60
    if backfill_lookback_minutes == 0:
        for candidate in (
            payload.get("backfill_lookback_minutes"),
            payload.get("backfill_minutes"),
            payload.get("backfill_lookback"),
        ):
            if candidate is None:
                continue
            try:
                backfill_lookback_minutes = int(float(candidate))
            except (TypeError, ValueError):
                logger.warning("Invalid backfill lookback value %r; ignoring", candidate)
                backfill_lookback_minutes = 0
                continue
            break
    if backfill_lookback_minutes <= 0:
        hours_candidate = payload.get("backfill_lookback_hours") or payload.get("backfill_hours")
        if hours_candidate is not None:
            try:
                backfill_lookback_minutes = int(float(hours_candidate) * 60.0)
            except (TypeError, ValueError):
                logger.warning("Invalid backfill hours value %r; ignoring", hours_candidate)
                backfill_lookback_minutes = 0
    if backfill_lookback_minutes < 0:
        backfill_lookback_minutes = 0
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

    load_historical_warmup = _coerce_bool(payload.get("load_historical_warmup"), False)
    historical_lookback_days_raw = payload.get("historical_lookback_days")
    historical_lookback_days = 252
    if historical_lookback_days_raw is not None:
        try:
            historical_lookback_days = max(1, int(float(historical_lookback_days_raw)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid historical lookback days value %r; defaulting to 252",
                historical_lookback_days_raw,
            )

    historical_hourly_lookback_days_raw = payload.get("historical_hourly_lookback_days")
    historical_hourly_lookback_days = 15
    if historical_hourly_lookback_days_raw is not None:
        try:
            historical_hourly_lookback_days = max(1, int(float(historical_hourly_lookback_days_raw)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid historical hourly lookback days value %r; defaulting to 15",
                historical_hourly_lookback_days_raw,
            )

    historical_warmup_batch_size_raw = payload.get("historical_warmup_batch_size")
    if historical_warmup_batch_size_raw is None:
        historical_warmup_batch_size = 5000
    else:
        try:
            historical_warmup_batch_size = max(1, int(float(historical_warmup_batch_size_raw)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid historical warmup batch size %r; defaulting to 5000",
                historical_warmup_batch_size_raw,
            )
            historical_warmup_batch_size = 5000

    historical_warmup_queue_soft_limit_raw = payload.get("historical_warmup_queue_soft_limit")
    if historical_warmup_queue_soft_limit_raw is None:
        historical_warmup_queue_soft_limit = 20000
    else:
        try:
            historical_warmup_queue_soft_limit = int(float(historical_warmup_queue_soft_limit_raw))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid historical warmup queue limit %r; defaulting to 20000",
                historical_warmup_queue_soft_limit_raw,
            )
            historical_warmup_queue_soft_limit = 20000
    if historical_warmup_queue_soft_limit < historical_warmup_batch_size:
        historical_warmup_queue_soft_limit = historical_warmup_batch_size

    historical_warmup_compression = _normalise_warmup_compression(
        payload.get("historical_warmup_compression")
    )
    logger.info(
        "Historical warmup compression configured: %s (raw value: %r)",
        historical_warmup_compression,
        payload.get("historical_warmup_compression"),
    )

    traderspost_payload = payload.get("traderspost") or {}
    webhook_url = (
        payload.get("traderspost_webhook")
        or traderspost_payload.get("webhook")
        or os.environ.get("TRADERSPOST_WEBHOOK_URL")
    )

    # Discord webhook configuration
    discord_webhook = (
        payload.get("discord_webhook_url")
        or payload.get("discord_webhook")
        or os.environ.get("DISCORD_WEBHOOK_URL")
    )

    session_start_raw = (
        payload.get("resample_session_start")
        or payload.get("session_start")
        or payload.get("session_anchor")
    )
    session_start = _parse_time_of_day(session_start_raw) or dt.time(23, 0)
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

    killswitch_raw = payload.get("killswitch")
    if killswitch_raw is None:
        killswitch_raw = payload.get("policy_killswitch")
    if killswitch_raw is not None:
        killswitch = _coerce_bool(killswitch_raw, False)
    else:
        killswitch = _coerce_bool(os.environ.get("POLICY_KILLSWITCH"), False)

    # Portfolio configuration
    portfolio_payload = payload.get("portfolio") or {}
    portfolio_max_positions = None
    portfolio_daily_stop_loss = None
    portfolio_instruments = None
    use_factory_strategies = False
    factory_strategy_mapping: Dict[str, str] = {}

    if portfolio_payload:
        use_factory_strategies = _coerce_bool(portfolio_payload.get("use_factory_strategies"), False)
        max_pos_raw = portfolio_payload.get("max_positions")
        if max_pos_raw is not None:
            try:
                portfolio_max_positions = int(max_pos_raw)
            except (TypeError, ValueError):
                logger.warning("Invalid portfolio max_positions value %r; ignoring", max_pos_raw)

        stop_loss_raw = portfolio_payload.get("daily_stop_loss")
        if stop_loss_raw is not None:
            try:
                portfolio_daily_stop_loss = float(stop_loss_raw)
            except (TypeError, ValueError):
                logger.warning("Invalid portfolio daily_stop_loss value %r; ignoring", stop_loss_raw)

        # Read instruments to actually trade (vs all symbols loaded for features)
        # Supports two formats:
        # 1. Simple list: ["ES", "NQ", "CL"]
        # 2. Strategy-specified list: [{symbol: ES, strategy: AvgHLRangeIBS}, ...]
        instruments_raw = portfolio_payload.get("instruments")
        if instruments_raw:
            parsed_instruments = []
            for item in instruments_raw:
                if isinstance(item, str):
                    # Simple format: just symbol name
                    parsed_instruments.append(item.strip().upper())
                elif isinstance(item, Mapping):
                    # Strategy-specified format
                    sym = item.get("symbol", "")
                    if sym:
                        parsed_instruments.append(str(sym).strip().upper())
                        strategy_name = item.get("strategy")
                        if strategy_name:
                            factory_strategy_mapping[str(sym).strip().upper()] = str(strategy_name)
            portfolio_instruments = tuple(parsed_instruments) if parsed_instruments else None

    return RuntimeConfig(
        databento_api_key=payload.get("databento_api_key") or os.environ.get("DATABENTO_API_KEY"),
        contract_map_path=contract_map_path,
        models_path=models_path,
        symbols=symbols,
        starting_cash=starting_cash_val,
        backfill=backfill,
        backfill_days=backfill_days_val,
        backfill_lookback=backfill_lookback_minutes,
        load_historical_warmup=load_historical_warmup,
        historical_lookback_days=historical_lookback_days,
        historical_hourly_lookback_days=historical_hourly_lookback_days,
        historical_warmup_batch_size=historical_warmup_batch_size,
        historical_warmup_queue_soft_limit=historical_warmup_queue_soft_limit,
        historical_warmup_compression=historical_warmup_compression,
        queue_maxsize=queue_maxsize,
        heartbeat_interval=heartbeat_interval,
        heartbeat_file=heartbeat_file,
        heartbeat_write_interval=heartbeat_write_interval,
        poll_interval=poll_interval,
        traderspost_webhook=webhook_url,
        discord_webhook_url=discord_webhook,
        resample_session_start=session_start,
        instruments=instruments,
        preflight=preflight_config,
        killswitch=killswitch,
        portfolio_max_positions=portfolio_max_positions,
        portfolio_daily_stop_loss=portfolio_daily_stop_loss,
        portfolio_instruments=portfolio_instruments,
        use_factory_strategies=use_factory_strategies,
        factory_strategy_mapping=factory_strategy_mapping,
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

        reference_symbols = set(self.contract_map.reference_symbols())
        for symbol in self.symbols:
            pair_symbol = PAIR_MAP.get(symbol)
            if not pair_symbol:
                continue
            if pair_symbol in self.symbols:
                continue
            if pair_symbol in self.contract_map or self.contract_map.reference_feed(pair_symbol):
                reference_symbols.add(pair_symbol)
        self.reference_symbols = tuple(sorted(reference_symbols))

        product_to_root: Dict[str, str] = {}
        for code, symbol in self.contract_map.product_to_root(contract_symbols).items():
            product_to_root[code] = symbol
        for code, symbol in self.contract_map.reference_product_to_root(
            self.reference_symbols
        ).items():
            product_to_root.setdefault(code, symbol)

        self.data_symbols = tuple(sorted(set(contract_symbols) | set(self.reference_symbols)))

        # Initialize contract selector for highest OI contract selection
        # Must be done before building dataset groups so we can subscribe to selected contracts
        self.contract_selector: Optional[ContractSelector] = None
        self._selected_contracts: Dict[str, ContractSelection] = {}
        trading_symbols = list(config.portfolio_instruments or self.symbols)

        if config.databento_api_key and trading_symbols:
            try:
                self.contract_selector = ContractSelector(
                    api_key=config.databento_api_key,
                    dataset="GLBX.MDP3",
                )
                logger.info("Selecting highest OI contracts for %d trading symbols...", len(trading_symbols))
                self._selected_contracts = self.contract_selector.select_contracts(trading_symbols)
                for sym, sel in self._selected_contracts.items():
                    logger.info(
                        "  %s -> %s (OI=%d, Vol=%d)",
                        sym, sel.contract_symbol, sel.open_interest, sel.volume
                    )
            except Exception:
                logger.exception("Failed to initialize contract selector; using default contracts")
                self.contract_selector = None

        # Build dataset groups, using selected contracts for trading symbols
        dataset_groups, symbols_by_group = self._build_dataset_groups(
            contract_symbols, self.reference_symbols
        )
        self._symbols_by_dataset_group = symbols_by_group

        # Update product_to_root mapping to include selected contract symbols
        for sym, selection in self._selected_contracts.items():
            # Map the selected contract symbol back to root (e.g., "ESH6" -> "ES")
            product_to_root[selection.contract_symbol] = sym
            # Also map full year format (e.g., "ESH2026" -> "ES")
            product_to_root[selection.contract_symbol.replace("202", "2")] = sym

        self.queue_manager = QueueFanout(product_to_root=product_to_root, maxsize=config.queue_maxsize)
        self._data_feeds: Dict[str, bt.feeds.DataBase] = {}
        self.subscribers = []
        backfill_start: Optional[dt.datetime] = None
        if self.config.backfill and self.config.backfill_lookback > 0:
            backfill_start = (
                dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(minutes=self.config.backfill_lookback)
            ).replace(second=0, microsecond=0)

        earliest_start_cache: dict[str, Optional[dt.datetime]] = {}

        for (dataset, stype_in), codes in dataset_groups.items():
            start = backfill_start
            if start is not None:
                boundary = earliest_start_cache.get(dataset)
                if dataset not in earliest_start_cache:
                    boundary = self._earliest_live_start_for_dataset(dataset)
                    earliest_start_cache[dataset] = boundary
                if boundary is not None and start < boundary:
                    logger.info(
                        "Clamping backfill start for dataset=%s stype=%s from %s to %s",
                        dataset,
                        stype_in,
                        start.isoformat(),
                        boundary.isoformat(),
                    )
                    start = boundary
            self.subscribers.append(
                DatabentoSubscriber(
                    dataset=dataset,
                    product_codes=sorted(codes),
                    queue_manager=self.queue_manager,
                    api_key=config.databento_api_key,
                    heartbeat_interval=config.heartbeat_interval,
                    stype_in=stype_in,
                    start=start,
                )
            )

        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(config.starting_cash)
        self._cerebro_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop_callbacks: list[Callable[[], None]] = []
        self._last_heartbeat_update: float = 0.0
        self._preflight_summary: dict[str, Any] = {"status": "not_run"}
        self._traderspost_status: dict[str, Any] = {
            "last_success": None,
            "last_error": None,
        }

        self.ml_feature_tracker = MlFeatureTracker()
        self._ml_feature_lock = threading.Lock()
        self._pending_ml_warmup: set[str] = set()
        self._ml_features_seen: dict[str, set[str]] = {}
        self._ml_feature_collectors: dict[str, MlFeatureTracker.Collector] = {}
        self._ml_feature_requirements: dict[str, tuple[str, ...]] = {}
        self._ml_warmup_published: dict[str, set[str]] = {}
        self._minute_warmup_symbols: set[str] = set()

        self.traderspost_client: Optional[TradersPostClient]
        if config.traderspost_webhook:
            try:
                self.traderspost_client = TradersPostClient(
                    config.traderspost_webhook,
                )
                logger.info("TradersPost client initialised")
            except Exception:
                logger.exception("Failed to initialise TradersPost client")
                self.traderspost_client = None
        else:
            logger.info(
                "TradersPost client disabled: no webhook URL configured"
            )
            self.traderspost_client = None

        # Initialize Discord notifier for alerts and notifications
        self.discord_notifier: Optional[DiscordNotifier]
        if config.discord_webhook_url:
            try:
                self.discord_notifier = DiscordNotifier(config.discord_webhook_url)
                logger.info("Discord notifier initialized")
                # Send startup notification
                self.discord_notifier.send_system_alert(
                    title="System Started",
                    message=f"Rooney Capital trading system started\nSymbols: {', '.join(config.symbols)}",
                    alert_type="info",
                )
            except Exception:
                logger.exception("Failed to initialize Discord notifier")
                self.discord_notifier = None
        else:
            logger.info("Discord notifier disabled: no webhook URL configured")
            self.discord_notifier = None

        # Initialize portfolio coordinator for portfolio-wide constraints
        self.portfolio_coordinator: Optional[PortfolioCoordinator] = None

        # Read portfolio settings from config.yml
        if config.portfolio_max_positions is not None and config.portfolio_daily_stop_loss is not None:
            max_positions = config.portfolio_max_positions
            daily_stop_loss = config.portfolio_daily_stop_loss

            # Filter to only traded instruments if specified
            if config.portfolio_instruments:
                original_count = len(self.symbols)
                filtered_symbols = tuple(s for s in self.symbols if s in config.portfolio_instruments)
                if len(filtered_symbols) < original_count:
                    logger.info(
                        "📊 Filtered symbols: %d loaded for features → %d for trading",
                        original_count, len(filtered_symbols)
                    )
                    logger.info("   Loaded (all): %s", ', '.join(sorted(self.symbols)))
                    logger.info("   Trading (filtered): %s", ', '.join(sorted(filtered_symbols)))
                    self.symbols = filtered_symbols
                else:
                    logger.info("All %d loaded symbols will be traded", len(self.symbols))

            logger.info(
                "Portfolio config loaded from config.yml: max_positions=%d, daily_stop_loss=$%.0f",
                max_positions, daily_stop_loss
            )

            # Create portfolio coordinator with emergency exit callback
            def emergency_exit_callback(reason: str, context: dict) -> None:
                """Handle portfolio stop loss trigger."""
                logger.critical(
                    "🚨 PORTFOLIO STOP LOSS TRIGGERED 🚨\n"
                    "Reason: %s\nDaily P&L: $%.2f\nOpen positions: %s",
                    reason,
                    context.get('daily_pnl', 0),
                    ', '.join(context.get('open_positions', []))
                )

                # Send Discord alert
                if self.discord_notifier:
                    try:
                        self.discord_notifier.send_system_alert(
                            title="🚨 PORTFOLIO STOP LOSS HIT",
                            message=(
                                f"Daily P&L: ${context.get('daily_pnl', 0):,.2f}\n"
                                f"Limit: ${context.get('stop_loss_limit', 0):,.2f}\n"
                                f"Open positions: {', '.join(context.get('open_positions', []))}\n"
                                f"Time: {context.get('trigger_time', 'unknown')}"
                            ),
                            alert_type="critical"
                        )
                    except Exception:
                        logger.exception("Failed to send Discord alert for stop loss")

                # Note: Individual strategies will handle closing their positions
                # when they receive the stopped_out signal from coordinator

            self.portfolio_coordinator = PortfolioCoordinator(
                max_positions=max_positions,
                daily_stop_loss=daily_stop_loss,
                emergency_exit_callback=emergency_exit_callback,
                daily_summary_callback=self.send_daily_summary
            )
            logger.info("Portfolio coordinator initialized successfully")
        else:
            logger.info(
                "No portfolio configuration in config.yml - running without portfolio constraints"
            )
            self.portfolio_coordinator = None

        self._setup_data_and_strategies()

        self._historical_warmup_counts: dict[str, int] = {}
        self._historical_warmup_started = False
        self._historical_warmup_lock = threading.Lock()
        self._historical_warmup_wait_log_interval = 5.0

        if self.config.heartbeat_file:
            logger.info("Heartbeat file configured at %s", self.config.heartbeat_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strategy_uses_minute_timeframe(
        self,
        strategy_cls: type,
        strategy_kwargs: Mapping[str, Any],
    ) -> bool:
        """Return ``True`` when timeframe params require minute warmup bars."""

        for candidate in self._iter_timeframe_values(strategy_cls, strategy_kwargs):
            if self._timeframe_value_requires_minutes(candidate):
                return True
        return False

    @staticmethod
    def _iter_timeframe_values(
        strategy_cls: type | None,
        overrides: Mapping[str, Any],
    ) -> Sequence[Any]:
        values: list[Any] = []
        params = getattr(strategy_cls, "params", None) if strategy_cls is not None else None
        values.extend(LiveWorker._extract_timeframe_values(params))
        values.extend(LiveWorker._extract_timeframe_values(overrides))
        return values

    @staticmethod
    def _extract_timeframe_values(source: Any) -> list[Any]:
        extracted: list[Any] = []
        if isinstance(source, Mapping):
            for key, value in source.items():
                key_str = str(key).lower()
                if key_str.endswith("tf") or "timeframe" in key_str:
                    extracted.append(value)
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            for item in source:
                if isinstance(item, tuple) and len(item) >= 2:
                    key = str(item[0]).lower()
                    if key.endswith("tf") or "timeframe" in key:
                        extracted.append(item[1])
        return extracted

    @staticmethod
    def _timeframe_value_requires_minutes(value: Any) -> bool:
        if isinstance(value, str):
            label = value.strip().lower()
            if not label:
                return False
            if label in {"minute", "minutes"}:
                return True
            if _MINUTE_FEATURE_PATTERN.search(label):
                return True
        try:
            timeframe_value = int(value)
        except Exception:
            timeframe_value = None
        if timeframe_value is not None:
            try:
                if timeframe_value == int(bt.TimeFrame.Minutes):
                    return True
            except Exception:  # pragma: no cover - defensive guard
                return False
        return value == bt.TimeFrame.Minutes

    @staticmethod
    def _feature_name_implies_minute(feature: Any) -> bool:
        if not isinstance(feature, str):
            return False
        label = feature.strip().lower()
        if not label:
            return False
        if label in {"minute", "minutes"}:
            return True
        return bool(_MINUTE_FEATURE_PATTERN.search(label))

    def _symbol_requires_minute_warmup(self, symbol: str) -> bool:
        canonical = str(symbol or "").strip().upper()
        if not canonical:
            logger.debug("Empty symbol requires minute warmup")
            return True
        minute_symbols = getattr(self, "_minute_warmup_symbols", set())
        if canonical in minute_symbols:
            logger.debug(
                "%s requires minute warmup (strategy uses minute timeframe)",
                canonical,
            )
            return True
        requirements = getattr(self, "_ml_feature_requirements", {})
        for feature in requirements.get(canonical, ()):  # type: ignore[arg-type]
            if self._feature_name_implies_minute(feature):
                logger.debug(
                    "%s requires minute warmup (ML feature %s implies minute data)",
                    canonical,
                    feature,
                )
                return True
        logger.debug(
            "%s does not require minute warmup (strategy timeframes: %s, ML features: %s)",
            canonical,
            "minute" if canonical in minute_symbols else "hourly/daily",
            len(requirements.get(canonical, ())),
        )
        return False

    def _build_dataset_groups(
        self,
        contract_symbols: Sequence[str],
        reference_symbols: Sequence[str],
    ) -> tuple[
        dict[tuple[str, str], tuple[str, ...]],
        dict[tuple[str, str], tuple[str, ...]],
    ]:
        """Return grouped subscription codes and their associated symbols.

        For trading symbols with selected contracts (highest OI), uses
        stype_in='raw_symbol' with the specific contract code.
        For reference symbols, uses the default from contract_map.
        """

        grouped_codes: dict[tuple[str, str], set[str]] = {}
        grouped_symbols: dict[tuple[str, str], set[str]] = {}

        for symbol in sorted(set(contract_symbols) | set(reference_symbols)):
            symbol_upper = symbol.upper()

            # Check if we have a selected contract for this symbol
            selection = self._selected_contracts.get(symbol_upper)
            if selection:
                # Use the selected contract with raw_symbol stype
                subscription = self.contract_map.subscription_for(symbol)
                if not subscription:
                    continue
                dataset = subscription.dataset

                # Convert full year to short format for Databento (ESH2026 -> ESH6)
                contract_code = selection.contract_symbol
                # Extract the year part and convert to single digit
                # ESH2026 -> ESH6, CLZ2025 -> CLZ5
                if len(contract_code) > 4 and contract_code[-4:].isdigit():
                    year_digit = contract_code[-1]  # Last digit of year
                    contract_code = contract_code[:-4] + year_digit

                key = (dataset, "raw_symbol")
                grouped_codes.setdefault(key, set()).add(contract_code)
                grouped_symbols.setdefault(key, set()).add(symbol)
                logger.info(
                    "Subscribing to selected contract: %s -> %s (stype=raw_symbol)",
                    symbol, contract_code
                )
            else:
                # Use default subscription from contract map
                subscription = self.contract_map.subscription_for(symbol)
                if not subscription:
                    continue
                key = (subscription.dataset, subscription.stype_in)
                if subscription.codes:
                    grouped_codes.setdefault(key, set()).update(subscription.codes)
                grouped_symbols.setdefault(key, set()).add(symbol)

        dataset_groups = {
            key: tuple(sorted(codes)) for key, codes in grouped_codes.items()
        }
        symbols_by_group = {
            key: tuple(sorted(symbols)) for key, symbols in grouped_symbols.items()
        }
        return dataset_groups, symbols_by_group

    def _earliest_live_start_for_dataset(self, dataset: str) -> Optional[dt.datetime]:
        """Return the earliest timestamp permitted for live subscriptions."""

        return dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    def _setup_data_and_strategies(self) -> None:
        session_start = self.config.resample_session_start
        hour_kwargs: Dict[str, Any] = {
            "timeframe": bt.TimeFrame.Minutes,
            "compression": 60,
            "bar2edge": True,
            "adjbartime": True,
            "rightedge": False,
        }
        day_kwargs: Dict[str, Any] = {
            "timeframe": bt.TimeFrame.Days,
            "compression": 1,
            "bar2edge": True,
            "adjbartime": True,
            "rightedge": False,
        }
        if session_start:
            day_kwargs["sessionend"] = session_start
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

            # Create custom hourly feed that aggregates from base feed
            # During warmup: consumes pre-aggregated hourly bars
            # During live: aggregates minute bars from 'data' into hourly bars
            hourly_feed = HourlyResampledLiveData(
                symbol=symbol,
                source_feed=data,  # Aggregate from minute feed
                session_end_hour=session_start.hour if session_start else 23,
                session_end_minute=session_start.minute if session_start else 0,
            )
            self.cerebro.adddata(hourly_feed, name=f"{symbol}_hour")

            # Create custom daily feed that aggregates from base feed
            # During warmup: consumes pre-aggregated daily bars
            # During live: aggregates minute bars from 'data' into daily bars
            daily_feed = DailyResampledLiveData(
                symbol=symbol,
                source_feed=data,  # Aggregate from minute feed
                session_end_hour=session_start.hour if session_start else 23,
                session_end_minute=session_start.minute if session_start else 0,
            )
            self.cerebro.adddata(daily_feed, name=f"{symbol}_day")

        for symbol in self.symbols:
            instrument_cfg = self.config.instrument(symbol)

            ml_features: Sequence[str] | None = None

            # Choose model loader based on strategy type
            if self.config.use_factory_strategies:
                # Use Strategy Factory model loader for meta-labeling models
                strategy_name = self.config.factory_strategy_mapping.get(
                    symbol.upper(),
                    VALIDATED_STRATEGIES.get(symbol.upper(), {}).get('strategy', 'AvgHLRangeIBS')
                )
                try:
                    bundle = load_factory_model_bundle(
                        symbol,
                        strategy_name=strategy_name,
                        base_dir=self.config.models_path
                    )
                    bundle_kwargs = factory_strategy_kwargs_from_bundle(bundle)
                    ml_features = bundle_kwargs.get("ml_features")
                    logger.info(
                        "Loaded Strategy Factory ML bundle for %s (%s) with %d features",
                        symbol, strategy_name, len(bundle.features)
                    )
                except FileNotFoundError:
                    logger.warning(
                        "No Strategy Factory ML bundle found for %s; running without ML filter",
                        symbol
                    )
                    bundle_kwargs = {}
                    ml_features = ()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Failed to load Strategy Factory ML bundle for %s: %s", symbol, exc)
                    bundle_kwargs = {}
                    ml_features = ()
            else:
                # Use original IBS model loader
                try:
                    bundle = load_model_bundle(symbol, base_dir=self.config.models_path)
                    bundle_kwargs = strategy_kwargs_from_bundle(bundle)
                    ml_features = bundle_kwargs.get("ml_features")
                    logger.info("Loaded ML bundle for %s with features=%s", symbol, bundle.features)
                except FileNotFoundError:
                    logger.warning("No ML bundle found for %s; running without ML filter", symbol)
                    bundle_kwargs = {}
                    ml_features = ()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Failed to load ML bundle for %s: %s", symbol, exc)
                    bundle_kwargs = {}
                    ml_features = ()

            strategy_kwargs: Dict[str, Any] = {"symbol": symbol, "size": instrument_cfg.size}
            strategy_kwargs.update(bundle_kwargs)
            strategy_kwargs.update(instrument_cfg.strategy_overrides)

            # Inject portfolio coordinator if available
            if self.portfolio_coordinator:
                strategy_kwargs['portfolio_coordinator'] = self.portfolio_coordinator

            canonical_symbol = symbol.strip().upper()
            if self._strategy_uses_minute_timeframe(NotifyingIbsStrategy, strategy_kwargs):
                self._minute_warmup_symbols.add(canonical_symbol)

            if ml_features is not None:
                collector = self.ml_feature_tracker.register_bundle(symbol, ml_features)
                if ml_features:
                    canonical_symbol = symbol.strip().upper()
                    if canonical_symbol:
                        self._ml_feature_collectors[canonical_symbol] = collector
                        normalised_features = tuple(
                            str(feature).strip() for feature in ml_features if str(feature).strip()
                        )
                        self._ml_feature_requirements[canonical_symbol] = normalised_features
                    strategy_kwargs["ml_feature_collector"] = collector
                else:
                    self.ml_feature_tracker.register_bundle(symbol, ())

            order_callbacks: list[Callable[[NotifyingIbsStrategy, Any], None]] = []
            trade_callbacks: list[
                Callable[[NotifyingIbsStrategy, Any, Optional[Mapping[str, Any]]], None]
            ] = []
            if self.traderspost_client:
                order_callbacks.append(self._traderspost_order_callback)
                trade_callbacks.append(self._traderspost_trade_callback)

            if self.discord_notifier:
                trade_callbacks.append(self._discord_trade_callback)

            # Choose strategy class based on configuration
            if self.config.use_factory_strategies:
                # Use Strategy Factory adapter with per-symbol strategy
                strategy_name = self.config.factory_strategy_mapping.get(
                    symbol.upper(),
                    VALIDATED_STRATEGIES.get(symbol.upper(), {}).get('strategy', 'AvgHLRangeIBS')
                )
                logger.info(
                    "Using Strategy Factory adapter for %s with strategy=%s",
                    symbol, strategy_name
                )
                self.cerebro.addstrategy(
                    NotifyingFactoryAdapter,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    strategy_params={},  # Default params; can be extended from config
                    order_callbacks=order_callbacks,
                    trade_callbacks=trade_callbacks,
                    portfolio_coordinator=self.portfolio_coordinator,
                    feature_tracker=self.ml_feature_tracker,
                    **{k: v for k, v in strategy_kwargs.items()
                       if k not in ('symbol', 'portfolio_coordinator')},
                )
            else:
                # Use original IbsStrategy
                self.cerebro.addstrategy(
                    NotifyingIbsStrategy,
                    order_callbacks=order_callbacks,
                    trade_callbacks=trade_callbacks,
                    live_worker_ref=weakref.ref(self),
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

    def _mark_warmup_features_pending(self, symbol: str) -> None:
        tracker = getattr(self, "ml_feature_tracker", None)
        if tracker is None:
            return
        canonical = str(symbol or "").strip().upper()
        if not canonical:
            return
        with self._ml_feature_lock:
            self._pending_ml_warmup.add(canonical)
            self._ml_features_seen.pop(canonical, None)

    def _run_historical_warmup(self) -> None:
        if not (self.config.load_historical_warmup and self.symbols):
            return

        if not self._symbols_by_dataset_group or not self.config.databento_api_key:
            logger.info(
                "Historical warmup skipped: dataset or API key unavailable"
            )
            return

        with self._historical_warmup_lock:
            if self._historical_warmup_started:
                return
            self._historical_warmup_started = True

        daily_lookback_days = self.config.historical_lookback_days or 252
        hourly_lookback_days = self.config.historical_hourly_lookback_days or 15

        logger.info(
            "Loading dual-timeframe historical data: %d days daily + %d days hourly for indicator warmup...",
            daily_lookback_days,
            hourly_lookback_days,
        )

        self._historical_warmup_counts.clear()
        try:
            # Load daily bars first (252 days for daily z-scores and returns)
            logger.info("Loading daily historical data (%d days)...", daily_lookback_days)
            for (dataset, stype_in), symbols in self._symbols_by_dataset_group.items():
                # Use closure factory to avoid lambda capture bug
                def daily_warmup_callback(sym: str, data: Any) -> None:
                    self._warmup_symbol_indicators(sym, data, compression="1D")

                load_historical_data(
                    api_key=self.config.databento_api_key,
                    dataset=dataset,
                    symbols=symbols,
                    stype_in=stype_in,
                    days=daily_lookback_days,
                    contract_map=self.contract_map,
                    on_symbol_loaded=daily_warmup_callback,
                )

            # Load hourly bars second (15 days for hourly indicators)
            logger.info("Loading hourly historical data (%d days)...", hourly_lookback_days)
            for (dataset, stype_in), symbols in self._symbols_by_dataset_group.items():
                # Use closure factory to avoid lambda capture bug
                def hourly_warmup_callback(sym: str, data: Any) -> None:
                    self._warmup_symbol_indicators(sym, data, compression="1h")

                load_historical_data(
                    api_key=self.config.databento_api_key,
                    dataset=dataset,
                    symbols=symbols,
                    stype_in=stype_in,
                    days=hourly_lookback_days,
                    contract_map=self.contract_map,
                    on_symbol_loaded=hourly_warmup_callback,
                )
        except Exception:
            logger.exception("Failed to load historical data for warmup")
            self._historical_warmup_counts.clear()
            with self._historical_warmup_lock:
                self._historical_warmup_started = False
            return

        logger.info("Historical data loaded - warmup bars queued to feeds")

        # Enable fast warmup mode - Cerebro will drain bars quickly when it starts
        self._set_strategies_warmup_mode(enabled=True)

        # Note: We don't drain here because Cerebro isn't running yet.
        # The warmup bars will be consumed automatically when cerebro.run() starts
        # thanks to the _qcheck=0.0 optimization in the custom feeds.

        self._finalize_indicator_warmup()

    def _warmup_symbol_indicators(self, symbol: str, data: Any, compression: str = None) -> None:
        # If compression not specified, use configured compression (for backward compatibility)
        if compression is None:
            configured_compression = _normalise_warmup_compression(
                getattr(self.config, "historical_warmup_compression", "1min")
            )
            if self._symbol_requires_minute_warmup(symbol):
                compression = "1min"
                if configured_compression != "1min":
                    logger.warning(
                        "Minute warmup enforced for %s despite configured compression %s (reason: strategy or ML features require minute data)",
                        symbol,
                        configured_compression,
                    )
            else:
                compression = configured_compression
        else:
            # Compression explicitly specified (dual-timeframe warmup)
            compression = _normalise_warmup_compression(compression)

        logger.info(
            "Using %s compression for %s historical warmup",
            compression,
            symbol,
        )

        # Determine target feed based on compression
        # We need to queue directly to resampled feeds, but they don't have extend_warmup()
        # So we'll add it dynamically if needed
        if compression == "1d":
            feed_name = f"{symbol}_day"
            feed = self._get_resampled_feed(symbol, "_day")
        elif compression == "1h":
            feed_name = f"{symbol}_hour"
            feed = self._get_resampled_feed(symbol, "_hour")
        else:
            # Minute data goes to base feed
            feed_name = symbol
            feed = self._data_feeds.get(symbol)

        if feed is None:
            logger.warning(
                "Skipping %s warmup for %s: no matching feed found", compression, symbol
            )
            return

        # Our custom resampled feeds have extend_warmup() built-in
        if not hasattr(feed, "extend_warmup"):
            logger.warning(
                "Feed %s does not support warmup (missing extend_warmup method)", feed_name
            )
            return

        try:
            bars = self._convert_databento_to_bt_bars(symbol, data, compression=compression)
        except Exception as exc:
            logger.error("Failed to convert warmup data for %s: %s", symbol, exc)
            raise

        if not bars:
            logger.info("No historical bars available for %s warmup", symbol)
            return

        logger.info(
            "Converted %s historical data to %d %s bars (queuing to feed: %s)",
            symbol,
            len(bars),
            compression,
            feed_name,
        )

        batch_size = max(int(self.config.historical_warmup_batch_size), 1)
        queue_limit = max(
            int(self.config.historical_warmup_queue_soft_limit),
            batch_size,
        )

        total_appended = 0
        total_batches = max((len(bars) + batch_size - 1) // batch_size, 1)
        for batch_index, start in enumerate(range(0, len(bars), batch_size)):
            chunk = bars[start : start + batch_size]
            if not chunk:
                continue

            if total_appended > 0:
                remaining_batches = max(total_batches - batch_index, 1)
                self._wait_for_warmup_capacity(
                    feed,
                    symbol=symbol,
                    incoming=len(chunk),
                    limit=queue_limit,
                    pending_batches=remaining_batches,
                    total_batches=total_batches,
                )

            appended = int(feed.extend_warmup(chunk))
            total_appended += appended

        previous = int(self._historical_warmup_counts.get(symbol, 0))
        self._historical_warmup_counts[symbol] = previous + total_appended
        logger.info(
            "Buffered %d historical bars for %s (feed: %s)", total_appended, symbol, feed_name
        )
        if total_appended:
            self._mark_warmup_features_pending(symbol)

    def _wait_for_warmup_capacity(
        self,
        feed: Any,
        *,
        symbol: str,
        incoming: int,
        limit: int,
        pending_batches: int = 1,
        total_batches: Optional[int] = None,
    ) -> None:
        if incoming <= 0:
            return

        base_limit = max(limit, incoming)
        pending_batches = max(int(pending_batches), 1)
        total_batches = max(int(total_batches or pending_batches), pending_batches)
        extra_batches = max(pending_batches - 1, total_batches - 1)
        if incoming:
            effective_limit = base_limit + extra_batches * incoming
        else:
            effective_limit = base_limit
        log_interval = max(float(self._historical_warmup_wait_log_interval), 0.0)
        next_log: Optional[float]
        if log_interval:
            next_log = time.monotonic()
        else:
            next_log = None

        while not self._stop_event.is_set():
            backlog = self._warmup_backlog_size(feed)
            if backlog + incoming <= effective_limit:
                return

            now = time.monotonic()
            if next_log is not None and now >= next_log:
                logger.info(
                    "Waiting for warmup backlog to drain for %s (queued=%d, limit=%d)",
                    symbol,
                    backlog,
                    effective_limit,
                )
                next_log = now + log_interval

            time.sleep(0.05)

        logger.debug("Warmup backlog wait aborted for %s: stop requested", symbol)

    @staticmethod
    def _warmup_backlog_size(feed: Any) -> int:
        getter = getattr(feed, "warmup_backlog_size", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to query warmup backlog size", exc_info=True)
        deque_obj = getattr(feed, "_warmup_bars", None)
        if deque_obj is None:
            return 0
        try:
            return len(deque_obj)
        except Exception:  # pragma: no cover - defensive guard
            return 0

    def _drain_warmup_backlog(
        self,
        symbols: Sequence[str],
        *,
        poll_interval: Optional[float] = None,
        max_backlog_poll_iterations: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        if not symbols:
            return True

        # Track all feeds that have warmup bars (base feeds AND resampled feeds)
        tracked: dict[str, Any] = {}

        # Add base feeds
        for raw_symbol in symbols:
            if raw_symbol is None:
                continue
            symbol = str(raw_symbol)
            if not symbol:
                continue
            feed = self._data_feeds.get(symbol)
            if feed is not None and hasattr(feed, 'warmup_backlog_size'):
                tracked[symbol] = feed

        # Add resampled feeds that have warmup support
        try:
            for data in self.cerebro.datas:
                if not hasattr(data, 'warmup_backlog_size'):
                    continue
                feed_name = getattr(data, '_name', None)
                if feed_name and feed_name not in tracked:
                    tracked[feed_name] = data
        except Exception as exc:
            logger.debug("Error checking resampled feeds for warmup: %s", exc)

        if not tracked:
            return True

        sleep_interval = 0.05 if poll_interval is None else max(float(poll_interval), 0.0)
        log_interval = max(float(self._historical_warmup_wait_log_interval), 0.0)
        next_log = time.monotonic() + log_interval if log_interval else None
        start_time = time.monotonic()
        iterations = 0

        while not self._stop_event.is_set():
            pending: dict[str, int] = {}
            for symbol, feed in tracked.items():
                backlog = self._warmup_backlog_size(feed)
                if backlog > 0:
                    pending[symbol] = backlog

            if not pending:
                return True

            iterations += 1
            now = time.monotonic()

            if timeout is not None and now - start_time >= timeout:
                summary = ", ".join(
                    f"{symbol}:{size}" for symbol, size in sorted(pending.items())
                )
                logger.warning(
                    "Warmup backlog drain timed out after %.1f seconds; outstanding=%s",
                    now - start_time,
                    summary or "none",
                )
                return False

            if (
                max_backlog_poll_iterations is not None
                and iterations >= max_backlog_poll_iterations
            ):
                summary = ", ".join(
                    f"{symbol}:{size}" for symbol, size in sorted(pending.items())
                )
                logger.warning(
                    "Warmup backlog drain aborted after %s iterations; outstanding=%s",
                    iterations,
                    summary or "none",
                )
                return False

            if next_log is not None and now >= next_log:
                summary = ", ".join(
                    f"{symbol}:{size}" for symbol, size in sorted(pending.items())
                )
                logger.info(
                    "Waiting for warmup backlog to drain (%s)",
                    summary,
                )
                next_log = now + log_interval

            time.sleep(sleep_interval)

        summary = ", ".join(
            f"{symbol}:{self._warmup_backlog_size(feed)}"
            for symbol, feed in sorted(tracked.items())
            if self._warmup_backlog_size(feed) > 0
        )
        logger.debug(
            "Warmup backlog drain aborted: stop requested; outstanding=%s",
            summary or "none",
        )
        return False

    def _convert_databento_to_bt_bars(
        self,
        symbol: str,
        data: Any,
        *,
        compression: str = "1min",
    ) -> list[Bar]:
        if data is None:
            return []

        if hasattr(data, "to_df"):
            df = data.to_df()
        elif hasattr(data, "to_pandas"):
            df = data.to_pandas()
        else:
            df = pd.DataFrame(data)

        if df is None or df.empty:
            return []

        df = df.copy()
        try:
            self._publish_warmup_ml_features(symbol, df)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to publish warmup ML features for %s", symbol, exc_info=True)
        if "ts_event" in df.columns:
            timestamps = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        elif df.index.name == "ts_event":
            timestamps = pd.to_datetime(df.index, unit="ns", utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError("Historical payload missing ts_event column")

        timestamps = pd.DatetimeIndex(timestamps)

        column_aliases: dict[str, tuple[str, ...]] = {
            "open": ("open", "open_px", "open_price"),
            "high": ("high", "high_px", "high_price"),
            "low": ("low", "low_px", "low_price"),
            "close": ("close", "close_px", "px", "price"),
        }

        resolved_columns: dict[str, str] = {}
        for target, candidates in column_aliases.items():
            for candidate in candidates:
                if candidate in df.columns:
                    resolved_columns[target] = candidate
                    break

        volume_column = next(
            (col for col in ("volume", "size", "qty", "trade_sz") if col in df.columns),
            None,
        )

        if set(resolved_columns) == set(column_aliases):
            ohlcv = pd.DataFrame(
                {
                    key: pd.to_numeric(df[col], errors="coerce").to_numpy()
                    for key, col in resolved_columns.items()
                },
                index=timestamps,
            )
            if volume_column:
                ohlcv["volume"] = pd.to_numeric(df[volume_column], errors="coerce").to_numpy()
            else:
                ohlcv["volume"] = 0.0
            ohlcv = ohlcv.sort_index()
            ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")]
        else:
            price_column = resolved_columns.get("close") or next(
                (
                    col
                    for col in ("price", "px", "close", "trade_px")
                    if col in df.columns
                ),
                None,
            )
            if price_column is None:
                raise ValueError("Historical payload missing price column")

            prices = pd.to_numeric(df[price_column], errors="coerce").to_numpy()
            if volume_column:
                volumes = pd.to_numeric(df[volume_column], errors="coerce").to_numpy()
            else:
                volumes = [0.0] * len(df)

            frame = pd.DataFrame(
                {"price": prices, "volume": volumes}, index=timestamps
            )
            frame = frame.sort_index()
            frame = frame.dropna(subset=["price"])
            if frame.empty:
                return []

            ohlcv = frame.resample("1min").agg(
                {"price": ["first", "max", "min", "last"], "volume": "sum"}
            )

            if ohlcv.empty:
                return []

            ohlcv.columns = ["open", "high", "low", "close", "volume"]
            ohlcv = ohlcv.sort_index()

        ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")]

        compression_norm = _normalise_warmup_compression(compression)
        if compression_norm != "1min":
            aggregation = ohlcv.resample(compression_norm).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )
            if aggregation.empty:
                return []
            ohlcv = aggregation
        else:
            start = ohlcv.index.min()
            end = ohlcv.index.max()
            full_index = pd.date_range(start=start, end=end, freq="1min", tz=ohlcv.index.tz)
            ohlcv = ohlcv.reindex(full_index)

        bars: list[Bar] = []
        last_close: Optional[float] = None
        for timestamp, row in ohlcv.iterrows():
            open_price = row.open
            close_price = row.close
            high_price = row.high
            low_price = row.low
            volume = row.volume

            if pd.notna(open_price) and pd.notna(close_price):
                open_value = float(open_price)
                close_value = float(close_price)
                high_value = float(high_price) if pd.notna(high_price) else close_value
                low_value = float(low_price) if pd.notna(low_price) else close_value
                volume_value = float(volume) if pd.notna(volume) else 0.0
                last_close = close_value
            elif last_close is not None:
                open_value = high_value = low_value = close_value = float(last_close)
                volume_value = 0.0
            else:
                continue

            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=open_value,
                    high=high_value,
                    low=low_value,
                    close=close_value,
                    volume=volume_value,
                )
            )
        return bars

    def _publish_warmup_ml_features(self, symbol: str, frame: pd.DataFrame) -> None:
        tracker = getattr(self, "ml_feature_tracker", None)
        if tracker is None:
            return

        canonical = str(symbol or "").strip().upper()
        if not canonical:
            return

        collectors = getattr(self, "_ml_feature_collectors", {})
        collector = collectors.get(canonical)
        if collector is None:
            return

        requirements = getattr(self, "_ml_feature_requirements", {})
        required = requirements.get(canonical)
        if not required:
            return

        if frame is None or frame.empty:
            return

        warmed_map = getattr(self, "_ml_warmup_published", None)
        if warmed_map is None:
            warmed_map = {}
            setattr(self, "_ml_warmup_published", warmed_map)
        warmed = warmed_map.setdefault(canonical, set())

        try:
            snapshot = tracker.snapshot(canonical)
        except Exception:
            snapshot = {}

        lower_column_map: dict[str, str] = {}
        for column in frame.columns:
            if not isinstance(column, str):
                continue
            key = column.strip()
            if not key:
                continue
            lower_column_map.setdefault(key.lower(), column)

        for feature in required:
            key = str(feature or "").strip()
            if not key:
                continue
            if key in warmed:
                continue

            existing = snapshot.get(key)
            if existing is not None:
                warmed.add(key)
                continue

            if key in frame.columns:
                column_name: Optional[str] = key
            else:
                column_name = lower_column_map.get(key.lower())

            if column_name is None:
                continue

            series = frame[column_name]
            try:
                numeric_series = pd.to_numeric(series, errors="coerce")
            except Exception:
                continue

            numeric_series = numeric_series.dropna()
            if numeric_series.empty:
                continue

            value = float(numeric_series.iloc[-1])
            collector.record_feature(key, value)
            warmed.add(key)
            snapshot[key] = value

    def _set_strategies_warmup_mode(self, enabled: bool) -> None:
        """Enable or disable historical warmup mode on all strategies."""
        runstrats = getattr(self.cerebro, "runstrats", None)
        if not runstrats:
            return

        for strat_list in runstrats:
            if not strat_list:
                continue
            for strategy in strat_list:
                try:
                    setattr(strategy, "_in_historical_warmup", enabled)
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Failed to set warmup mode on strategy", exc_info=True)

        if enabled:
            logger.info("Historical warmup mode ENABLED on all strategies (fast mode)")
        else:
            logger.info("Historical warmup mode DISABLED on all strategies (full processing)")

    def _get_resampled_feed(self, symbol: str, suffix: str) -> Any:
        """Get a resampled feed (e.g., {symbol}_day or {symbol}_hour).

        Returns the resampled feed if found, otherwise None.
        """
        feed_name = f"{symbol}{suffix}"
        try:
            for data in self.cerebro.datas:
                if getattr(data, '_name', None) == feed_name:
                    return data
        except Exception as exc:
            logger.debug("Error accessing resampled feed %s: %s", feed_name, exc)
        return None

    def _finalize_indicator_warmup(self) -> None:
        if not self._historical_warmup_counts:
            logger.info("No historical data available for indicator warmup")
            return

        total_bars = sum(int(count) for count in self._historical_warmup_counts.values())
        logger.info(
            "Indicator warmup completed using %d bars across %d symbols",
            total_bars,
            len(self._historical_warmup_counts),
        )

        missing_symbols = set(self.symbols) - set(self._historical_warmup_counts)
        if missing_symbols:
            logger.warning(
                "Missing warmup data for symbols: %s", ", ".join(sorted(missing_symbols))
            )

        self._historical_warmup_counts.clear()

        tracker = getattr(self, "ml_feature_tracker", None)
        if tracker is not None:
            tracker.refresh_all(self.symbols)

    def _monitor_warmup_drain(self) -> None:
        """Background thread that monitors warmup backlog and disables warmup mode when drained.

        This runs after Cerebro starts and checks all feeds for warmup bars.
        Once all warmup bars are consumed, it disables strategy warmup mode.
        """
        # Give Cerebro a moment to start processing
        time.sleep(0.5)

        # Track all feeds that might have warmup bars
        tracked_feeds: list[Any] = []

        # Add base feeds
        for symbol in self.data_symbols:
            feed = self._data_feeds.get(symbol)
            if feed is not None and hasattr(feed, 'warmup_backlog_size'):
                tracked_feeds.append(feed)

        # Add resampled feeds
        try:
            for data in self.cerebro.datas:
                if hasattr(data, 'warmup_backlog_size'):
                    feed_name = getattr(data, '_name', None)
                    if feed_name and (feed_name.endswith('_day') or feed_name.endswith('_hour')):
                        tracked_feeds.append(data)
        except Exception as exc:
            logger.debug("Error checking resampled feeds for warmup monitoring: %s", exc)

        if not tracked_feeds:
            logger.debug("No feeds with warmup backlog found - skipping warmup monitoring")
            return

        logger.info("Monitoring %d feeds for warmup drain...", len(tracked_feeds))

        # Poll until all warmup bars are drained
        max_iterations = 200  # Safety limit
        iteration = 0

        while iteration < max_iterations and not self._stop_event.is_set():
            total_backlog = 0
            for feed in tracked_feeds:
                try:
                    backlog = feed.warmup_backlog_size()
                    total_backlog += backlog
                except Exception:
                    pass  # Ignore errors

            if total_backlog == 0:
                # All warmup bars have been consumed!
                logger.info("Warmup backlog drained - disabling warmup mode for full processing")
                self._set_strategies_warmup_mode(enabled=False)
                return

            iteration += 1
            time.sleep(0.1)  # Check every 100ms

        # Timeout or stop requested
        if not self._stop_event.is_set():
            logger.warning(
                "Warmup drain monitoring timed out after %d iterations - disabling warmup mode anyway",
                max_iterations
            )
            self._set_strategies_warmup_mode(enabled=False)

    def _on_strategy_feature_snapshot(self, strategy: NotifyingIbsStrategy) -> None:
        tracker = getattr(self, "ml_feature_tracker", None)
        if tracker is None:
            return

        symbol = getattr(getattr(strategy, "p", None), "symbol", None)
        if not isinstance(symbol, str):
            return
        canonical = symbol.strip().upper()
        if not canonical:
            return

        with self._ml_feature_lock:
            if canonical not in self._pending_ml_warmup:
                return
            seen = self._ml_features_seen.setdefault(canonical, set())

        features = getattr(getattr(strategy, "p", None), "ml_features", None)
        if not features:
            with self._ml_feature_lock:
                self._pending_ml_warmup.discard(canonical)
                self._ml_features_seen.pop(canonical, None)
            return

        collector = getattr(strategy, "ml_feature_collector", None)
        if collector is None:
            return

        try:
            snapshot = collector.snapshot  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to obtain ML snapshot for %s", canonical, exc_info=True)
            return

        if not isinstance(snapshot, Mapping):
            try:
                snapshot = dict(snapshot)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Snapshot for %s is not a mapping", canonical, exc_info=True)
                return

        updated = False
        required: list[str] = []
        for feature in features:
            key = str(feature or "").strip()
            if not key:
                continue
            required.append(key)
            if key in seen:
                continue
            value = snapshot.get(key)
            if value is None:
                continue
            seen.add(key)
            updated = True

        if updated:
            tracker.refresh(canonical)

        if not required:
            return

        with self._ml_feature_lock:
            if all(key in seen for key in required):
                self._pending_ml_warmup.discard(canonical)
                self._ml_features_seen.pop(canonical, None)

    def validate_policy_killswitch(self) -> bool:
        if getattr(self.config, "killswitch", False):
            logger.error("POLICY KILLSWITCH is enabled; aborting pre-flight validation")
            return False
        logger.info("Policy killswitch disabled; continuing pre-flight validation")
        return True

    def _traderspost_order_callback(self, strategy: NotifyingIbsStrategy, order: Any) -> None:
        if not self.traderspost_client:
            return
        payload = order_notification_to_message(strategy, order, self.queue_manager)
        if not payload:
            status = getattr(order, "status", None)
            if status != bt.Order.Completed:
                logger.debug(
                    "Skipping TradersPost order payload for %s: status %s not completed",
                    getattr(order, "ref", None),
                    status,
                )
            else:
                executed = getattr(order, "executed", None)
                if executed is None:
                    logger.debug(
                        "Skipping TradersPost order payload for %s: missing execution snapshot",
                        getattr(order, "ref", None),
                    )
                elif getattr(executed, "size", None) is None:
                    logger.debug(
                        "Skipping TradersPost order payload for %s: executed size unavailable",
                        getattr(order, "ref", None),
                    )
                else:
                    logger.debug(
                        "Skipping TradersPost order payload for %s: helper returned no payload",
                        getattr(order, "ref", None),
                    )
            return
        metadata = self._metadata_for_symbol(payload.get("symbol"))
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
        try:
            if getattr(self.config, "killswitch", False):
                logger.warning(
                    "🛑 KILLSWITCH: Would have posted order to TradersPost: %s %s size=%s",
                    payload.get("symbol"),
                    payload.get("side"),
                    payload.get("size"),
                )
                return
            self.traderspost_client.post_order(payload)
        except TradersPostError as exc:
            self._record_traderspost_result(
                "order", payload, success=False, error=str(exc)
            )
            logger.error(
                "TradersPost order post failed for %s %s size=%s: %s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
                exc,
            )
        except Exception:  # pragma: no cover - defensive guard
            self._record_traderspost_result(
                "order", payload, success=False, error="unexpected error"
            )
            logger.exception("Unexpected TradersPost order post failure")
        else:
            self._record_traderspost_result("order", payload, success=True)
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
        payload = trade_notification_to_message(strategy, trade, exit_snapshot, self.queue_manager)
        if not payload:
            if not getattr(trade, "isclosed", False):
                logger.debug(
                    "Skipping TradersPost trade payload for %s: trade not closed",
                    getattr(trade, "ref", None),
                )
            else:
                logger.debug(
                    "Skipping TradersPost trade payload for %s: helper returned no payload",
                    getattr(trade, "ref", None),
                )
            return
        metadata = self._metadata_for_symbol(payload.get("symbol"))
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
        try:
            self.traderspost_client.post_trade(payload)
        except TradersPostError as exc:
            self._record_traderspost_result(
                "trade", payload, success=False, error=str(exc)
            )
            logger.error(
                "TradersPost trade post failed for %s %s size=%s: %s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
                exc,
            )
        except Exception:  # pragma: no cover - defensive guard
            self._record_traderspost_result(
                "trade", payload, success=False, error="unexpected error"
            )
            logger.exception("Unexpected TradersPost trade post failure")
        else:
            self._record_traderspost_result("trade", payload, success=True)
            logger.info(
                "Posted trade event to TradersPost: %s %s size=%s",
                payload.get("symbol"),
                payload.get("side"),
                payload.get("size"),
            )

    def _discord_trade_callback(
        self,
        strategy: NotifyingIbsStrategy,
        trade: Any,
        exit_snapshot: Optional[Mapping[str, Any]],
    ) -> None:
        """Send Discord notification for trade events."""
        if not self.discord_notifier:
            return

        if not getattr(trade, "isclosed", False):
            # Trade opened - send entry notification
            symbol = getattr(strategy, "p", None) and strategy.p.symbol
            if not symbol:
                return

            try:
                from datetime import datetime
                from strategy.contract_specs import point_value

                # Get entry details
                entry_price = float(trade.price) if hasattr(trade, "price") else 0.0
                size = abs(float(strategy.p.size)) if hasattr(strategy.p, "size") else 1.0
                side = "long"  # IBS strategy is long-only

                # Get IBS and ML score from strategy state
                ibs_val = None
                ml_score = None
                if hasattr(strategy, "_latest_ibs_value"):
                    ibs_val = strategy._latest_ibs_value
                if hasattr(strategy, "_latest_ml_score"):
                    ml_score = strategy._latest_ml_score

                self.discord_notifier.send_trade_entry(
                    symbol=symbol,
                    side=side,
                    price=entry_price,
                    size=size,
                    ibs=ibs_val,
                    ml_score=ml_score,
                )
            except Exception:
                logger.exception("Failed to send Discord entry notification")

        else:
            # Trade closed - send exit notification
            symbol = getattr(strategy, "p", None) and strategy.p.symbol
            if not symbol:
                return

            try:
                from datetime import datetime
                from strategy.contract_specs import point_value, CONTRACT_SPECS
                from config import COMMISSION_PER_SIDE

                # Get trade details
                entry_price = float(trade.price) if hasattr(trade, "price") else 0.0
                exit_price = float(trade.price + trade.pnl) if hasattr(trade, "pnl") else entry_price
                size = abs(float(strategy.p.size)) if hasattr(strategy.p, "size") else 1.0
                side = "long"

                # Calculate P&L
                pv = point_value(symbol)
                commission_usd = 2 * COMMISSION_PER_SIDE * size
                pnl_usd = trade.pnl * pv - commission_usd if hasattr(trade, "pnl") else 0.0
                pnl_percent = (trade.pnl / entry_price * 100) if entry_price > 0 and hasattr(trade, "pnl") else 0.0

                # Get exit reason and IBS from exit snapshot
                exit_reason = "Unknown"
                ibs_val = None
                if exit_snapshot:
                    exit_reason = exit_snapshot.get("exit_reason", "Unknown")
                    ibs_val = exit_snapshot.get("ibs_value")

                # Calculate duration
                duration_hours = None
                if hasattr(trade, "dtopen") and hasattr(trade, "dtclose"):
                    import backtrader as bt
                    entry_time = bt.num2date(trade.dtopen)
                    exit_time = bt.num2date(trade.dtclose)
                    duration_hours = (exit_time - entry_time).total_seconds() / 3600

                self.discord_notifier.send_trade_exit(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    pnl=pnl_usd,
                    pnl_percent=pnl_percent,
                    exit_reason=exit_reason,
                    ibs=ibs_val,
                    duration_hours=duration_hours,
                )
            except Exception:
                logger.exception("Failed to send Discord exit notification")

    def send_daily_summary(self) -> bool:
        """Send a daily performance summary to Discord.

        Returns:
            True if summary was sent successfully
        """
        if not self.discord_notifier:
            logger.debug("Discord notifier not available for daily summary")
            return False

        try:
            from datetime import datetime, timedelta
            from utils.trades_db import TradesDB

            db = TradesDB()

            # Get today's trades
            today = datetime.now().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())

            trades = db.get_trades_between(start_of_day, end_of_day)

            if not trades:
                logger.info("No trades today, skipping daily summary")
                return False

            # Calculate summary stats
            total_pnl = sum(t["pnl"] for t in trades)
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] < 0]
            win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
            best_trade = max((t["pnl"] for t in trades), default=0.0)
            worst_trade = min((t["pnl"] for t in trades), default=0.0)
            symbols = list(set(t["symbol"] for t in trades))

            # Calculate profit factor
            gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0.0
            gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

            logger.info(
                f"Daily summary: {len(trades)} trades, ${total_pnl:.2f} P&L, "
                f"{win_rate:.1f}% win rate, {profit_factor:.2f}x PF"
            )

            return self.discord_notifier.send_daily_summary(
                total_pnl=total_pnl,
                num_trades=len(trades),
                win_rate=win_rate,
                best_trade=best_trade,
                worst_trade=worst_trade,
                symbols_traded=symbols,
                date=datetime.now(),
                profit_factor=profit_factor,
            )
        except Exception:
            logger.exception("Failed to send daily summary")
            return False

    def _required_reference_feed_names(self) -> set[str]:
        feeds: set[str] = set()
        for feed_name in REQUIRED_REFERENCE_FEEDS:
            feed = str(feed_name or "").strip()
            if feed:
                feeds.add(feed)
        for symbol in self.contract_map.reference_symbols():
            base = str(symbol or "").strip().upper()
            if not base:
                continue
            feeds.add(f"{base}_day")
        for symbol in self.symbols:
            pair_symbol = PAIR_MAP.get(symbol)
            if not pair_symbol:
                continue
            base = str(pair_symbol).strip().upper()
            if not base:
                continue
            feeds.add(f"{base}_day")
            feeds.add(f"{base}_hour")
        return feeds

    def _run_cerebro(self) -> None:
        logger.info("Starting Backtrader runtime for symbols: %s", ", ".join(self.symbols))
        # Wait for initial data to begin streaming so Cerebro has initial bars
        logger.info("Waiting for initial data from Databento feeds...")

        if not self._wait_for_initial_data():
            self._stop_event.set()
            logger.info("Cerebro runtime aborted: initial data unavailable")
            return

        # Start background thread to monitor warmup drain and disable warmup mode
        # once all warmup bars have been consumed by Cerebro
        warmup_monitor = threading.Thread(
            target=self._monitor_warmup_drain,
            name="warmup-monitor",
            daemon=True
        )
        warmup_monitor.start()

        try:
            self.cerebro.run(runonce=False, stdstats=False, maxcpus=1)
        except Exception as exc:  # pragma: no cover - unexpected runtime error
            logger.exception("Cerebro execution failed: %s", exc)
        finally:
            self._stop_event.set()
            logger.info("Cerebro runtime stopped")

    def _wait_for_initial_data(self, max_wait_seconds: int = 60) -> bool:
        required_queues: set[str] = {str(symbol) for symbol in self.data_symbols}
        for feed_name in self._required_reference_feed_names():
            feed = str(feed_name or "").strip()
            if not feed:
                continue
            if feed.endswith("_day"):
                feed = feed[: -len("_day")]
            elif feed.endswith("_hour"):
                feed = feed[: -len("_hour")]
            if feed:
                required_queues.add(feed)

        start_time = time.monotonic()
        countdown_start: Optional[float] = None
        live_seen = False

        def _has_warmup(symbol: str) -> bool:
            feed = self._data_feeds.get(symbol)
            if not isinstance(feed, DatabentoLiveData):
                return False
            try:
                backlog = feed.warmup_backlog_size()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to query warmup backlog size for %s", symbol, exc_info=True)
                return False
            return backlog > 0

        while True:
            warmup_pending = False
            missing_queues: set[str] = set()

            for queue_name in required_queues:
                queue_obj = self.queue_manager.get_queue(queue_name)
                if queue_obj.qsize() > 0:
                    live_seen = True
                    continue

                if _has_warmup(queue_name):
                    warmup_pending = True
                    continue

                missing_queues.add(queue_name)

            if not missing_queues:
                elapsed = int(time.monotonic() - start_time) + 1
                logger.info("Initial data received after %s seconds", max(elapsed, 1))
                return True

            now = time.monotonic()

            if warmup_pending and not live_seen:
                countdown_start = None
            elif countdown_start is None:
                countdown_start = now

            if countdown_start is not None and now - countdown_start >= max_wait_seconds:
                logger.error(
                    "Timeout waiting for initial data after %s seconds; missing queues: %s",
                    max_wait_seconds,
                    ", ".join(sorted(missing_queues)),
                )
                return False

            time.sleep(1)

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

        # Run historical warmup BEFORE starting Cerebro to ensure bars are loaded and drained
        # Otherwise Cerebro starts consuming bars before drain completes
        self._run_historical_warmup()

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

        metadata = self.contract_map.traderspost_metadata(sym)

        # Add selected contract info if available
        selection = self._selected_contracts.get(sym)
        if selection:
            metadata["selected_contract"] = selection.contract_symbol
            metadata["contract_open_interest"] = selection.open_interest
            metadata["contract_volume"] = selection.volume
            if selection.selected_at:
                metadata["contract_selected_at"] = selection.selected_at.isoformat()

        return metadata

    def _select_contracts(self) -> None:
        """Select highest OI contracts for trading symbols."""
        if not self.contract_selector:
            logger.info("Contract selector not available; using default contract selection")
            return

        # Get trading symbols (from portfolio instruments if specified)
        trading_symbols = list(self.config.portfolio_instruments or self.symbols)

        if not trading_symbols:
            logger.warning("No trading symbols to select contracts for")
            return

        logger.info("Selecting highest OI contracts for %d symbols...", len(trading_symbols))

        try:
            selections = self.contract_selector.select_contracts(trading_symbols)
            self._selected_contracts = selections

            for sym, sel in selections.items():
                logger.info(
                    "  %s -> %s (OI=%d, Vol=%d)",
                    sym, sel.contract_symbol, sel.open_interest, sel.volume
                )

        except Exception as e:
            logger.error("Failed to select contracts: %s", e)

    def get_selected_contract(self, root_symbol: str) -> Optional[str]:
        """Get the selected contract symbol for a root."""
        selection = self._selected_contracts.get(root_symbol.strip().upper())
        return selection.contract_symbol if selection else None

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
            "updated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "symbols": list(self.symbols),
        }
        heartbeat_details = self._collect_heartbeat_details()
        if details:
            try:
                heartbeat_details.update(dict(details))
            except Exception:  # pragma: no cover - defensive guard
                heartbeat_details["message"] = str(details)
        if heartbeat_details:
            payload["details"] = heartbeat_details

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
            self._last_heartbeat_update = now
        except Exception:  # pragma: no cover - best effort monitoring
            logger.exception("Failed to write heartbeat file at %s", path)

    def _collect_heartbeat_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {"preflight": dict(self._preflight_summary)}

        databento_snapshot: dict[str, Any] = {
            "queue_fanout": {},
            "subscribers": [],
        }
        try:
            databento_snapshot["queue_fanout"] = self.queue_manager.snapshot()
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to gather queue snapshot")
        for subscriber in self.subscribers:
            try:
                databento_snapshot["subscribers"].append(subscriber.status_snapshot())
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Failed to gather subscriber snapshot")
        details["databento"] = databento_snapshot

        if self.traderspost_client:
            details["traderspost"] = copy.deepcopy(self._traderspost_status)

        tracker = getattr(self, "ml_feature_tracker", None)
        if tracker is not None:
            details["ml_features"] = tracker.readiness_report()

        return details

    @staticmethod
    def _now_iso() -> str:
        return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _record_traderspost_result(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        event = {
            "kind": kind,
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "size": payload.get("size"),
        }
        timestamp = self._now_iso()
        if success:
            self._traderspost_status["last_success"] = {"at": timestamp, **event}
            self._traderspost_status["last_error"] = None
        else:
            self._traderspost_status["last_error"] = {
                "at": timestamp,
                "message": error,
                **event,
            }

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------

    def validate_ml_models(self) -> bool:
        # Only validate trading symbols, not all feature symbols
        trading_symbols = list(self.config.portfolio_instruments or self.symbols)
        total = len(trading_symbols)
        logger.info("Validating ML models for %s symbols...", total)
        if not total:
            logger.info("✓ ML models validated for %s symbols", total)
            return True

        failed = False
        for symbol in trading_symbols:
            # Use appropriate loader based on strategy type
            if self.config.use_factory_strategies:
                strategy_name = self.config.factory_strategy_mapping.get(
                    symbol.upper(),
                    VALIDATED_STRATEGIES.get(symbol.upper(), {}).get('strategy', 'AvgHLRangeIBS')
                )
                try:
                    bundle = load_factory_model_bundle(
                        symbol,
                        strategy_name=strategy_name,
                        base_dir=self.config.models_path
                    )
                    bundle_kwargs = factory_strategy_kwargs_from_bundle(bundle)
                except Exception as exc:
                    logger.error("❌ %s: Failed to load factory model bundle - %s", symbol, exc)
                    failed = True
                    continue
            else:
                try:
                    bundle = load_model_bundle(symbol, base_dir=self.config.models_path)
                    bundle_kwargs = strategy_kwargs_from_bundle(bundle)
                except Exception as exc:
                    logger.error("❌ %s: Failed to load model bundle - %s", symbol, exc)
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

        payload = {"type": "health_check", "timestamp": dt.datetime.utcnow().isoformat()}
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
            reference_symbols = {sym.upper() for sym in self.contract_map.reference_symbols()}
        except Exception as exc:
            logger.error("❌ Failed to load contract map: %s", exc)
            return False

        failed = False
        available_symbols = {sym.upper() for sym in self.data_symbols}

        for feed_name in sorted(self._required_reference_feed_names()):
            feed = str(feed_name or "").strip()
            if not feed:
                continue
            base = feed
            if base.endswith("_day"):
                base = base[: -len("_day")]
            elif base.endswith("_hour"):
                base = base[: -len("_hour")]
            base_upper = base.upper()
            if base_upper not in reference_symbols and base_upper not in available_symbols:
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
            pair_upper = pair_symbol.upper()
            if (
                pair_upper not in reference_symbols
                and pair_upper not in available_symbols
            ):
                logger.error(
                    "❌ Pair data feed missing for %s (expected %s_day/%s_hour)",
                    symbol,
                    pair_upper,
                    pair_upper,
                )
                failed = True
            elif pair_symbol not in self.symbols:
                logger.warning(
                    "⚠️ Pair symbol %s not traded directly; using reference feed only",
                    pair_symbol,
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

        for feed_name in self._required_reference_feed_names():
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

        self._preflight_summary = {
            "status": "running",
            "started_at": self._now_iso(),
        }

        if not preflight_cfg.enabled:
            logger.info("")
            logger.info("Pre-flight validation disabled via configuration; skipping checks.")
            logger.info("")
            logger.info(separator)
            logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
            logger.info(separator)
            self._preflight_summary = {
                "status": "skipped",
                "checked_at": self._now_iso(),
                "failed_checks": [],
            }
            return True

        failed_checks: list[str] = []
        checks: list[tuple[str, Callable[[], bool]]] = [
            ("Policy Killswitch", self.validate_policy_killswitch)
        ]

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
            self._preflight_summary = {
                "status": "failed",
                "checked_at": self._now_iso(),
                "failed_checks": failed_checks,
            }
            return False

        logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
        logger.info(separator)
        self._preflight_summary = {
            "status": "passed",
            "checked_at": self._now_iso(),
            "failed_checks": [],
        }
        return True

    def run(self) -> None:
        """Synchronous helper that drives :meth:`run_async`."""

        asyncio.run(self.run_async())

    def add_loop_callback(self, callback: Callable[[], None]) -> None:
        self._loop_callbacks.append(callback)
