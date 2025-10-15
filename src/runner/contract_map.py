"""Utilities for loading and validating Databento contract metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

__all__ = [
    "ContractMapError",
    "DatabentoFeed",
    "RollRule",
    "ContractRoot",
    "ContractMap",
    "load_contract_map",
]


class ContractMapError(ValueError):
    """Raised when the contract map payload fails validation."""


@dataclass(frozen=True)
class DatabentoFeed:
    """Metadata describing how to subscribe to a Databento dataset."""

    dataset: str
    feed_symbol: Optional[str] = None
    product_id: Optional[str] = None


@dataclass(frozen=True)
class RollRule:
    """Rolling behaviour for a futures root."""

    stype_in: str = "product_id"
    stype_out: Optional[str] = None
    roll_days: Optional[int] = None
    offset_days: Optional[int] = None


@dataclass(frozen=True)
class DatabentoSubscription:
    """Resolved subscription information for a futures root."""

    dataset: str
    stype_in: str
    codes: Tuple[str, ...]


@dataclass(frozen=True)
class ContractRoot:
    """Representation of a tradable futures root."""

    symbol: str
    tradovate_symbol: Optional[str]
    tradovate_description: Optional[str]
    optimized: bool
    databento: DatabentoFeed
    roll: RollRule

    @property
    def feed_symbol(self) -> str:
        """Return the preferred feed symbol for live data."""

        return self.databento.feed_symbol or self.symbol

    @property
    def product_id(self) -> Optional[str]:
        """Return the Databento product id if one was provided."""

        value = self.databento.product_id
        return value.strip() if isinstance(value, str) and value else None

    def subscription(self) -> DatabentoSubscription:
        """Return the dataset, stype, and subscription codes for this root."""

        codes: list[str] = []
        product_id = self.product_id
        if product_id:
            codes.append(product_id)
        # Only add feed_symbol if explicitly set (not falling back to the root symbol)
        if self.databento.feed_symbol and self.databento.feed_symbol not in codes:
            codes.append(self.databento.feed_symbol)
        if not codes:
            # Fallback to the internal symbol only when no Databento mapping exists
            codes.append(self.symbol)
        deduped: list[str] = []
        seen: set[str] = set()
        for code in codes:
            norm = code.strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)
        return DatabentoSubscription(
            dataset=self.databento.dataset,
            stype_in=self.roll.stype_in or "parent",
            codes=tuple(deduped),
        )

    def traderspost_metadata(self) -> dict[str, Any]:
        """Return additional attributes included in TradersPost payloads."""

        payload: dict[str, Any] = {
            "tradovate_symbol": self.tradovate_symbol,
            "tradovate_description": self.tradovate_description,
            "databento_dataset": self.databento.dataset,
            "databento_feed_symbol": self.feed_symbol,
            "databento_product_id": self.product_id,
            "optimized": self.optimized,
            "roll_stype_in": self.roll.stype_in,
        }
        return {key: value for key, value in payload.items() if value is not None}


def _subscription_from_feed(symbol: str, feed: DatabentoFeed) -> DatabentoSubscription:
    """Return a :class:`DatabentoSubscription` built from a reference feed."""

    codes: list[str] = []
    if feed.product_id:
        codes.append(feed.product_id)
    if feed.feed_symbol:
        codes.append(feed.feed_symbol)
    if not codes:
        # Fallback to the internal symbol only when no Databento mapping exists
        codes.append(symbol)

    deduped: list[str] = []
    seen: set[str] = set()
    for code in codes:
        norm = code.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)

    return DatabentoSubscription(
        dataset=feed.dataset,
        stype_in="parent",
        codes=tuple(deduped),
    )


class ContractMap:
    """Container exposing lookup helpers for contract metadata."""

    def __init__(
        self,
        contracts: Mapping[str, ContractRoot],
        reference_feeds: Mapping[str, DatabentoFeed],
    ) -> None:
        self._contracts: Dict[str, ContractRoot] = {
            symbol.upper(): value for symbol, value in contracts.items()
        }
        self._reference_feeds: Dict[str, DatabentoFeed] = {
            symbol.upper(): value for symbol, value in reference_feeds.items()
        }

    # ------------------------------------------------------------------
    # Container helpers
    # ------------------------------------------------------------------
    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        return symbol.upper() in self._contracts

    def __iter__(self):  # pragma: no cover - convenience iterator
        return iter(self._contracts)

    def keys(self) -> Iterable[str]:  # pragma: no cover - convenience iterator
        return self._contracts.keys()

    # ------------------------------------------------------------------
    # Lookup utilities
    # ------------------------------------------------------------------
    def active_contract(self, symbol: str) -> ContractRoot:
        """Return the active contract for ``symbol``."""

        try:
            return self._contracts[symbol.upper()]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown contract root: {symbol}") from exc

    def reference_feed(self, symbol: str) -> Optional[DatabentoFeed]:
        """Return the reference Databento feed for ``symbol`` if available."""

        return self._reference_feeds.get(symbol.upper())

    def symbols(self) -> Tuple[str, ...]:
        """Return all known tradable roots."""

        return tuple(sorted(self._contracts.keys()))

    def reference_symbols(self) -> Tuple[str, ...]:
        """Return all reference feed symbols."""

        return tuple(sorted(self._reference_feeds.keys()))

    def product_to_root(self, symbols: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Return a mapping of subscription codes to their root symbol."""

        mapping: Dict[str, str] = {}
        for symbol, subscription in self._iter_subscriptions(symbols):
            for code in subscription.codes:
                mapping[code] = symbol
        return mapping

    def reference_product_to_root(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Dict[str, str]:
        """Return a mapping of reference subscription codes to their root symbol."""

        mapping: Dict[str, str] = {}
        for symbol, subscription in self._iter_reference_subscriptions(symbols):
            for code in subscription.codes:
                mapping[code] = symbol
        return mapping

    def dataset_groups(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Dict[Tuple[str, str], Tuple[str, ...]]:
        """Group subscription codes by dataset and ``stype_in``."""

        grouped: Dict[Tuple[str, str], set[str]] = {}
        for _, subscription in self._iter_subscriptions(symbols):
            key = (subscription.dataset, subscription.stype_in)
            grouped.setdefault(key, set()).update(subscription.codes)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}

    def reference_dataset_groups(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Dict[Tuple[str, str], Tuple[str, ...]]:
        """Group reference subscription codes by dataset and ``stype_in``."""

        grouped: Dict[Tuple[str, str], set[str]] = {}
        for _, subscription in self._iter_reference_subscriptions(symbols):
            key = (subscription.dataset, subscription.stype_in)
            grouped.setdefault(key, set()).update(subscription.codes)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}

    def traderspost_metadata(self, symbol: str) -> dict[str, Any]:
        """Return supplemental metadata for the TradersPost payload builder."""

        return self.active_contract(symbol).traderspost_metadata()

    def _iter_subscriptions(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterable[tuple[str, DatabentoSubscription]]:
        if symbols is None:
            items = self._contracts.items()
        else:
            items = ((sym.upper(), self.active_contract(sym)) for sym in symbols)
        for symbol, contract in items:
            yield symbol, contract.subscription()

    def _iter_reference_subscriptions(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterable[tuple[str, DatabentoSubscription]]:
        if symbols is None:
            items = self._reference_feeds.items()
        else:
            pairs: list[tuple[str, DatabentoFeed]] = []
            for sym in symbols:
                feed = self.reference_feed(sym)
                if not feed:
                    continue
                pairs.append((sym.upper(), feed))
            items = pairs
        for symbol, feed in items:
            if not feed:
                continue
            yield symbol, _subscription_from_feed(symbol, feed)


def load_contract_map(path: str | Path) -> ContractMap:
    """Load and validate contract metadata from ``path``."""

    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract map file does not exist: {contract_path}")
    raw_payload = contract_path.read_text(encoding="utf-8")
    payload = _parse_payload(raw_payload, contract_path)

    contracts_payload = payload.get("contracts")
    if contracts_payload is None:
        raise ContractMapError("Contract map payload missing 'contracts' section")
    if not isinstance(contracts_payload, Sequence):
        raise ContractMapError("'contracts' section must be a sequence")

    contracts: Dict[str, ContractRoot] = {}
    for entry in contracts_payload:
        root = _parse_contract_entry(entry)
        if root.symbol in contracts:
            raise ContractMapError(f"Duplicate contract root encountered: {root.symbol}")
        contracts[root.symbol] = root

    reference_payload = payload.get("reference_feeds", []) or []
    if not isinstance(reference_payload, Sequence):
        raise ContractMapError("'reference_feeds' section must be a sequence if provided")

    reference_feeds: Dict[str, DatabentoFeed] = {}
    for entry in reference_payload:
        symbol, feed = _parse_reference_entry(entry)
        reference_feeds[symbol] = feed

    return ContractMap(contracts=contracts, reference_feeds=reference_feeds)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_payload(text: str, path: Path) -> Mapping[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass

    if yaml is None:
        raise ContractMapError(
            "PyYAML is not installed and JSON decoding failed for contract map"
        )
    try:
        parsed_yaml = yaml.safe_load(text)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - surface parsing errors
        raise ContractMapError(f"Unable to parse contract map {path}: {exc}") from exc
    if not isinstance(parsed_yaml, Mapping):
        raise ContractMapError("Contract map root must be a mapping")
    return parsed_yaml


def _parse_contract_entry(payload: Any) -> ContractRoot:
    if not isinstance(payload, Mapping):
        raise ContractMapError("Each contract entry must be a mapping")

    symbol_raw = payload.get("symbol")
    symbol = _normalize_symbol(symbol_raw)
    if not symbol:
        raise ContractMapError("Contract entry missing 'symbol'")

    tradovate_symbol = _normalize_symbol(payload.get("tradovate_symbol"))
    tradovate_description = _normalize_str(payload.get("tradovate_description"))
    optimized = bool(payload.get("optimized", False))

    databento_payload = payload.get("databento")
    if not isinstance(databento_payload, Mapping):
        raise ContractMapError(f"Contract {symbol} missing 'databento' mapping")

    dataset = _normalize_str(databento_payload.get("dataset"))
    feed_symbol = _normalize_symbol(databento_payload.get("feed_symbol") or symbol)
    product_id_raw = databento_payload.get("product_id")
    product_id = _normalize_str(product_id_raw) if product_id_raw is not None else None
    if feed_symbol is None and product_id:
        feed_symbol = product_id

    if not dataset:
        raise ContractMapError(f"Contract {symbol} missing Databento dataset")
    if not feed_symbol:
        raise ContractMapError(f"Contract {symbol} missing Databento feed_symbol")

    roll_payload = payload.get("roll") or {}
    if roll_payload and not isinstance(roll_payload, Mapping):
        raise ContractMapError(f"Contract {symbol} roll rules must be a mapping if provided")

    stype_in = _normalize_str((roll_payload or {}).get("stype_in")) or "product_id"
    stype_out = _normalize_str((roll_payload or {}).get("stype_out"))
    roll_days = _normalize_int((roll_payload or {}).get("roll_days"))
    offset_days = _normalize_int((roll_payload or {}).get("offset_days"))

    databento = DatabentoFeed(dataset=dataset, feed_symbol=feed_symbol, product_id=product_id)
    roll = RollRule(stype_in=stype_in, stype_out=stype_out, roll_days=roll_days, offset_days=offset_days)

    return ContractRoot(
        symbol=symbol,
        tradovate_symbol=tradovate_symbol,
        tradovate_description=tradovate_description,
        optimized=optimized,
        databento=databento,
        roll=roll,
    )


def _parse_reference_entry(payload: Any) -> tuple[str, DatabentoFeed]:
    if not isinstance(payload, Mapping):
        raise ContractMapError("Each reference feed entry must be a mapping")

    databento_payload = payload.get("databento")
    if not isinstance(databento_payload, Mapping):
        raise ContractMapError("Reference feed missing 'databento' mapping")

    symbol = _normalize_symbol(payload.get("symbol") or databento_payload.get("feed_symbol"))
    dataset = _normalize_str(databento_payload.get("dataset"))
    feed_symbol_raw = databento_payload.get("feed_symbol")
    feed_symbol = _normalize_symbol(feed_symbol_raw) if feed_symbol_raw is not None else None
    product_id_raw = databento_payload.get("product_id")
    product_id = _normalize_str(product_id_raw) if product_id_raw is not None else None
    if feed_symbol is None and product_id:
        feed_symbol = product_id

    if not symbol:
        raise ContractMapError("Reference feed missing symbol")
    if not dataset:
        raise ContractMapError(f"Reference feed {symbol} missing Databento dataset")
    return symbol, DatabentoFeed(dataset=dataset, feed_symbol=feed_symbol, product_id=product_id)


def _normalize_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.upper() if stripped else None
    return _normalize_str(value)


def _normalize_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return str(value)


def _normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ContractMapError(f"Invalid integer value: {value}")
