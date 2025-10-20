"""Helpers for tracking ML feature readiness per symbol."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence


def _normalise_features(features: Sequence[str] | None) -> tuple[str, ...]:
    """Return a deterministic, de-duplicated tuple of feature names."""

    if not features:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for feature in features:
        key = str(feature or "").strip()
        if not key or key in seen:
            continue
        ordered.append(key)
        seen.add(key)
    return tuple(ordered)


@dataclass
class _SymbolSnapshot:
    """Container holding the latest feature snapshot for a symbol."""

    features: Dict[str, float | None]
    required: tuple[str, ...]
    ready: bool = False

    def update(self, key: str, value: float | None) -> None:
        self.features[key] = value
        self.ready = self._compute_ready()

    def refresh(self) -> None:
        self.ready = self._compute_ready()

    def _compute_ready(self) -> bool:
        if not self.required:
            # No required features – treat the bundle as immediately ready.
            return True
        if not self.features:
            return False
        for feature in self.required:
            if self.features.get(feature) is None:
                return False
        return True


class MlFeatureTracker:
    """Track per-symbol ML feature readiness for loaded bundles."""

    class Collector:
        """Adapter passed into strategies to collect feature updates."""

        def __init__(self, tracker: "MlFeatureTracker", symbol: str) -> None:
            self._tracker = tracker
            self._symbol = symbol

        # Backtrader strategies call `record_feature`/`update_feature` when
        # publishing ML inputs.  Additional helpers mirror mapping semantics so
        # fallback code paths remain compatible.
        def record_feature(self, key: str, value) -> None:  # type: ignore[override]
            self._tracker.update_feature(self._symbol, key, value)

        def update_feature(self, key: str, value) -> None:  # pragma: no cover - alias
            self.record_feature(key, value)

        def publish(self, key: str, value) -> None:  # pragma: no cover - alias
            self.record_feature(key, value)

        def update(self, mapping: Mapping[str, float | None]) -> None:  # pragma: no cover - alias
            for key, value in mapping.items():
                self.record_feature(key, value)

        def __setitem__(self, key: str, value) -> None:  # pragma: no cover - alias
            self.record_feature(key, value)

        @property
        def snapshot(self) -> Mapping[str, float | None]:
            return self._tracker.snapshot(self._symbol)

    def __init__(self) -> None:
        self._snapshots: Dict[str, _SymbolSnapshot] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_bundle(self, symbol: str, features: Sequence[str] | None) -> "Collector":
        """Register ``symbol`` with its required ``features``.

        Returns a collector object that strategies use to publish feature
        updates.  The collector keeps the latest feature snapshot in sync with
        readiness tracking.
        """

        normalised = _normalise_features(features)
        with self._lock:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None:
                snapshot = _SymbolSnapshot(features={}, required=normalised)
                self._snapshots[symbol] = snapshot
            else:
                snapshot.required = normalised
                snapshot.refresh()
        return MlFeatureTracker.Collector(self, symbol)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------
    def update_feature(self, symbol: str, key: str, value) -> None:
        """Record the latest ``value`` for ``key`` under ``symbol``."""

        if not key:
            return

        numeric: float | None
        if value is None:
            numeric = None
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            else:
                if math.isnan(numeric):
                    numeric = None

        with self._lock:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None:
                snapshot = _SymbolSnapshot(features={}, required=_normalise_features(None))
                self._snapshots[symbol] = snapshot
            snapshot.update(key, numeric)

    def refresh(self, symbol: str) -> None:
        """Re-evaluate readiness for ``symbol`` using the cached snapshot."""

        with self._lock:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None:
                return
            snapshot.refresh()

    def refresh_all(self, symbols: Iterable[str] | None = None) -> None:
        """Recompute readiness for every tracked symbol."""

        with self._lock:
            if symbols is None:
                values = list(self._snapshots.values())
            else:
                values = [self._snapshots.get(sym) for sym in symbols]
            for snapshot in values:
                if snapshot is not None:
                    snapshot.refresh()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def is_ready(self, symbol: str) -> bool:
        with self._lock:
            snapshot = self._snapshots.get(symbol)
            return bool(snapshot.ready) if snapshot else False

    def snapshot(self, symbol: str) -> Mapping[str, float | None]:
        with self._lock:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None:
                return {}
            return dict(snapshot.features)

    def readiness_report(self) -> Dict[str, Mapping[str, object]]:
        """Return a serialisable snapshot for monitoring/heartbeat payloads."""

        with self._lock:
            report: Dict[str, Mapping[str, object]] = {}
            for symbol, snapshot in self._snapshots.items():
                report[symbol] = {
                    "ready": snapshot.ready,
                    "required_features": snapshot.required,
                    "feature_count": len(snapshot.features),
                }
            return report

