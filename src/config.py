"""Central configuration shared by backtests and live workers.

This module consolidates the runtime constants consumed by
``strategy.ibs_strategy.IbsStrategy`` so both the research notebooks and the
production runner rely on the same defaults.  Every value can be overridden via
environment variables in order to keep live deployments and local experiments
in sync.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping

__all__ = [
    "COMMISSION_PER_SIDE",
    "PAIR_MAP",
    "DEFAULT_COMMISSION_PER_SIDE",
    "DEFAULT_PAIR_MAP",
    "REQUIRED_REFERENCE_FEEDS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Documented defaults
# ---------------------------------------------------------------------------

#: Default commission charged per side of a trade (per contract).  The value is
#: derived from the deployment notes for the Tradovate micro futures account and
#: covers exchange, clearing, and broker fees.  Live workers can override this
#: via ``PINE_COMMISSION_PER_SIDE`` or ``COMMISSION_PER_SIDE``.
DEFAULT_COMMISSION_PER_SIDE: float = 1.00

#: Canonical symbol pairings used by the IBS strategy when constructing
#: secondary filters (pair IBS, RSI-2 cross instrument, etc.).  The mapping is
#: mirrored from the production configuration so that offline backtests operate
#: with the same assumptions as the live system.
DEFAULT_PAIR_MAP: Mapping[str, str] = MappingProxyType(
    {
        "ES": "NQ",
        "NQ": "ES",
        "RTY": "YM",
        "YM": "RTY",
        "GC": "SI",
        "SI": "GC",
        "HG": "SI",
        "CL": "NG",
        "NG": "CL",
        "6A": "6E",
        "6B": "6E",
        "6E": "6A",
    }
)

#: Reference data feeds that must be present for ``IbsStrategy`` to initialise.
#: Both the research notebooks and live workers should expose these names via
#: Backtrader's ``adddata(..., name="<SYMBOL>_day")`` helper.
REQUIRED_REFERENCE_FEEDS: tuple[str, ...] = ("TLT_day",)


def _get_env(name: str) -> str | None:
    """Return the highest precedence environment variable for ``name``.

    Production deployments use the ``PINE_<NAME>`` convention while notebooks
    often opt for the shorter ``<NAME>`` variant.  The helper checks both so
    callers can override values without editing code.
    """

    prefixed = f"PINE_{name}"
    return os.getenv(prefixed) or os.getenv(name)


def _load_commission(default: float) -> float:
    raw = _get_env("COMMISSION_PER_SIDE")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid COMMISSION_PER_SIDE override %r; using default %.2f", raw, default
        )
        return default
    if value < 0:
        logger.warning(
            "Negative COMMISSION_PER_SIDE override %r; using default %.2f", raw, default
        )
        return default
    return value


def _normalise_pair_map(values: Mapping[str, str]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for key, value in values.items():
        if key is None or value is None:
            continue
        left = str(key).strip().upper()
        right = str(value).strip().upper()
        if not left or not right:
            continue
        pairs[left] = right
    return pairs


def _parse_pair_map(raw: str) -> Dict[str, str]:
    """Parse a pair-map override from JSON or ``key:value`` pairs."""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, Mapping):
        parsed = _normalise_pair_map(data)
        if parsed:
            return parsed
    pairs: Dict[str, str] = {}
    for chunk in raw.split(","):
        if not chunk or ":" not in chunk:
            continue
        left, right = chunk.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs[left] = right
    return _normalise_pair_map(pairs)


def _load_pair_map(default: Mapping[str, str]) -> Mapping[str, str]:
    path_override = _get_env("PAIR_MAP_PATH")
    if path_override:
        path = Path(path_override).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("PAIR_MAP_PATH %s not found; using default", path)
        except OSError:
            logger.warning("PAIR_MAP_PATH %s unreadable; using default", path)
        else:
            parsed = _parse_pair_map(text)
            if parsed:
                return MappingProxyType(parsed)
            logger.warning(
                "PAIR_MAP_PATH %s did not contain a valid mapping; using default", path
            )
    raw = _get_env("PAIR_MAP")
    if raw:
        parsed = _parse_pair_map(raw)
        if parsed:
            return MappingProxyType(parsed)
        logger.warning("Invalid PAIR_MAP override %r; using default mapping", raw)
    return MappingProxyType(dict(default))


COMMISSION_PER_SIDE: float = _load_commission(DEFAULT_COMMISSION_PER_SIDE)
PAIR_MAP: Mapping[str, str] = _load_pair_map(DEFAULT_PAIR_MAP)

