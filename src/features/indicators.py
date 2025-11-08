"""Shared helpers for building normalized feature matrices.

These utilities centralize the feature-ingestion logic used by both the
Bayesian optimisation driver (``rf_cpcv_random_then_bo``) and the worker / BO
pipeline (``oos_cpcv_rf_tune``).  The previous implementation relied on
module-local white-lists that quickly diverged as new columns were added to the
``transformed_trades`` export.  The functions here instead inspect the provided
DataFrame directly, normalising column names and selecting every non-admin
field while still preferring percentile companions when present.

Normalisation rules:

* lower-case every column name;
* replace any non alpha-numeric character with an underscore;
* collapse duplicate underscores and strip leading/trailing underscores;
* ensure resulting names are unique by appending ``__{i}`` when necessary.

Binary-like columns are coerced to ``0.0``/``1.0`` using a permissive lookup,
numeric columns are cast to ``float`` and string columns with more than two
distinct values are one-hot encoded (again using normalised column names for
the resulting dummy variables).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

__all__ = [
    "ADMIN_NORMALISED_NAMES",
    "BINARY_FLAG_MAP",
    "normalize_column_name",
    "rename_with_normalised_columns",
    "build_normalised_feature_matrix",
]


_NORMALISE_RE = re.compile(r"[^0-9a-z]+")


def normalize_column_name(name: str) -> str:
    """Normalise a column name to ``snake_case`` using only ASCII characters."""

    normalised = _NORMALISE_RE.sub("_", str(name).strip().lower())
    normalised = re.sub(r"_+", "_", normalised).strip("_")
    return normalised or "col"


def rename_with_normalised_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Return a mapping from the original columns to unique normalised names."""

    counts: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for col in columns:
        base = normalize_column_name(col)
        idx = counts.get(base, 0)
        if idx:
            name = f"{base}__{idx}"
        else:
            name = base
        counts[base] = idx + 1
        mapping[col] = name
    return mapping


# Identifiers / targets removed from the feature matrix.
ADMIN_NORMALISED_NAMES = {
    "trade_id",
    "trade",
    "date_time",
    "date",
    "exit_date_time",
    "exit_signal",
    "y_return",
    "y_binary",
    "y_pnl_usd",
    "net_p_l",
    "net_p_l_usd",
    "ibs_exit",
}


BINARY_FLAG_MAP = {
    "above": 1.0,
    "below": 0.0,
    "on": 1.0,
    "off": 0.0,
    "true": 1.0,
    "false": 0.0,
    "yes": 1.0,
    "no": 0.0,
    "y": 1.0,
    "n": 0.0,
    "long": 1.0,
    "short": 0.0,
    "1": 1.0,
    "0": 0.0,
    "1.0": 1.0,
    "0.0": 0.0,
}


def _select_preferred_columns(columns: List[str]) -> List[str]:
    """Prefer ``*_pct`` variants when both the raw and percentile exist."""

    by_base: Dict[str, List[str]] = {}
    for col in columns:
        base = col[:-4] if col.endswith("_pct") else col
        by_base.setdefault(base, []).append(col)

    ordered: List[str] = []
    processed_bases = set()
    for col in columns:
        base = col[:-4] if col.endswith("_pct") else col
        if base in processed_bases:
            continue
        processed_bases.add(base)
        options = by_base.get(base, [])
        pct_cols = [c for c in options if c.endswith("_pct")]
        if pct_cols:
            # Keep the first percentile column encountered in the original order.
            first_pct = next(c for c in columns if c in pct_cols)
            ordered.append(first_pct)
        else:
            for opt in options:
                if opt not in ordered:
                    ordered.append(opt)

    return ordered


def _coerce_binary(series: pd.Series) -> pd.Series | None:
    """Attempt to coerce a text column into a binary float series."""

    if series.dropna().empty:
        return series.astype(float)

    normalised = series.astype(str).str.strip().str.lower()
    mapped = normalised.map(BINARY_FLAG_MAP)
    if mapped.notna().any():
        other = normalised[~normalised.isin(BINARY_FLAG_MAP.keys()) & series.notna()]
        if other.empty:
            return mapped.astype(float)
    return None


def _encode_categorical(series: pd.Series, col_name: str) -> pd.DataFrame:
    """One-hot encode a categorical column after normalising its values."""

    values = series.fillna("__nan__").astype(str).str.strip().str.lower()
    dummies = pd.get_dummies(values, prefix=col_name, prefix_sep="_", drop_first=True, dtype=float)
    if dummies.empty:
        return pd.DataFrame(index=series.index)
    return dummies.rename(columns=lambda c: normalize_column_name(c))


def build_normalised_feature_matrix(Xy: pd.DataFrame) -> pd.DataFrame:
    """Construct the numeric feature matrix used by RF tuning pipelines."""

    if Xy.empty:
        return pd.DataFrame(index=Xy.index)

    mapping = rename_with_normalised_columns(Xy.columns)
    df = Xy.rename(columns=mapping)

    candidate_cols = [c for c in df.columns if c not in ADMIN_NORMALISED_NAMES]
    if not candidate_cols:
        return pd.DataFrame(index=df.index)

    ordered_cols = _select_preferred_columns(candidate_cols)

    parts: List[pd.DataFrame] = []
    for col in ordered_cols:
        series = df[col]

        # Always preserve boolean dtype as binary floats.
        if series.dtype == bool:
            parts.append(series.astype(float).to_frame(col))
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            parts.append(numeric.astype(float).to_frame(col))
            continue

        binary = _coerce_binary(series)
        if binary is not None:
            parts.append(binary.to_frame(col))
            continue

        encoded = _encode_categorical(series, col)
        if not encoded.empty:
            parts.append(encoded)

    if not parts:
        return pd.DataFrame(index=df.index)

    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)

    pct_cols = [c for c in X.columns if c.endswith("_pct")]
    if pct_cols:
        X[pct_cols] = X[pct_cols].fillna(0.50)
    other_cols = [c for c in X.columns if c not in pct_cols]
    if other_cols:
        X[other_cols] = X[other_cols].fillna(0.0)

    return X

