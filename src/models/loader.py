"""Utilities for loading trained Random Forest bundles for the IBS strategy.

The optimisation pipeline exports two artefacts per instrument:

* ``<SYMBOL>_best.json`` – metadata describing the tuned hyper-parameters,
  production probability threshold, and feature list that produced the best
  cross-validated performance.
* ``<SYMBOL>_rf_model.pkl`` – the trained ``RandomForestClassifier`` (and
  accompanying feature list) serialised with :mod:`joblib`.

These helpers make it easy for the live strategy runner to hydrate both the
trained model and its configuration in a consistent fashion.  The returned
``ModelBundle`` can be plugged straight into ``IbsStrategy`` via the
``ml_model``, ``ml_features`` and ``ml_threshold`` parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import joblib

__all__ = [
    "ModelBundle",
    "load_model_bundle",
    "strategy_kwargs_from_bundle",
]


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Container for a trained ML filter and its configuration."""

    symbol: str
    model: object
    features: tuple[str, ...]
    threshold: float | None
    metadata: Mapping[str, object]

    def strategy_kwargs(self) -> dict[str, object]:
        """Return ``IbsStrategy`` keyword arguments for this bundle."""

        return strategy_kwargs_from_bundle(self)


def _models_dir(base_dir: str | Path | None) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parent
    return Path(base_dir)


def _normalise_symbol(symbol: str) -> str:
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    return symbol.upper()


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            header = fh.read(200)
    except FileNotFoundError:
        raise
    try:
        text = header.decode("ascii")
    except UnicodeDecodeError:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/")


def _load_metadata(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing metadata file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file {path} did not contain a JSON object")
    return data


def _coerce_features(values: Sequence[object] | None) -> tuple[str, ...]:
    if not values:
        return ()
    coerced: list[str] = []
    for value in values:
        if value is None:
            continue
        coerced.append(str(value))
    return tuple(coerced)


def _load_model_payload(path: Path) -> tuple[object, tuple[str, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Model file {path} is a Git LFS pointer; fetch the artefact with 'git lfs pull'."
        )
    payload = joblib.load(path)
    features: tuple[str, ...] = ()
    model = payload
    if isinstance(payload, Mapping):
        model = payload.get("model", payload)
        features = _coerce_features(payload.get("features"))
    if not features:
        feature_attr: Iterable[object] | None = getattr(model, "feature_names_in_", None)
        features = _coerce_features(feature_attr)
    return model, features


def load_model_bundle(symbol: str, base_dir: str | Path | None = None) -> ModelBundle:
    """Load the ML model, feature set and production threshold for ``symbol``."""

    sym = _normalise_symbol(symbol)
    models_dir = _models_dir(base_dir)
    metadata_path = models_dir / f"{sym}_best.json"
    model_path = models_dir / f"{sym}_rf_model.pkl"

    metadata = _load_metadata(metadata_path)
    model, model_features = _load_model_payload(model_path)

    # Try both old format (Prod_Threshold/Features) and new format (threshold/features)
    threshold = metadata.get("threshold") or metadata.get("Prod_Threshold")
    try:
        threshold_val = float(threshold) if threshold is not None else None
    except (TypeError, ValueError):
        threshold_val = None

    metadata_features = _coerce_features(metadata.get("features") or metadata.get("Features"))
    features = metadata_features or model_features

    return ModelBundle(
        symbol=sym,
        model=model,
        features=features,
        threshold=threshold_val,
        metadata=metadata,
    )


def strategy_kwargs_from_bundle(bundle: ModelBundle) -> dict[str, object]:
    """Return keyword arguments for :class:`~strategy.ibs_strategy.IbsStrategy`."""

    return {
        "ml_model": bundle.model,
        "ml_features": bundle.features,
        "ml_threshold": bundle.threshold,
    }
