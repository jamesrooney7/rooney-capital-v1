"""Loader for Strategy Factory ML Meta-Labeling models.

Strategy Factory models are trained using the ML meta-labeling pipeline which
saves models in a different format than the original IBS model loader:

* ``{SYMBOL}_ml_meta_labeling_final_model.pkl`` - The trained model (LightGBM or ensemble)
* ``{SYMBOL}_ml_meta_labeling_selected_features.json`` - Selected features list
* ``{SYMBOL}_ml_meta_labeling_held_out_results.json`` - Held-out test performance

This loader provides compatibility with the live worker's ML filtering system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb

__all__ = [
    "FactoryModelBundle",
    "load_factory_model_bundle",
    "factory_strategy_kwargs_from_bundle",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactoryModelBundle:
    """Container for a trained Strategy Factory ML model and its configuration."""

    symbol: str
    strategy_name: str
    model: object
    features: tuple[str, ...]
    threshold: float
    metadata: Mapping[str, object]

    def strategy_kwargs(self) -> dict[str, object]:
        """Return strategy keyword arguments for this bundle."""
        return factory_strategy_kwargs_from_bundle(self)


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


def _load_features(path: Path) -> tuple[str, ...]:
    """Load features from selected_features.json file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        logger.warning(f"Missing features file: {path}")
        return ()

    # The file structure has 'selected_features' as a list
    if isinstance(data, dict):
        features = data.get("selected_features", [])
        if isinstance(features, list):
            return tuple(str(f) for f in features if f)
        # Also check for direct list of features
        features = data.get("features", [])
        if isinstance(features, list):
            return tuple(str(f) for f in features if f)
    elif isinstance(data, list):
        return tuple(str(f) for f in data if f)

    return ()


def _load_held_out_results(path: Path) -> Mapping[str, object]:
    """Load held-out test results for metadata."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}

    if isinstance(data, dict):
        return data
    return {}


def _load_model(path: Path) -> object:
    """Load the trained model from pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")

    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Model file {path} is a Git LFS pointer; fetch the artefact with 'git lfs pull'."
        )

    try:
        # Try loading as joblib pickle (could be ensemble or other format)
        payload = joblib.load(path)

        # Check if it's an ensemble dictionary
        if isinstance(payload, dict):
            if 'meta_learner' in payload:
                # This is an ensemble model
                return EnsembleModelWrapper(payload)
            elif 'model' in payload:
                return payload['model']
            else:
                return payload

        return payload
    except Exception as e:
        logger.warning(f"Failed to load as joblib, trying LightGBM native format: {e}")

    # Try loading as native LightGBM format
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")


class EnsembleModelWrapper:
    """Wrapper for ensemble models to provide predict_proba interface."""

    def __init__(self, ensemble_data: dict):
        self.lightgbm_model = ensemble_data.get('lightgbm_model')
        self.catboost_model = ensemble_data.get('catboost_model')
        self.xgboost_model = ensemble_data.get('xgboost_model')
        self.meta_learner = ensemble_data.get('meta_learner')
        self.model_weights = ensemble_data.get('model_weights', {})

    def predict_proba(self, X):
        """Get probability predictions using ensemble."""
        import numpy as np
        import pandas as pd

        # Ensure X is DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        base_preds = []

        # Get predictions from each model
        if self.lightgbm_model is not None:
            try:
                pred = self.lightgbm_model.predict_proba(X)[:, 1]
                base_preds.append(('lightgbm', pred))
            except Exception:
                try:
                    # Try booster interface
                    pred = self.lightgbm_model.predict(X)
                    base_preds.append(('lightgbm', pred))
                except Exception:
                    pass

        if self.catboost_model is not None:
            try:
                pred = self.catboost_model.predict_proba(X)[:, 1]
                base_preds.append(('catboost', pred))
            except Exception:
                pass

        if self.xgboost_model is not None:
            try:
                pred = self.xgboost_model.predict_proba(X)[:, 1]
                base_preds.append(('xgboost', pred))
            except Exception:
                pass

        if not base_preds:
            raise RuntimeError("No base models available for prediction")

        # If meta-learner available, use it
        if self.meta_learner is not None and len(base_preds) > 1:
            X_meta = np.column_stack([pred for _, pred in base_preds])
            ensemble_pred = self.meta_learner.predict_proba(X_meta)[:, 1]
            # Return as 2D array for compatibility
            return np.column_stack([1 - ensemble_pred, ensemble_pred])

        # Simple average of available predictions
        avg_pred = np.mean([pred for _, pred in base_preds], axis=0)
        return np.column_stack([1 - avg_pred, avg_pred])


class LightGBMBoosterWrapper:
    """Wrapper for LightGBM Booster to provide sklearn-like interface."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict_proba(self, X):
        """Get probability predictions."""
        import numpy as np
        import pandas as pd

        # Ensure X is DataFrame with correct columns
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        # LightGBM Booster returns probabilities directly for binary classification
        preds = self.booster.predict(X_values)

        # Return as 2D array for compatibility with sklearn interface
        return np.column_stack([1 - preds, preds])


def load_factory_model_bundle(
    symbol: str,
    strategy_name: str = "",
    base_dir: str | Path | None = None,
) -> FactoryModelBundle:
    """Load the ML model, feature set and configuration for a Strategy Factory model.

    Args:
        symbol: Trading symbol (e.g., 'ES', 'NQ')
        strategy_name: Strategy name (e.g., 'ATRBuyDip', 'AvgHLRangeIBS')
        base_dir: Base directory containing model files. If None, uses src/models.

    Returns:
        FactoryModelBundle with model, features, and metadata

    File naming convention:
        {SYMBOL}_ml_meta_labeling_final_model.pkl
        {SYMBOL}_ml_meta_labeling_selected_features.json
        {SYMBOL}_ml_meta_labeling_held_out_results.json
    """
    sym = _normalise_symbol(symbol)
    models_dir = _models_dir(base_dir)

    # Look for model file with symbol prefix
    model_path = models_dir / f"{sym}_ml_meta_labeling_final_model.pkl"
    features_path = models_dir / f"{sym}_ml_meta_labeling_selected_features.json"
    results_path = models_dir / f"{sym}_ml_meta_labeling_held_out_results.json"

    # Also try with strategy name suffix for multi-strategy setups
    if strategy_name and not model_path.exists():
        alt_model_path = models_dir / f"{sym}_{strategy_name}_ml_meta_labeling_final_model.pkl"
        alt_features_path = models_dir / f"{sym}_{strategy_name}_ml_meta_labeling_selected_features.json"
        alt_results_path = models_dir / f"{sym}_{strategy_name}_ml_meta_labeling_held_out_results.json"

        if alt_model_path.exists():
            model_path = alt_model_path
            features_path = alt_features_path
            results_path = alt_results_path

    # Check symbol subdirectory
    if not model_path.exists():
        subdir = models_dir / sym
        if subdir.exists():
            sub_model_path = subdir / f"{sym}_ml_meta_labeling_final_model.pkl"
            if sub_model_path.exists():
                model_path = sub_model_path
                features_path = subdir / f"{sym}_ml_meta_labeling_selected_features.json"
                results_path = subdir / f"{sym}_ml_meta_labeling_held_out_results.json"

    # Load model
    model = _load_model(model_path)

    # Wrap LightGBM Booster if needed
    if isinstance(model, lgb.Booster):
        model = LightGBMBoosterWrapper(model)

    # Load features
    features = _load_features(features_path)

    # Load metadata
    metadata = _load_held_out_results(results_path)

    # Get threshold from metadata or use default
    threshold = metadata.get("optimal_threshold", 0.5)
    if threshold is None:
        threshold = 0.5
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.5

    logger.info(
        f"Loaded Strategy Factory model for {sym}: "
        f"{len(features)} features, threshold={threshold:.3f}"
    )

    return FactoryModelBundle(
        symbol=sym,
        strategy_name=strategy_name,
        model=model,
        features=features,
        threshold=threshold,
        metadata=metadata,
    )


def factory_strategy_kwargs_from_bundle(bundle: FactoryModelBundle) -> dict[str, object]:
    """Return keyword arguments for StrategyFactoryAdapter."""
    return {
        "ml_model": bundle.model,
        "ml_features": bundle.features,
        "ml_threshold": bundle.threshold,
    }
