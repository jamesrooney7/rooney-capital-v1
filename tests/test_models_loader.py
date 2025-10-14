from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.loader import ModelBundle, load_model_bundle, strategy_kwargs_from_bundle


class DummyModel:
    def __init__(self, feature_names: tuple[str, ...]):
        self.feature_names_in_ = feature_names


@pytest.fixture
def models_tmpdir(tmp_path: Path) -> Path:
    """Create a temporary directory that mimics ``src/models``."""

    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _write_metadata(path: Path, **overrides: object) -> None:
    payload = {
        "Symbol": "ES",
        "Prod_Threshold": 0.62,
        "Features": ["feature_from_metadata"],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_model_placeholder(path: Path) -> None:
    path.write_text("not a pointer", encoding="ascii")


def test_load_model_bundle_prefers_metadata_features(monkeypatch, models_tmpdir):
    metadata_path = models_tmpdir / "ES_best.json"
    model_path = models_tmpdir / "ES_rf_model.pkl"
    _write_metadata(metadata_path, Features=["feature_from_metadata"])
    _write_model_placeholder(model_path)

    monkeypatch.setattr(
        "models.loader.joblib.load",
        lambda path: {"model": object(), "features": ("feature_from_model",)},
    )

    bundle = load_model_bundle("es", base_dir=models_tmpdir)

    assert bundle.features == ("feature_from_metadata",)
    assert bundle.threshold == pytest.approx(0.62)


def test_load_model_bundle_falls_back_to_model_features(monkeypatch, models_tmpdir):
    metadata_path = models_tmpdir / "ES_best.json"
    model_path = models_tmpdir / "ES_rf_model.pkl"
    _write_metadata(metadata_path, Features=None, Prod_Threshold="0.7")
    _write_model_placeholder(model_path)

    dummy_model = DummyModel(("f1", "f2"))
    monkeypatch.setattr("models.loader.joblib.load", lambda path: dummy_model)

    bundle = load_model_bundle("ES", base_dir=models_tmpdir)

    assert bundle.features == ("f1", "f2")
    assert bundle.threshold == pytest.approx(0.7)


def test_load_model_bundle_raises_for_lfs_pointer(models_tmpdir):
    metadata_path = models_tmpdir / "ES_best.json"
    model_path = models_tmpdir / "ES_rf_model.pkl"
    _write_metadata(metadata_path)
    model_path.write_text(
        """version https://git-lfs.github.com/spec/v1
        oid sha256:123
        size 10
        """,
        encoding="ascii",
    )

    with pytest.raises(RuntimeError):
        load_model_bundle("ES", base_dir=models_tmpdir)


def test_strategy_kwargs_from_bundle_roundtrip():
    bundle = ModelBundle(
        symbol="ES",
        model=object(),
        features=("f1", "f2"),
        threshold=0.55,
        metadata={"Features": ["f1", "f2"]},
    )

    kwargs = strategy_kwargs_from_bundle(bundle)

    assert kwargs == {
        "ml_model": bundle.model,
        "ml_features": bundle.features,
        "ml_threshold": bundle.threshold,
    }
