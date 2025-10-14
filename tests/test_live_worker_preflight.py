"""Pre-flight validation coverage for :mod:`runner.live_worker`."""

from __future__ import annotations

import logging
from pathlib import Path

import databento
import pytest
import requests

from models.loader import ModelBundle
from runner import live_worker
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig


class DummyModel:
    def predict_proba(self, values):  # pragma: no cover - simple stub
        return [[0.5, 0.5] for _ in values]


class DummyMetadata:
    def list_datasets(self, timeout: int | None = None):  # pragma: no cover - simple stub
        return ["dataset"]


class DummyLive:
    def __init__(self, key: str | None = None):  # pragma: no cover - simple stub
        self.metadata = DummyMetadata()


def _successful_model_bundle(symbol: str) -> ModelBundle:
    return ModelBundle(
        symbol=symbol,
        model=DummyModel(),
        features=("f1", "f2"),
        threshold=0.5,
        metadata={},
    )


def _successful_config(preflight: PreflightConfig | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        databento_api_key="demo-key",
        contract_map_path=Path("Data/Databento_contract_map.yml"),
        models_path=None,
        symbols=("ES", "NQ"),
        starting_cash=0.0,
        backfill=True,
        queue_maxsize=64,
        heartbeat_interval=None,
        poll_interval=0.1,
        traderspost_webhook="https://example.com/webhook",
        traderspost_api_base=None,
        traderspost_api_key=None,
        reconciliation_enabled=False,
        reconciliation_wait_seconds=0.0,
        reconciliation_max_retries=0,
        instruments={},
        preflight=preflight or PreflightConfig(),
    )


def _patch_successful_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_worker, "load_model_bundle", lambda symbol, base_dir=None: _successful_model_bundle(symbol))

    class _Response:
        def __init__(self, status_code: int = 200, text: str = "OK") -> None:
            self.status_code = status_code
            self.text = text

    monkeypatch.setattr(
        live_worker.requests,
        "post",
        lambda url, json=None, timeout=None: _Response(),
    )
    monkeypatch.setattr(databento, "Live", DummyLive)


def test_preflight_happy_path(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is True
    assert "✅ ALL PRE-FLIGHT CHECKS PASSED" in caplog.text


def test_preflight_ml_model_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _missing_bundle(symbol: str, base_dir=None):
        raise FileNotFoundError("bundle missing")

    monkeypatch.setattr(live_worker, "load_model_bundle", _missing_bundle)
    monkeypatch.setattr(live_worker.requests, "post", lambda *args, **kwargs: type("R", (), {"status_code": 200, "text": ""})())
    monkeypatch.setattr(databento, "Live", DummyLive)

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: ML Models" in caplog.text


def test_preflight_traderspost_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    monkeypatch.setattr(
        live_worker.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectionError("boom")),
    )

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: TradersPost Connection" in caplog.text


def test_preflight_databento_invalid_key(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)

    class _FailingLive:
        def __init__(self, key: str | None = None):  # pragma: no cover - simple stub
            raise RuntimeError("Unauthorized")

    monkeypatch.setattr(databento, "Live", _FailingLive)

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: Databento Connection" in caplog.text


def test_preflight_missing_contract_spec(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    trimmed_specs = {key: value for key, value in live_worker.CONTRACT_SPECS.items() if key != "ES"}
    monkeypatch.setattr(live_worker, "CONTRACT_SPECS", trimmed_specs)

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: Reference Data" in caplog.text


def test_preflight_missing_reference_feed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    worker = LiveWorker(_successful_config())
    original_reference_symbols = worker.contract_map.reference_symbols()
    monkeypatch.setattr(
        worker.contract_map,
        "reference_symbols",
        lambda: tuple(sym for sym in original_reference_symbols if sym != "TLT"),
    )

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: Reference Data" in caplog.text


def test_preflight_multiple_failures(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    monkeypatch.setattr(
        live_worker.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectionError("boom")),
    )

    class _FailingLive:
        def __init__(self, key: str | None = None):  # pragma: no cover - simple stub
            raise RuntimeError("Unauthorized")

    monkeypatch.setattr(databento, "Live", _FailingLive)

    worker = LiveWorker(_successful_config(preflight=PreflightConfig(fail_fast=False)))

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: TradersPost Connection, Databento Connection" in caplog.text
