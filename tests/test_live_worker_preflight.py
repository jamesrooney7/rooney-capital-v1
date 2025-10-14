"""Pre-flight validation coverage for :mod:`runner.live_worker`."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
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
        heartbeat_file=None,
        heartbeat_write_interval=30.0,
        poll_interval=0.1,
        traderspost_webhook="https://example.com/webhook",
        traderspost_api_base="https://api.example.com",
        traderspost_api_key="secret",
        reconciliation_enabled=False,
        reconciliation_wait_seconds=0.0,
        reconciliation_max_retries=0,
        instruments={},
        preflight=preflight or PreflightConfig(),
        killswitch=False,
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
    monkeypatch.setattr(
        live_worker.TradersPostClient,
        "get_pending_orders",
        lambda self, symbol: [],
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


def test_preflight_traderspost_missing_api_base(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)

    config = replace(
        _successful_config(),
        reconciliation_enabled=True,
        traderspost_api_base=None,
    )
    worker = LiveWorker(config)

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "TradersPost API base URL not configured" in caplog.text


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


def test_preflight_aborts_when_policy_killswitch_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    _patch_successful_dependencies(monkeypatch)

    config_path = tmp_path / "runtime.json"
    config_payload = {
        "contract_map": "Data/Databento_contract_map.yml",
        "symbols": ["ES"],
        "databento_api_key": "demo-key",
        "preflight": {
            "skip_ml_validation": True,
            "skip_connection_checks": True,
        },
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("POLICY_KILLSWITCH", "true")

    config = live_worker.load_runtime_config(config_path)
    worker = LiveWorker(config)

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "POLICY KILLSWITCH" in caplog.text


def test_load_runtime_config_expands_env_placeholders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DB_KEY", "interpolated-databento")
    monkeypatch.setenv("TP_HOOK", "https://example.test/hook")

    config_path = tmp_path / "runtime.json"
    config_payload = {
        "contract_map": "Data/Databento_contract_map.yml",
        "symbols": ["ES"],
        "databento_api_key": "${DB_KEY}",
        "traderspost": {"webhook": "${TP_HOOK}"},
        "contracts": {
            "ES": {
                "strategy_overrides": {
                    "note": "${TP_HOOK}",
                }
            }
        },
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    config = live_worker.load_runtime_config(config_path)

    assert config.databento_api_key == "interpolated-databento"
    assert config.traderspost_webhook == "https://example.test/hook"
    assert (
        config.instruments["ES"].strategy_overrides["note"]
        == "https://example.test/hook"
    )


def test_heartbeat_file_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_successful_dependencies(monkeypatch)

    heartbeat_path = tmp_path / "heartbeat.json"
    config = replace(
        _successful_config(),
        heartbeat_file=heartbeat_path,
        heartbeat_write_interval=0.0,
    )

    worker = LiveWorker(config)
    worker._update_heartbeat(status="running", force=True)

    payload = json.loads(heartbeat_path.read_text())
    assert payload["status"] == "running"
    assert payload["symbols"] == list(config.symbols)

    worker.stop()

    payload = json.loads(heartbeat_path.read_text())
    assert payload["status"] == "stopped"
