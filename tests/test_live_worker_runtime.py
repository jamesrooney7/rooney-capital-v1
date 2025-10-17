"""Runtime behaviour tests for :mod:`runner.live_worker`."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import databento
import pytest

from runner import live_worker
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig


class DummyMetadata:
    def list_datasets(self, timeout: int | None = None):  # pragma: no cover - simple stub
        return ["dataset"]


class DummyLive:
    def __init__(self, key: str | None = None):  # pragma: no cover - simple stub
        self.metadata = DummyMetadata()


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        databento_api_key="demo-key",
        contract_map_path=Path("Data/Databento_contract_map.yml"),
        models_path=None,
        symbols=("ES",),
        starting_cash=0.0,
        backfill=True,
        backfill_lookback=0,
        queue_maxsize=64,
        heartbeat_interval=None,
        heartbeat_file=None,
        heartbeat_write_interval=30.0,
        poll_interval=0.1,
        traderspost_webhook=None,
        instruments={},
        preflight=PreflightConfig(),
        killswitch=False,
    )


def _patch_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_worker,
        "load_model_bundle",
        lambda symbol, base_dir=None: SimpleNamespace(features=("f1", "f2")),
    )
    monkeypatch.setattr(live_worker, "strategy_kwargs_from_bundle", lambda bundle: {})

    class _Response:
        def __init__(self, status_code: int = 200, text: str = "OK") -> None:  # pragma: no cover
            self.status_code = status_code
            self.text = text

    monkeypatch.setattr(
        live_worker.requests,
        "post",
        lambda url, json=None, timeout=None: _Response(),
    )
    monkeypatch.setattr(databento, "Live", DummyLive)


def test_run_cerebro_aborts_without_data_feeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    worker = LiveWorker(_runtime_config())

    # Simulate missing feeds after setup.
    worker._data_feeds.clear()
    worker.cerebro.datas.clear()

    run_called = False

    def _fake_run(*args, **kwargs):  # pragma: no cover - should not be invoked
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(worker.cerebro, "run", _fake_run)

    heartbeat_calls: list[tuple[str, bool, dict | None]] = []

    def _fake_heartbeat(status: str, *, force: bool = False, details=None) -> None:
        heartbeat_calls.append((status, force, details if isinstance(details, dict) else None))

    monkeypatch.setattr(worker, "_update_heartbeat", _fake_heartbeat)

    caplog.set_level(logging.INFO, logger=live_worker.logger.name)

    worker._run_cerebro()

    assert "No data feeds configured" in caplog.text
    assert run_called is False
    assert any(status == "failed" and force for status, force, _ in heartbeat_calls)
    assert any(
        details and details.get("reason") == "no_data_feeds"
        for _, _, details in heartbeat_calls
    )
