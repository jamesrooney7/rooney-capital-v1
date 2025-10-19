"""Pre-flight validation coverage for :mod:`runner.live_worker`."""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import replace
from pathlib import Path

import databento
import pandas as pd
import pytest
import requests

from models.loader import ModelBundle
from runner import live_worker
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig
from runner import contract_map


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
    monkeypatch.setattr(databento, "Live", DummyLive)


def test_preflight_happy_path(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is True
    assert "✅ ALL PRE-FLIGHT CHECKS PASSED" in caplog.text
    assert worker._preflight_summary["status"] == "passed"
    assert worker._preflight_summary.get("failed_checks") == []


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
    assert worker._preflight_summary["status"] == "failed"
    assert worker._preflight_summary.get("failed_checks") == ["ML Models"]


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
    assert worker._preflight_summary["status"] == "failed"
    assert worker._preflight_summary.get("failed_checks") == ["TradersPost Connection"]


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
    original_loader = contract_map.load_contract_map

    def _load_without_tlt(path: Path | str):
        cmap = original_loader(path)
        refs = tuple(sym for sym in cmap.reference_symbols() if sym != "TLT")
        monkeypatch.setattr(cmap, "reference_symbols", lambda: refs)
        return cmap

    monkeypatch.setattr(live_worker, "load_contract_map", _load_without_tlt)

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Failed checks: Reference Data" in caplog.text


def test_preflight_missing_pair_feed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _patch_successful_dependencies(monkeypatch)
    monkeypatch.setattr(live_worker, "PAIR_MAP", {"ES": "ZZ"})

    original_loader = contract_map.load_contract_map

    def _load_contracts(path: Path | str):
        return original_loader(path)

    monkeypatch.setattr(live_worker, "load_contract_map", _load_contracts)

    worker = LiveWorker(_successful_config())

    caplog.set_level(logging.INFO)
    assert worker.run_preflight_checks() is False
    assert "Pair data feed missing" in caplog.text


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
    assert worker._preflight_summary["status"] == "failed"
    assert worker._preflight_summary.get("failed_checks") == [
        "TradersPost Connection",
        "Databento Connection",
    ]


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
    assert worker._preflight_summary["status"] == "failed"
    assert worker._preflight_summary.get("failed_checks") == ["Policy Killswitch"]


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


def test_load_runtime_config_parses_backfill_and_session(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    config_payload = {
        "contract_map": "Data/Databento_contract_map.yml",
        "symbols": ["ES"],
        "databento_api_key": "demo-key",
        "backfill_minutes": 90,
        "resample_session_start": "22:00",
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    config = live_worker.load_runtime_config(config_path)

    assert config.backfill_lookback == 90
    assert config.resample_session_start == dt.time(22, 0)


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
    details = payload["details"]
    assert details["preflight"]["status"] == "not_run"
    assert "queue_fanout" in details["databento"]
    assert "subscribers" in details["databento"]
    assert details.get("traderspost", {}).get("last_success") is None

    worker.stop()

    payload = json.loads(heartbeat_path.read_text())
    assert payload["status"] == "stopped"
    assert "databento" in payload["details"]


def test_run_cerebro_aborts_until_all_queues_ready(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_successful_dependencies(monkeypatch)
    worker = LiveWorker(_successful_config())

    required_queues = {str(symbol) for symbol in worker.data_symbols}
    for feed_name in worker._required_reference_feed_names():
        base = str(feed_name or "").strip()
        if base.endswith("_day"):
            base = base[: -len("_day")]
        elif base.endswith("_hour"):
            base = base[: -len("_hour")]
        if base:
            required_queues.add(base)

    queue_sizes: dict[str, int] = {name: 0 for name in required_queues}
    queues: dict[str, object] = {}

    class _Queue:
        def __init__(self, name: str) -> None:
            self._name = name

        def qsize(self) -> int:
            return queue_sizes.get(self._name, 0)

    def _get_queue(name: str):  # pragma: no cover - helper stub
        queue = queues.get(name)
        if queue is None:
            queue = _Queue(name)
            queues[name] = queue
        return queue

    monkeypatch.setattr(worker.queue_manager, "get_queue", _get_queue)

    run_called = False

    def _fake_run(*args, **kwargs):  # pragma: no cover - helper stub
        nonlocal run_called
        run_called = True
        return []

    monkeypatch.setattr(worker.cerebro, "run", _fake_run)
    monkeypatch.setattr(live_worker.time, "sleep", lambda _: None)

    caplog.set_level(logging.INFO)
    worker._run_cerebro()

    assert worker._stop_event.is_set()
    assert run_called is False

    error_messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert any("Timeout waiting for initial data" in message for message in error_messages)

    expected_missing = ", ".join(sorted(required_queues))
    assert any(expected_missing in message for message in error_messages)


def test_historical_warmup_includes_reference_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_dependencies(monkeypatch)

    captured_symbols: list[tuple[str, str, tuple[str, ...]]] = []
    warmup_snapshots: list[dict[str, int]] = []

    original_finalize = LiveWorker._finalize_indicator_warmup

    def _capture_finalize(self: LiveWorker) -> None:
        warmup_snapshots.append(dict(self._historical_warmup_counts))
        original_finalize(self)

    monkeypatch.setattr(LiveWorker, "_finalize_indicator_warmup", _capture_finalize)

    base_timestamp = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    sample_frame = pd.DataFrame(
        {
            "ts_event": [
                base_timestamp.value,
                (base_timestamp + pd.Timedelta(seconds=30)).value,
            ],
            "price": [100.0, 101.0],
            "volume": [1.0, 2.0],
        }
    )

    class _Payload:
        def __init__(self, frame: pd.DataFrame) -> None:
            self._frame = frame

        def to_df(self) -> pd.DataFrame:
            return self._frame.copy()

    def _load_historical_data(
        *,
        api_key: str,
        dataset: str,
        symbols: tuple[str, ...],
        stype_in: str,
        days: int,
        contract_map,
        on_symbol_loaded,
    ) -> None:
        captured_symbols.append((dataset, stype_in, tuple(symbols)))
        for symbol in symbols:
            on_symbol_loaded(symbol, _Payload(sample_frame))

    monkeypatch.setattr(live_worker, "load_historical_data", _load_historical_data)

    config = replace(
        _successful_config(),
        load_historical_warmup=True,
        historical_lookback_days=1,
    )

    worker = LiveWorker(config)

    assert captured_symbols, "Historical loader should be invoked"
    assert warmup_snapshots, "Warmup counts should be captured"

    loaded_symbol_sets = {symbol for _, _, group in captured_symbols for symbol in group}
    assert loaded_symbol_sets == set(worker.data_symbols)

    warmup_counts = warmup_snapshots[0]
    reference_symbols = [
        symbol for symbol in worker.reference_symbols if symbol in worker.data_symbols
    ]
    assert reference_symbols, "Expected at least one reference symbol"

    for symbol in reference_symbols:
        assert warmup_counts.get(symbol, 0) > 0


def test_historical_warmup_uses_dataset_group_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_dependencies(monkeypatch)

    original_build = LiveWorker._build_dataset_groups

    def _custom_build(self: LiveWorker, contract_symbols, reference_symbols):
        dataset_groups, symbols_by_group = original_build(
            self, contract_symbols, reference_symbols
        )
        dataset_groups = {key: tuple(values) for key, values in dataset_groups.items()}
        symbols_by_group = {key: tuple(values) for key, values in symbols_by_group.items()}

        original_key: tuple[str, str] | None = None
        for key, symbols in list(symbols_by_group.items()):
            if "NQ" in symbols:
                original_key = key
                remaining = tuple(sym for sym in symbols if sym != "NQ")
                if remaining:
                    symbols_by_group[key] = remaining
                else:
                    symbols_by_group.pop(key)
                break

        if original_key is not None:
            original_codes = dataset_groups.get(original_key, tuple())
            filtered_codes = tuple(code for code in original_codes if code != "MNQ.FUT")
            if filtered_codes:
                dataset_groups[original_key] = filtered_codes
            else:
                dataset_groups.pop(original_key, None)

        custom_key = ("CUSTOM.DATA", "product_id")
        dataset_groups[custom_key] = ("MNQ.FUT",)
        symbols_by_group[custom_key] = ("NQ",)

        return dataset_groups, symbols_by_group

    monkeypatch.setattr(LiveWorker, "_build_dataset_groups", _custom_build)

    load_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def _load_historical_data(
        *,
        api_key: str,
        dataset: str,
        symbols: tuple[str, ...],
        stype_in: str,
        days: int,
        contract_map,
        on_symbol_loaded,
    ) -> None:
        load_calls.append((dataset, stype_in, tuple(symbols)))

    monkeypatch.setattr(live_worker, "load_historical_data", _load_historical_data)

    config = replace(
        _successful_config(),
        load_historical_warmup=True,
        historical_lookback_days=1,
    )

    LiveWorker(config)

    assert load_calls
    assert any(
        dataset == "GLBX.MDP3" and stype == "parent" and "ES" in symbols and "NQ" not in symbols
        for dataset, stype, symbols in load_calls
    )
    assert any(
        dataset == "CUSTOM.DATA" and stype == "product_id" and symbols == ("NQ",)
        for dataset, stype, symbols in load_calls
    )

