import datetime as dt
import logging
import threading
import types
from pathlib import Path

import pandas as pd
import pytest

from runner.live_worker import LiveWorker, RuntimeConfig
from runner.ml_feature_tracker import MlFeatureTracker


def test_convert_databento_preaggregated_skips_resample(monkeypatch):
    worker = LiveWorker.__new__(LiveWorker)

    index = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    payload = pd.DataFrame(
        {
            "ts_event": index.view("int64"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 11, 12],
        }
    )

    resample_calls: list[tuple[tuple, dict]] = []
    original_resample = pd.DataFrame.resample

    def recording_resample(self, *args, **kwargs):
        resample_calls.append((args, kwargs))
        return original_resample(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "resample", recording_resample)

    bars = LiveWorker._convert_databento_to_bt_bars(worker, "ES", payload)

    assert [bar.open for bar in bars] == [100.0, 101.0, 102.0]
    assert [bar.close for bar in bars] == [100.5, 101.5, 102.5]
    assert [bar.volume for bar in bars] == [10.0, 11.0, 12.0]
    assert resample_calls == []


def test_live_worker_clamps_backfill_start(monkeypatch, caplog):
    boundary = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    monkeypatch.setattr(
        LiveWorker,
        "_earliest_live_start_for_dataset",
        lambda self, dataset: boundary,
    )
    monkeypatch.setattr(LiveWorker, "_setup_data_and_strategies", lambda self: None)

    config = RuntimeConfig(
        databento_api_key="demo",
        contract_map_path=Path("Data/Databento_contract_map.yml"),
        models_path=None,
        symbols=("ES",),
        backfill=True,
        backfill_lookback=7 * 24 * 60,
        traderspost_webhook=None,
    )

    caplog.set_level(logging.INFO)
    worker = LiveWorker(config)

    assert worker.subscribers, "Expected at least one subscriber"
    starts = {subscriber.start_time for subscriber in worker.subscribers}
    assert starts == {boundary}

    clamp_messages = [
        record.getMessage()
        for record in caplog.records
        if "Clamping backfill start" in record.getMessage()
    ]
    assert clamp_messages, "Expected a clamp log message"


def test_warmup_feature_snapshot_marks_tracker_ready():
    symbol = "ES"
    worker = LiveWorker.__new__(LiveWorker)
    worker.symbols = (symbol,)
    worker.ml_feature_tracker = MlFeatureTracker()
    worker._ml_feature_lock = threading.Lock()
    worker._pending_ml_warmup = set()
    worker._ml_features_seen = {}

    tracker = worker.ml_feature_tracker
    tracker.register_bundle(symbol, ("feat_a", "feat_b"))

    worker._mark_warmup_features_pending(symbol)

    tracker.update_feature(symbol, "feat_a", 1.0)
    tracker.update_feature(symbol, "feat_b", None)

    class _Collector:
        def __init__(self):
            self.payload = {"feat_a": 1.0, "feat_b": None}

        @property
        def snapshot(self):
            return dict(self.payload)

    class _Strategy:
        def __init__(self, collector):
            self.ml_feature_collector = collector
            self.p = types.SimpleNamespace(symbol=symbol, ml_features=("feat_a", "feat_b"))

    strategy = _Strategy(_Collector())

    # First snapshot leaves one feature missing; pending set remains.
    worker._on_strategy_feature_snapshot(strategy)
    assert symbol in worker._pending_ml_warmup
    first_report = tracker.readiness_report().get(symbol, {})
    assert first_report.get("missing_features") == ["feat_b"]
    features_snapshot = first_report.get("features") or {}
    assert features_snapshot.get("feat_a") == pytest.approx(1.0)
    assert features_snapshot.get("feat_b") is None

    tracker.update_feature(symbol, "feat_b", -0.25)
    strategy.ml_feature_collector.payload["feat_b"] = -0.25

    worker._on_strategy_feature_snapshot(strategy)

    report = tracker.readiness_report().get(symbol, {})
    assert report.get("feature_count") == 2
    assert report.get("ready") is True
    assert report.get("missing_features") == []
    final_features_snapshot = report.get("features") or {}
    assert final_features_snapshot.get("feat_b") == pytest.approx(-0.25)
    assert symbol not in worker._pending_ml_warmup
