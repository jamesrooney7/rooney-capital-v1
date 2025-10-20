import math
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from runner.databento_bridge import Bar, DatabentoLiveData, QueueFanout, QueueSignal
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig
from runner.ml_feature_tracker import MlFeatureTracker


def _runtime_config(symbols, batch_size, queue_limit) -> RuntimeConfig:
    return RuntimeConfig(
        databento_api_key="demo-key",
        contract_map_path=Path("Data/Databento_contract_map.yml"),
        models_path=None,
        symbols=tuple(symbols),
        starting_cash=0.0,
        backfill=False,
        backfill_days=None,
        backfill_lookback=0,
        load_historical_warmup=True,
        historical_lookback_days=1,
        historical_warmup_batch_size=batch_size,
        historical_warmup_queue_soft_limit=queue_limit,
        queue_maxsize=64,
        heartbeat_interval=None,
        heartbeat_file=None,
        heartbeat_write_interval=30.0,
        poll_interval=0.1,
        traderspost_webhook=None,
        resample_session_start=None,
        instruments={},
        preflight=PreflightConfig(),
        killswitch=False,
    )


def test_historical_warmup_batches_respect_queue_limit():
    symbols = ("ES", "NQ", "YM", "RTY", "CL")
    batch_size = 5000
    queue_limit = 6000
    bars_per_symbol = batch_size * 2

    queue_manager = QueueFanout({symbol: symbol for symbol in symbols}, maxsize=200000)

    worker = LiveWorker.__new__(LiveWorker)
    worker.config = _runtime_config(symbols, batch_size, queue_limit)
    worker.symbols = symbols
    worker._data_feeds = {}
    worker._historical_warmup_counts = {}
    worker._stop_event = threading.Event()
    worker._historical_warmup_wait_log_interval = 0.0
    worker._historical_warmup_lock = threading.Lock()
    worker._historical_warmup_started = False

    max_backlog: dict[str, int] = {}
    drain_stop = threading.Event()
    lock = threading.Lock()
    consumer_threads: list[threading.Thread] = []

    try:
        for symbol in symbols:
            feed = DatabentoLiveData(symbol=symbol, queue_manager=queue_manager, backfill=False)
            worker._data_feeds[symbol] = feed
            max_backlog[symbol] = 0

            original_extend = feed.extend_warmup

            def recording_extend(bars, *, _orig=original_extend, _feed=feed, _symbol=symbol):
                count = _orig(bars)
                backlog = _feed.warmup_backlog_size()
                with lock:
                    if backlog > max_backlog[_symbol]:
                        max_backlog[_symbol] = backlog
                return count

            feed.extend_warmup = recording_extend  # type: ignore[assignment]

            def consumer(_feed=feed):
                while not drain_stop.is_set() or _feed.warmup_backlog_size() > 0:
                    if _feed.warmup_backlog_size():
                        try:
                            _feed._warmup_bars.popleft()
                        except IndexError:
                            time.sleep(0.0005)
                    else:
                        time.sleep(0.0005)

            thread = threading.Thread(target=consumer, daemon=True)
            thread.start()
            consumer_threads.append(thread)

        for idx, symbol in enumerate(symbols):
            index = pd.date_range("2024-01-01", periods=bars_per_symbol, freq="1min", tz="UTC")
            base = 100.0 + idx * 10.0
            increments = pd.Series(range(bars_per_symbol), dtype=float) * 0.1
            payload = pd.DataFrame(
                {
                    "ts_event": index.view("int64"),
                    "open": base + increments,
                    "high": base + 0.5 + increments,
                    "low": base - 0.5 + increments,
                    "close": base + 0.25 + increments,
                    "volume": 1000 + increments,
                }
            )

            # Use LiveWorker helper directly against the stub.
            LiveWorker._warmup_symbol_indicators(worker, symbol, payload)

            assert worker._historical_warmup_counts[symbol] == bars_per_symbol

    finally:
        drain_stop.set()
        for thread in consumer_threads:
            thread.join(timeout=2)

    expected_limit = queue_limit + (math.ceil(bars_per_symbol / batch_size) - 1) * batch_size
    for symbol in symbols:
        assert (
            max_backlog[symbol] <= expected_limit
        ), f"Warmup backlog exceeded limit for {symbol}"


def test_large_warmup_batches_drain_without_stall():
    symbol = "ES"
    batch_size = 5000
    queue_limit = 20000
    total_batches = 6
    bars_per_symbol = batch_size * total_batches

    queue_manager = QueueFanout({symbol: symbol}, maxsize=200000)

    worker = LiveWorker.__new__(LiveWorker)
    worker.config = _runtime_config([symbol], batch_size, queue_limit)
    worker.symbols = (symbol,)
    worker._data_feeds = {}
    worker._historical_warmup_counts = {}
    worker._stop_event = threading.Event()
    worker._historical_warmup_wait_log_interval = 0.0
    worker._historical_warmup_lock = threading.Lock()
    worker._historical_warmup_started = False

    feed = DatabentoLiveData(
        symbol=symbol,
        queue_manager=queue_manager,
        backfill=False,
        qcheck=0.5,
    )
    worker._data_feeds[symbol] = feed
    feed.start()

    class _DummyLine:
        def __setitem__(self, index, value):
            return None

    feed.lines = types.SimpleNamespace(
        datetime=_DummyLine(),
        open=_DummyLine(),
        high=_DummyLine(),
        low=_DummyLine(),
        close=_DummyLine(),
        volume=_DummyLine(),
    )

    original_extend = feed.extend_warmup
    lock = threading.Lock()
    max_backlog = 0
    backlog_history: list[int] = []
    load_timestamps: list[float] = []
    drained = 0
    loads_done = threading.Event()
    drain_stop = threading.Event()

    def recording_extend(bars):
        nonlocal max_backlog
        count = original_extend(bars)
        with lock:
            backlog = feed.warmup_backlog_size()
            if backlog > max_backlog:
                max_backlog = backlog
        return count

    feed.extend_warmup = recording_extend  # type: ignore[assignment]

    def consumer() -> None:
        nonlocal drained
        while not drain_stop.is_set():
            if feed.warmup_backlog_size() <= 0:
                if drained >= bars_per_symbol:
                    break
                time.sleep(0.0005)
                continue

            result = feed._load()
            if result:
                now = time.perf_counter()
                with lock:
                    backlog_after = feed.warmup_backlog_size()
                    backlog_history.append(backlog_after)
                    load_timestamps.append(now)
                drained += 1
                if drained >= bars_per_symbol:
                    break
            elif result is False:
                break

        loads_done.set()

    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()

    index = pd.date_range("2024-01-01", periods=bars_per_symbol, freq="1min", tz="UTC")
    increments = pd.Series(range(bars_per_symbol), dtype=float) * 0.1
    payload = pd.DataFrame(
        {
            "ts_event": index.view("int64"),
            "open": 100.0 + increments,
            "high": 100.5 + increments,
            "low": 99.5 + increments,
            "close": 100.25 + increments,
            "volume": 1000 + increments,
        }
    )

    start_time = time.perf_counter()
    LiveWorker._warmup_symbol_indicators(worker, symbol, payload)
    warmup_duration = time.perf_counter() - start_time

    assert worker._historical_warmup_counts[symbol] == bars_per_symbol

    assert loads_done.wait(timeout=5), "Warmup consumer failed to drain backlog"
    drain_stop.set()
    feed._stopped = True
    feed._queue.put_nowait(QueueSignal(QueueSignal.SHUTDOWN, symbol))
    thread.join(timeout=2)

    assert drained == bars_per_symbol
    assert warmup_duration < 5.0

    assert max_backlog <= queue_limit + (total_batches - 1) * batch_size

    assert load_timestamps, "No warmup bars were drained"
    first_window = min(100, len(load_timestamps))
    if first_window > 1:
        elapsed = load_timestamps[first_window - 1] - load_timestamps[0]
        assert elapsed < 0.2

    below_limit_index = next(
        (idx for idx, backlog in enumerate(backlog_history) if backlog <= queue_limit),
        None,
    )
    assert below_limit_index is not None
    assert below_limit_index < queue_limit + batch_size


def test_wait_for_warmup_backlog_drain_blocks_until_empty():
    symbol = "ES"
    queue_manager = QueueFanout({symbol: symbol}, maxsize=256)
    feed = DatabentoLiveData(symbol=symbol, queue_manager=queue_manager, backfill=False)

    class _DummyLine:
        def __setitem__(self, index, value):
            return None

    feed.lines = types.SimpleNamespace(
        datetime=_DummyLine(),
        open=_DummyLine(),
        high=_DummyLine(),
        low=_DummyLine(),
        close=_DummyLine(),
        volume=_DummyLine(),
    )

    worker = LiveWorker.__new__(LiveWorker)
    worker._data_feeds = {symbol: feed}
    worker._stop_event = threading.Event()
    worker._historical_warmup_wait_log_interval = 0.0

    index = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    bars = [
        Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime(),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.25,
            volume=1000.0,
        )
        for ts in index
    ]

    feed.extend_warmup(bars)
    assert feed.warmup_backlog_size() == len(bars)

    drained = 0

    def consumer() -> None:
        nonlocal drained
        time.sleep(0.05)
        while feed.warmup_backlog_size() > 0:
            result = feed._load()
            if result:
                drained += 1
            time.sleep(0.0005)

    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()

    assert LiveWorker._wait_for_warmup_backlog_drain(worker, [symbol]) is True

    thread.join(timeout=2)
    assert feed.warmup_backlog_size() == 0
    assert drained == len(bars)

def test_warmup_marks_ml_bundle_ready_without_live_ticks():
    symbol = "ES"
    worker = LiveWorker.__new__(LiveWorker)
    worker.symbols = (symbol,)
    worker._historical_warmup_counts = {symbol: 3}
    worker.ml_feature_tracker = MlFeatureTracker()

    collector = worker.ml_feature_tracker.register_bundle(symbol, ("f1", "f2"))

    # Warmup populates the feature snapshot without any live updates.
    collector.record_feature("f1", 1.0)
    collector.record_feature("f2", -0.5)

    worker._finalize_indicator_warmup()

    assert worker.ml_feature_tracker.is_ready(symbol)

    # Subsequent updates should continue to refresh readiness.
    collector.record_feature("f2", None)
    assert not worker.ml_feature_tracker.is_ready(symbol)

    collector.record_feature("f2", 0.25)
    assert worker.ml_feature_tracker.is_ready(symbol)


class _StubWarmupFeed:
    def __init__(self):
        self._warmup: list = []

    def extend_warmup(self, bars):
        items = list(bars)
        self._warmup.extend(items)
        return len(items)

    def warmup_backlog_size(self):  # pragma: no cover - convenience
        return len(self._warmup)


def test_indicator_warmup_populates_ml_features_without_live_bars():
    symbol = "ES"
    canonical = symbol.upper()
    worker = LiveWorker.__new__(LiveWorker)
    worker.symbols = (symbol,)
    worker.config = types.SimpleNamespace(
        historical_warmup_batch_size=10,
        historical_warmup_queue_soft_limit=10,
    )
    worker._data_feeds = {symbol: _StubWarmupFeed()}
    worker._historical_warmup_counts = {}
    worker._historical_warmup_wait_log_interval = 0.0
    worker.ml_feature_tracker = MlFeatureTracker()
    worker._ml_feature_lock = threading.Lock()
    worker._pending_ml_warmup = set()
    worker._ml_features_seen = {}
    worker._ml_feature_collectors = {}
    worker._ml_feature_requirements = {}
    worker._ml_warmup_published = {}

    features = ("feat_a", "feat_b")
    collector = worker.ml_feature_tracker.register_bundle(symbol, features)
    worker._ml_feature_collectors[canonical] = collector
    worker._ml_feature_requirements[canonical] = features

    index = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    payload = pd.DataFrame(
        {
            "ts_event": index.view("int64"),
            "open": [100.0, 100.5, 101.0, 101.5, 102.0],
            "high": [100.5, 101.0, 101.5, 102.0, 102.5],
            "low": [99.5, 100.0, 100.5, 101.0, 101.5],
            "close": [100.25, 100.75, 101.25, 101.75, 102.25],
            "volume": [1000, 1001, 1002, 1003, 1004],
            "feat_a": [None, 0.2, 0.3, None, 0.45],
            "feat_b": [None, None, 1.5, 1.75, 1.9],
        }
    )

    LiveWorker._warmup_symbol_indicators(worker, symbol, payload)

    snapshot = worker.ml_feature_tracker.snapshot(symbol)
    assert snapshot["feat_a"] == pytest.approx(0.45)
    assert snapshot["feat_b"] == pytest.approx(1.9)

    report = worker.ml_feature_tracker.readiness_report().get(symbol, {})
    assert report.get("feature_count") == 2
    assert report.get("missing_features") == []
    features_snapshot = report.get("features") or {}
    assert features_snapshot.get("feat_a") == pytest.approx(0.45)
    assert features_snapshot.get("feat_b") == pytest.approx(1.9)
    assert worker.ml_feature_tracker.is_ready(symbol)


def test_indicator_warmup_skips_duplicate_feature_publication(monkeypatch):
    symbol = "ES"
    canonical = symbol.upper()
    worker = LiveWorker.__new__(LiveWorker)
    worker.config = types.SimpleNamespace(
        historical_warmup_batch_size=10,
        historical_warmup_queue_soft_limit=10,
    )
    worker._data_feeds = {symbol: _StubWarmupFeed()}
    worker._historical_warmup_counts = {}
    worker._historical_warmup_wait_log_interval = 0.0
    worker.ml_feature_tracker = MlFeatureTracker()
    worker._ml_feature_lock = threading.Lock()
    worker._pending_ml_warmup = set()
    worker._ml_features_seen = {}
    worker._ml_feature_collectors = {}
    worker._ml_feature_requirements = {}
    worker._ml_warmup_published = {}

    features = ("feat_a", "feat_b")
    collector = worker.ml_feature_tracker.register_bundle(symbol, features)
    worker._ml_feature_collectors[canonical] = collector
    worker._ml_feature_requirements[canonical] = features

    call_count = {"feat_a": 0, "feat_b": 0}
    original_record = collector.record_feature

    def recording_record_feature(key, value):
        call_count[key] += 1
        original_record(key, value)

    monkeypatch.setattr(collector, "record_feature", recording_record_feature)

    index = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_event": index.view("int64"),
            "open": [100.0, 100.5, 101.0],
            "high": [100.5, 101.0, 101.5],
            "low": [99.5, 100.0, 100.5],
            "close": [100.25, 100.75, 101.25],
            "volume": [1000, 1001, 1002],
            "feat_a": [0.1, 0.2, 0.3],
            "feat_b": [0.4, 0.5, 0.6],
        }
    )

    worker._publish_warmup_ml_features(symbol, frame)
    worker._publish_warmup_ml_features(symbol, frame)

    assert call_count == {"feat_a": 1, "feat_b": 1}

