import math
import threading
import time
import types
from pathlib import Path

import pandas as pd

from runner.databento_bridge import DatabentoLiveData, QueueFanout, QueueSignal
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig


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
