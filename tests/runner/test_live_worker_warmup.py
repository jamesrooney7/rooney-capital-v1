import threading
import time
from pathlib import Path

import pandas as pd

from runner.databento_bridge import DatabentoLiveData, QueueFanout
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

    for symbol in symbols:
        assert max_backlog[symbol] <= queue_limit, f"Warmup backlog exceeded limit for {symbol}"
