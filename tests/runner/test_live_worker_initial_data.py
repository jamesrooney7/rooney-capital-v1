import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from runner.databento_bridge import Bar, DatabentoLiveData, QueueFanout
from runner.live_worker import LiveWorker, PreflightConfig, RuntimeConfig


def _runtime_config(symbol: str) -> RuntimeConfig:
    return RuntimeConfig(
        databento_api_key="demo-key",
        contract_map_path=Path("Data/Databento_contract_map.yml"),
        models_path=None,
        symbols=(symbol,),
        starting_cash=0.0,
        backfill=False,
        backfill_days=None,
        backfill_lookback=0,
        load_historical_warmup=False,
        historical_lookback_days=1,
        historical_warmup_batch_size=1,
        historical_warmup_queue_soft_limit=1,
        historical_warmup_compression="1min",
        queue_maxsize=8,
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


def test_wait_for_initial_data_accepts_warmup_bars(monkeypatch):
    symbol = "ES"
    queue_manager = QueueFanout({symbol: symbol}, maxsize=16)

    worker = LiveWorker.__new__(LiveWorker)
    worker.config = _runtime_config(symbol)
    worker.queue_manager = queue_manager
    worker.symbols = (symbol,)
    worker.data_symbols = (symbol,)
    worker._data_feeds = {}
    worker._required_reference_feed_names = lambda: set()

    feed = DatabentoLiveData(symbol=symbol, queue_manager=queue_manager, backfill=False)
    worker._data_feeds[symbol] = feed

    # Populate warmup data without pushing anything to the live queue.
    index = pd.date_range(
        datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3, freq="1min"
    )
    for ts in index:
        feed.extend_warmup(
            [
                Bar(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),
                    open=100.0,
                    high=101.0,
                    low=99.5,
                    close=100.5,
                    volume=1000.0,
                )
            ]
        )

    queue = queue_manager.get_queue(symbol)
    assert queue.qsize() == 0
    assert feed.warmup_backlog_size() > 0

    start = time.perf_counter()
    assert LiveWorker._wait_for_initial_data(worker, max_wait_seconds=1)
    elapsed = time.perf_counter() - start

    # The wait should exit quickly because warmup bars satisfy the requirement.
    assert elapsed < 0.5
