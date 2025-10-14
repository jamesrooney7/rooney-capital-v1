from __future__ import annotations

import datetime as dt
import queue

import pytest

from databento import SType, SymbolMappingMsg

from runner.databento_bridge import (
    Bar,
    DatabentoLiveData,
    DatabentoSubscriber,
    QueueFanout,
    QueueSignal,
)


def _ns(timestamp: dt.datetime) -> int:
    return int(timestamp.timestamp() * 1_000_000_000)


class _DummyTrade:
    def __init__(
        self,
        instrument_id: int,
        price: float,
        size: float,
        timestamp: dt.datetime,
    ) -> None:
        self.instrument_id = instrument_id
        self.pretty_price = price
        self.size = size
        self.ts_event = _ns(timestamp)


def test_queue_fanout_publishes_and_tracks_symbols() -> None:
    fanout = QueueFanout({"ESH4": "ES"}, maxsize=4)

    fanout.update_mapping(101, "ESH4")
    assert fanout.resolve_root(101) == "ES"

    symbol_queue = fanout.get_queue("ES")

    bar = Bar(
        symbol="ES",
        timestamp=dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc),
        open=1.0,
        high=1.5,
        low=0.5,
        close=1.25,
        volume=10.0,
    )
    fanout.publish_bar(bar)
    assert symbol_queue.get_nowait() == bar

    fanout.publish_reset("ES")
    reset = symbol_queue.get_nowait()
    assert isinstance(reset, QueueSignal)
    assert reset.kind == QueueSignal.RESET

    fanout.broadcast_shutdown()
    shutdown = symbol_queue.get_nowait()
    assert isinstance(shutdown, QueueSignal)
    assert shutdown.kind == QueueSignal.SHUTDOWN


def test_subscriber_builds_minute_bars_and_flushes() -> None:
    fanout = QueueFanout({"ESH4": "ES"}, maxsize=8)
    subscriber = DatabentoSubscriber(
        dataset="TEST",
        product_codes=("ESH4",),
        queue_manager=fanout,
        api_key="demo",
    )

    fanout.update_mapping(1, "ESH4")
    symbol_queue = fanout.get_queue("ES")

    first_minute = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    subscriber._apply_trade("ES", _DummyTrade(1, 4000.0, 1.0, first_minute.replace(second=10)))
    subscriber._apply_trade("ES", _DummyTrade(1, 4001.0, 2.0, first_minute.replace(second=45)))

    # Crossing into a new minute should emit the completed bar.
    subscriber._apply_trade(
        "ES",
        _DummyTrade(1, 3999.0, 1.0, first_minute.replace(minute=1, second=5)),
    )

    first_bar = symbol_queue.get_nowait()
    assert isinstance(first_bar, Bar)
    assert first_bar.open == 4000.0
    assert first_bar.high == 4001.0
    assert first_bar.low == 4000.0
    assert first_bar.close == 4001.0
    assert first_bar.volume == pytest.approx(3.0)

    # Flushing emits the final partial bar.
    subscriber.flush()
    second_bar = symbol_queue.get_nowait()
    assert second_bar.close == 3999.0
    assert second_bar.volume == pytest.approx(1.0)


def test_subscriber_handles_symbol_roll_resets() -> None:
    fanout = QueueFanout({"ESH4": "ES", "ESM4": "ES"}, maxsize=8)
    subscriber = DatabentoSubscriber(
        dataset="TEST",
        product_codes=("ESH4", "ESM4"),
        queue_manager=fanout,
        api_key="demo",
    )

    fanout.update_mapping(1, "ESH4")
    symbol_queue = fanout.get_queue("ES")

    minute = dt.datetime(2024, 1, 1, 13, 0, tzinfo=dt.timezone.utc)
    subscriber._apply_trade("ES", _DummyTrade(1, 4005.0, 1.0, minute.replace(second=5)))

    mapping_msg = SymbolMappingMsg(
        publisher_id=1,
        instrument_id=1,
        ts_event=0,
        stype_in=SType.RAW_SYMBOL,
        stype_in_symbol="ESM4",
        stype_out=SType.CONTINUOUS,
        stype_out_symbol="ES",
        start_ts=0,
        end_ts=0,
    )

    subscriber._handle_record(mapping_msg)

    signal = symbol_queue.get_nowait()
    assert isinstance(signal, QueueSignal)
    assert signal.kind == QueueSignal.RESET
    assert subscriber._current_bars == {}
    assert fanout.resolve_root(1) == "ES"


def test_live_data_consumes_queue_signals_and_bars() -> None:
    fanout = QueueFanout({"ESH4": "ES"}, maxsize=8)
    data = DatabentoLiveData(symbol="ES", queue_manager=fanout, backfill=False, qcheck=0.01)

    data.start()
    symbol_queue = fanout.get_queue("ES")
    bar = Bar(
        symbol="ES",
        timestamp=dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc),
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        volume=5.0,
    )

    for line in (
        data.lines.datetime,
        data.lines.open,
        data.lines.high,
        data.lines.low,
        data.lines.close,
        data.lines.volume,
    ):
        line.array.append(0.0)

    fanout.publish_bar(bar)
    assert data._load() is True
    assert data.lines.close[0] == 10.5

    fanout.publish_reset("ES")
    assert data._load() is None
    assert data._latest_dt is None

    fanout.broadcast_shutdown()
    assert data._load() is False
    assert data._stopped is True

    # Queue should be empty after consuming shutdown.
    with pytest.raises(queue.Empty):
        symbol_queue.get_nowait()

    data.stop()
