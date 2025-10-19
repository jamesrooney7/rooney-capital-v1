from __future__ import annotations

import datetime as dt
import queue

import pytest

from databento import SType, SymbolMappingMsg

from runner.databento_bridge import DatabentoSubscriber, QueueFanout, QueueSignal


def test_symbol_mapping_roll_emits_single_reset() -> None:
    fanout = QueueFanout({"ESH4": "ES"}, maxsize=4)
    subscriber = DatabentoSubscriber(
        dataset="TEST",
        product_codes=("ESH4",),
        queue_manager=fanout,
        api_key="demo",
    )

    root = "ES"
    minute = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    subscriber._current_bars[root] = {
        "minute": minute,
        "open": 1.0,
        "high": 1.5,
        "low": 0.5,
        "close": 1.25,
        "volume": 10.0,
        "last_ts": minute,
    }
    subscriber._last_emitted_minute[root] = minute
    subscriber._last_close[root] = 1.25
    subscriber._last_trade_ts[root] = minute

    mapping_msg_initial = SymbolMappingMsg(
        publisher_id=1,
        instrument_id=1,
        ts_event=0,
        stype_in=SType.RAW_SYMBOL,
        stype_in_symbol="ESH4",
        stype_out=SType.CONTINUOUS,
        stype_out_symbol="ES",
        start_ts=0,
        end_ts=0,
    )

    subscriber._handle_record(mapping_msg_initial)

    queue_es = fanout.get_queue(root)
    with pytest.raises(queue.Empty):
        queue_es.get_nowait()

    assert subscriber._current_bars[root]["minute"] == minute
    assert subscriber._last_emitted_minute[root] == minute
    assert subscriber._last_close[root] == 1.25
    assert subscriber._last_trade_ts[root] == minute

    mapping_msg_roll = SymbolMappingMsg(
        publisher_id=1,
        instrument_id=2,
        ts_event=0,
        stype_in=SType.RAW_SYMBOL,
        stype_in_symbol="ESH4",
        stype_out=SType.CONTINUOUS,
        stype_out_symbol="ES",
        start_ts=0,
        end_ts=0,
    )

    subscriber._handle_record(mapping_msg_roll)

    signal = queue_es.get_nowait()
    assert isinstance(signal, QueueSignal)
    assert signal.kind == QueueSignal.RESET

    assert subscriber._current_bars == {}
    assert subscriber._last_emitted_minute == {}
    assert subscriber._last_close == {}
    assert subscriber._last_trade_ts == {}

    with pytest.raises(queue.Empty):
        queue_es.get_nowait()

