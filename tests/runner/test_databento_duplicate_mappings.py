from __future__ import annotations

import datetime as dt
import queue

import pytest

from databento import SType, SymbolMappingMsg

from runner.databento_bridge import DatabentoSubscriber, QueueFanout, QueueSignal


def _mapping_message(
    instrument_id: int,
    stype_in_symbol: str,
    stype_out_symbol: str,
) -> SymbolMappingMsg:
    return SymbolMappingMsg(
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=0,
        stype_in=SType.RAW_SYMBOL,
        stype_in_symbol=stype_in_symbol,
        stype_out=SType.CONTINUOUS,
        stype_out_symbol=stype_out_symbol,
        start_ts=0,
        end_ts=0,
    )


def _build_subscriber(product_to_root: dict[str, str]) -> tuple[QueueFanout, DatabentoSubscriber]:
    fanout = QueueFanout(product_to_root, maxsize=4)
    subscriber = DatabentoSubscriber(
        dataset="TEST",
        product_codes=tuple(product_to_root.keys()),
        queue_manager=fanout,
        api_key="demo",
    )
    return fanout, subscriber


def test_symbol_mapping_initial_mapping_skips_reset() -> None:
    fanout, subscriber = _build_subscriber({"ESH4": "ES"})

    subscriber._handle_record(_mapping_message(1, "ESH4", "ES"))

    queue_es = fanout.get_queue("ES")
    with pytest.raises(queue.Empty):
        queue_es.get_nowait()

    assert subscriber._instrument_roots[1] == "ES"


def test_symbol_mapping_duplicate_mapping_skips_reset() -> None:
    fanout, subscriber = _build_subscriber({"ESH4": "ES"})

    message = _mapping_message(1, "ESH4", "ES")
    subscriber._handle_record(message)
    subscriber._handle_record(message)

    queue_es = fanout.get_queue("ES")
    with pytest.raises(queue.Empty):
        queue_es.get_nowait()

    assert subscriber._instrument_roots[1] == "ES"


def test_symbol_mapping_root_change_triggers_reset() -> None:
    fanout, subscriber = _build_subscriber({"ESH4": "ES", "NQH4": "NQ"})

    queue_es = fanout.get_queue("ES")
    queue_nq = fanout.get_queue("NQ")

    subscriber._handle_record(_mapping_message(1, "ESH4", "ES"))

    with pytest.raises(queue.Empty):
        queue_es.get_nowait()
    with pytest.raises(queue.Empty):
        queue_nq.get_nowait()

    minute = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    subscriber._current_bars["NQ"] = {
        "minute": minute,
        "open": 1.0,
        "high": 1.5,
        "low": 0.5,
        "close": 1.25,
        "volume": 10.0,
        "last_ts": minute,
    }
    subscriber._last_emitted_minute["NQ"] = minute
    subscriber._last_close["NQ"] = 1.25
    subscriber._last_trade_ts["NQ"] = minute

    subscriber._handle_record(_mapping_message(1, "NQH4", "NQ"))

    signal = queue_nq.get_nowait()
    assert isinstance(signal, QueueSignal)
    assert signal.kind == QueueSignal.RESET

    assert subscriber._current_bars.get("NQ") is None
    assert subscriber._last_emitted_minute.get("NQ") is None
    assert subscriber._last_close.get("NQ") is None
    assert subscriber._last_trade_ts.get("NQ") is None
    assert subscriber._instrument_roots[1] == "NQ"

    with pytest.raises(queue.Empty):
        queue_nq.get_nowait()
