from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import backtrader as bt
import pytest
import requests

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "runner" / "traderspost_client.py"
spec = importlib.util.spec_from_file_location("runner.traderspost_client", MODULE_PATH)
assert spec and spec.loader  # for mypy/linters
traderspost_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(traderspost_client)

TradersPostClient = traderspost_client.TradersPostClient
TradersPostError = traderspost_client.TradersPostError
order_notification_to_message = traderspost_client.order_notification_to_message
trade_notification_to_message = traderspost_client.trade_notification_to_message


class DummyStrategy:
    def __init__(self, symbol: str = "ES") -> None:
        self.p = SimpleNamespace(symbol=symbol)
        self.current_signal = "entry"
        self._ml_last_score = 0.42


def _dummy_order(is_buy: bool = True) -> SimpleNamespace:
    info = {
        "created": datetime(2024, 1, 2, 13, 0),
        "filter_snapshot": {"ibs": 0.12, "ml_score": 0.87},
        "ibs": 0.12,
    }
    executed = SimpleNamespace(size=2, price=4321.5)
    data = SimpleNamespace(_name="ES_hour")
    order = SimpleNamespace(
        status=bt.Order.Completed,
        info=info,
        executed=executed,
        data=data,
        ref=99,
    )
    order.isbuy = (lambda: True) if is_buy else (lambda: False)
    return order


def _dummy_trade(size: int = -2) -> SimpleNamespace:
    data = SimpleNamespace(_name="ES_hour")
    trade = SimpleNamespace(
        isclosed=True,
        data=data,
        price=4325.0,
        pnl=150.0,
        pnlcomm=142.5,
        size=size,
    )
    return trade


def test_order_notification_payload_includes_thresholds_and_metadata() -> None:
    strategy = DummyStrategy()
    order = _dummy_order()

    payload = order_notification_to_message(strategy, order)

    assert payload is not None
    assert payload["symbol"] == "ES"
    assert payload["side"] == "buy"
    assert payload["size"] == 2
    assert payload["thresholds"]["ml_score"] == pytest.approx(0.87)
    assert payload["metadata"]["ibs_value"] == pytest.approx(0.12)
    assert "created" in payload["metadata"]


def test_trade_notification_uses_exit_snapshot() -> None:
    strategy = DummyStrategy()
    trade = _dummy_trade()
    exit_snapshot = {
        "size": -2,
        "price": 4330.25,
        "exit_reason": "take profit",
        "filter_snapshot": {"ibs": 0.92, "ml_score": 0.33},
        "ibs_value": 0.92,
    }

    payload = trade_notification_to_message(strategy, trade, exit_snapshot)

    assert payload is not None
    assert payload["symbol"] == "ES"
    assert payload["side"] == "sell"
    assert payload["size"] == -2
    assert payload["price"] == pytest.approx(4330.25)
    assert payload["thresholds"]["ml_score"] == pytest.approx(0.33)
    assert payload["metadata"]["exit_reason"] == "take profit"
    assert payload["metadata"]["pnl"] == pytest.approx(142.5)


def test_client_retries_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = mock.Mock()
    first = mock.Mock()
    first.status_code = 502
    first.text = "upstream down"
    first.raise_for_status.side_effect = requests.HTTPError(response=first)
    second = mock.Mock()
    second.status_code = 200
    second.raise_for_status.return_value = None
    session.post.side_effect = [first, second]

    sleeps: list[float] = []
    monkeypatch.setattr(traderspost_client.time, "sleep", lambda s: sleeps.append(s))

    client = TradersPostClient("https://example.com/hook", session=session, max_retries=2, backoff_factor=0.25)
    payload = {"symbol": "ES", "side": "buy", "size": 1, "thresholds": {}, "metadata": {}}

    client.post_order(payload)

    assert session.post.call_count == 2
    assert sleeps == [0.25]
    last_call = session.post.call_args_list[-1]
    assert last_call.kwargs["json"]["event"] == "order"
    assert last_call.kwargs["json"]["symbol"] == "ES"


def test_client_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    session = mock.Mock()
    error = requests.ConnectionError("boom")
    session.post.side_effect = error

    sleeps: list[float] = []
    monkeypatch.setattr(traderspost_client.time, "sleep", lambda s: sleeps.append(s))

    client = TradersPostClient("https://example.com/hook", session=session, max_retries=2, backoff_factor=0.1)

    with pytest.raises(TradersPostError):
        client.post_trade({"symbol": "ES", "side": "sell", "size": -1, "thresholds": {}, "metadata": {}})

    assert session.post.call_count == 3
    assert sleeps == [0.1, 0.2]


def test_client_does_not_retry_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = mock.Mock()
    response = mock.Mock()
    response.status_code = 401
    response.text = "unauthorised"
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    session.post.return_value = response

    client = TradersPostClient("https://example.com/hook", session=session)

    with pytest.raises(TradersPostError):
        client.post_order({"symbol": "ES", "side": "buy", "size": 1, "thresholds": {}, "metadata": {}})

    assert session.post.call_count == 1
