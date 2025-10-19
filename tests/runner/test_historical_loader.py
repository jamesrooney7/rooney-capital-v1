import types

import pytest

from runner import historical_loader


class DummyTimeseries:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_range(self, *, schema, **kwargs):  # noqa: D401 - simple dummy method
        self.calls.append(schema)
        result = self.responses.get(schema)
        if isinstance(result, Exception):
            raise result
        return result


class DummyHistorical:
    def __init__(self, responses):
        self.timeseries = DummyTimeseries(responses)


@pytest.fixture
def stub_historical(monkeypatch):
    context = types.SimpleNamespace(client=None)

    def factory(responses):
        instance = DummyHistorical(responses)
        context.client = instance
        return instance

    def constructor(api_key):  # noqa: ARG001 - API key unused in stub
        if context.client is None:
            raise RuntimeError("Dummy historical client not initialised")
        return context.client

    monkeypatch.setattr(historical_loader.db, "Historical", constructor)
    return context, factory


def test_load_historical_prefers_aggregated(monkeypatch, stub_historical):
    context, factory = stub_historical
    responses = {"ohlcv-1min": {"schema": "ohlcv-1min"}}
    factory(responses)

    result = historical_loader.load_historical_data(
        api_key="demo", dataset="GLBX.MDP3", symbols=["ES"], stype_in="parent"
    )

    assert result == {"ES": {"schema": "ohlcv-1min"}}
    assert context.client.timeseries.calls == ["ohlcv-1min"]


def test_load_historical_falls_back_to_mbp(monkeypatch, stub_historical):
    context, factory = stub_historical
    responses = {
        "ohlcv-1min": RuntimeError("unsupported schema"),
        "ohlcv-1s": RuntimeError("still unsupported"),
        "mbp-1": {"schema": "mbp-1"},
    }
    factory(responses)

    result = historical_loader.load_historical_data(
        api_key="demo", dataset="GLBX.MDP3", symbols=["ES"], stype_in="parent"
    )

    assert result == {"ES": {"schema": "mbp-1"}}
    assert context.client.timeseries.calls == ["ohlcv-1min", "ohlcv-1s", "mbp-1"]
