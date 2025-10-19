import pandas as pd

from runner.live_worker import LiveWorker


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
