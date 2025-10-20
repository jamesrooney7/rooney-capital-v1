import types

import pytest

from strategy.ibs_strategy import IbsStrategy


class _DummyModel:
    classes_ = [0, 1]

    def predict_proba(self, payload):
        try:
            length = len(payload)
        except Exception:
            length = 1
        if length <= 0:
            length = 1
        return [[0.25, 0.75] for _ in range(length)]


class _DummyLine:
    def __setitem__(self, index, value):
        return None


class _DummyData:
    def __init__(self, length: int):
        self._length = length
        self.close = self

    def __len__(self):
        return self._length

    def set_length(self, length: int) -> None:
        self._length = length


@pytest.fixture
def strategy(monkeypatch):
    instance = IbsStrategy.__new__(IbsStrategy)
    instance._periods = []
    instance.cross_zscore_meta = {}
    instance.return_meta = {}
    instance._ml_feature_snapshot = {}
    instance._ml_warmup_pending = {}
    instance.ml_feature_collector = instance._ml_feature_snapshot
    instance.ml_model = _DummyModel()
    instance.ml_threshold = None
    instance.ml_features = ("es_hour_zscore_pipeline",)
    instance._normalized_ml_features = ("es_hour_zscore_pipeline",)
    instance.percentile_tracker = types.SimpleNamespace(update=lambda *args, **kwargs: 0.0)
    instance.collect_filter_values = types.MethodType(
        lambda self, intraday_ago=0: dict(self._ml_feature_snapshot),
        instance,
    )

    call_counts = {"line": 0, "denom": 0}

    def fake_timeframed_line_val(line, **kwargs):
        call_counts[line] = call_counts.get(line, 0) + 1
        return 1.0 if line == "line" else 2.0

    monkeypatch.setattr(
        "strategy.ibs_strategy.timeframed_line_val",
        fake_timeframed_line_val,
    )

    instance._call_counts = call_counts
    return instance


def test_cross_zscore_warmup_blocks_ml_scoring(strategy):
    dummy_data = _DummyData(length=10)
    meta = {
        "symbol": "ES",
        "timeframe": "Hourly",
        "feed_suffix": "hour",
        "len": 20,
        "window": 40,
        "data": dummy_data,
        "feature_key": "es_hour_zscore",
        "pipeline_feature_key": "es_hour_zscore_pipeline",
        "line": "line",
        "denom": "denom",
    }

    numeric, pipeline_val = strategy._record_cross_zscore_snapshot(meta)
    assert numeric is None
    assert pipeline_val is None
    assert strategy._ml_warmup_pending
    assert strategy._ml_feature_snapshot.get("es_hour_zscore_pipeline") is None
    assert strategy._call_counts["line"] == 0

    score_during_warmup = strategy._evaluate_ml_score()
    assert score_during_warmup is None

    dummy_data.set_length(80)
    numeric_ready, pipeline_ready = strategy._record_cross_zscore_snapshot(meta)
    assert numeric_ready == pytest.approx(1.0)
    assert pipeline_ready == pytest.approx(2.0)
    assert not strategy._ml_warmup_pending
    assert strategy._ml_feature_snapshot["es_hour_zscore_pipeline"] == pytest.approx(2.0)
    assert strategy._call_counts["line"] > 0
    assert strategy._call_counts["denom"] > 0

    score_after_warmup = strategy._evaluate_ml_score()
    assert score_after_warmup is not None
    assert 0.0 <= score_after_warmup <= 1.0


