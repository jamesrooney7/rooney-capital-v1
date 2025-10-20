import sys
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy import ibs_strategy as ibs_module  # noqa: E402
from strategy.ibs_strategy import (  # noqa: E402
    IbsStrategy,
    _metadata_feature_key,
)


class DummyLine:
    def __init__(self, values, timeframe=None):
        self.values = list(values)
        if timeframe is not None:
            self._timeframe = timeframe

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


class DummyData:
    def __init__(self, timeframe, **lines):
        self._timeframe = timeframe
        for name, line in lines.items():
            setattr(self, name, line)

    def __len__(self):  # pragma: no cover - compatibility
        return len(getattr(self, "close", []))


@pytest.fixture
def strategy():
    strat = IbsStrategy.__new__(IbsStrategy)
    strat._ml_feature_snapshot = {}
    strat.ml_feature_collector = strat._ml_feature_snapshot
    strat.percentile_tracker = ibs_module.ExpandingPercentileTracker()
    return strat


def test_cross_zscore_snapshot_publishes_aliases(strategy, monkeypatch):
    feature_key = _metadata_feature_key("VIX", "Day", "z_score")
    pipeline_key = _metadata_feature_key("VIX", "Day", "z_pipeline")
    alias_key = f"{feature_key}_alias"
    pipeline_alias = f"{pipeline_key}_alias"

    monkeypatch.setitem(ibs_module.FEATURE_KEY_ALIASES, feature_key, (alias_key,))
    monkeypatch.setitem(ibs_module.FEATURE_KEY_ALIASES, pipeline_key, (pipeline_alias,))

    price_line = DummyLine([0.25, 0.4], timeframe=bt.TimeFrame.Days)
    denom_line = DummyLine([1.8, 1.5], timeframe=bt.TimeFrame.Days)
    data = DummyData(bt.TimeFrame.Days, datetime=DummyLine([bt.date2num(datetime(2024, 1, 1))]))

    meta = {
        "symbol": "VIX",
        "timeframe": "Day",
        "data": data,
        "line": price_line,
        "denom": denom_line,
        "feature_key": feature_key,
        "pipeline_feature_key": pipeline_key,
    }

    numeric, pipeline_val = strategy._record_cross_zscore_snapshot(meta)

    assert numeric == pytest.approx(0.4)
    assert pipeline_val == pytest.approx(1.5)
    collector = strategy.ml_feature_collector
    assert collector[feature_key] == pytest.approx(0.4)
    assert collector[pipeline_key] == pytest.approx(1.5)
    assert collector[alias_key] == pytest.approx(0.4)
    assert collector[pipeline_alias] == pytest.approx(1.5)

    price_line.values[-1] = 0.55
    denom_line.values[-1] = 1.25
    numeric, pipeline_val = strategy._record_cross_zscore_snapshot(meta)
    assert numeric == pytest.approx(0.55)
    assert pipeline_val == pytest.approx(1.25)
    assert collector[feature_key] == pytest.approx(0.55)
    assert collector[pipeline_key] == pytest.approx(1.25)


def test_cross_return_snapshot_seeds_collector(strategy, monkeypatch):
    feature_key = _metadata_feature_key("TLT", "Day", "return")
    pipeline_key = _metadata_feature_key("TLT", "Day", "return_pipeline")
    alias_key = f"{feature_key}_alias"
    pipeline_alias = f"{pipeline_key}_alias"

    monkeypatch.setitem(ibs_module.FEATURE_KEY_ALIASES, feature_key, (alias_key,))
    monkeypatch.setitem(ibs_module.FEATURE_KEY_ALIASES, pipeline_key, (pipeline_alias,))

    return_line = DummyLine([0.1, 0.2], timeframe=bt.TimeFrame.Days)
    data = DummyData(
        bt.TimeFrame.Days,
        datetime=DummyLine([bt.date2num(datetime(2024, 1, 1))]),
    )

    meta = {
        "symbol": "TLT",
        "timeframe": "Day",
        "data": data,
        "line": return_line,
        "lookback": 1,
        "last_dt": None,
        "last_value": None,
        "feature_key": feature_key,
        "pipeline_feature_key": pipeline_key,
    }

    numeric, pipeline_val = strategy._record_cross_return_snapshot(meta)
    assert numeric == pytest.approx(0.2)
    assert pipeline_val == pytest.approx(0.2)
    collector = strategy.ml_feature_collector
    assert collector[feature_key] == pytest.approx(0.2)
    assert collector[pipeline_key] == pytest.approx(0.2)
    assert collector[alias_key] == pytest.approx(0.2)
    assert collector[pipeline_alias] == pytest.approx(0.2)

    return_line.values[-1] = 0.35
    data.datetime.values[-1] = bt.date2num(datetime(2024, 1, 2))
    numeric, pipeline_val = strategy._record_cross_return_snapshot(meta)
    assert numeric == pytest.approx(0.35)
    assert pipeline_val == pytest.approx(0.35)
    assert collector[feature_key] == pytest.approx(0.35)
    assert collector[pipeline_key] == pytest.approx(0.35)
