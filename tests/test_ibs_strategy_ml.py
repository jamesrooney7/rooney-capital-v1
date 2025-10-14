import json
import sys
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace

import backtrader as bt
import pytest

# Ensure the project "src" directory is on the import path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
STRATEGY_SRC = SRC / "strategy"
if str(STRATEGY_SRC) not in sys.path:
    sys.path.insert(0, str(STRATEGY_SRC))

from strategy.ibs_strategy import IbsStrategy  # noqa: E402
from strategy.filter_column import FilterColumn  # noqa: E402
from strategy.feature_utils import normalize_column_name  # noqa: E402


class DummyModel:
    def __init__(self, result):
        self.result = result
        self.seen = None
        self.classes_ = [0, 1]

    def predict_proba(self, rows):
        self.seen = rows
        return [self.result]


def test_evaluate_ml_score_uses_normalised_feature_names():
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.ml_features = ["prev_bar"]
    strategy.ml_model = DummyModel([0.2, 0.8])
    strategy.cross_zscore_meta = set()
    strategy.return_meta = set()

    def fake_collect(self, intraday_ago=0):
        assert intraday_ago == 0
        return {"Prev Bar %": 0.42}

    strategy.collect_filter_values = MethodType(fake_collect, strategy)

    score = strategy._evaluate_ml_score()

    assert score == pytest.approx(0.8)
    assert strategy.ml_model.seen == [[0.42]]


def test_collect_filter_values_emits_metadata_keys():
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    class DummyPercentileTracker:
        def update(self, *args, **kwargs):
            return None

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.percentile_tracker = DummyPercentileTracker()
    dt_num = bt.date2num(datetime(2024, 1, 2, 8, 30))
    strategy.hourly = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.vix_median = DummyLine(None)
    strategy.dom_threshold = None
    strategy.datr_pct_pct = 55.0
    strategy.hatr_pct_pct = 45.0
    columns = [
        FilterColumn("enableOpenClose", "", "", "enableOpenClose"),
        FilterColumn(
            "enableDailyATRPercentile",
            "",
            "",
            "enableDailyATRPercentile",
        ),
        FilterColumn(
            "enableHourlyATRPercentile",
            "",
            "",
            "enableHourlyATRPercentile",
        ),
    ]
    strategy.filter_columns = columns
    strategy.filter_keys = {col.parameter for col in columns}
    strategy.filter_column_keys = {col.column_key for col in columns}
    strategy.filter_columns_by_param = {col.parameter: [col] for col in columns}
    strategy.column_to_param = {col.column_key: col.parameter for col in columns}

    def make_cross_meta(symbol, timeframe, value, denom):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": None,
            "line": DummyLine(value),
            "denom": DummyLine(denom),
        }

    cross_specs = {
        "enable6MZScoreDay": ("6M", "Day", 0.5, 2.0),
        "enable6BZScoreHour": ("6B", "Hour", -0.1, 1.5),
        "enableRTYZScoreHour": ("RTY", "Hour", 0.2, 1.2),
        "enable6SZScoreDay": ("6S", "Day", -0.4, 1.8),
        "enablePLZScoreHour": ("PL", "Hour", 0.8, 1.6),
        "enableNGZScoreHour": ("NG", "Hour", 1.1, 1.7),
        "enableNGZScoreDay": ("NG", "Day", -0.3, 1.9),
        "enableVIXZScoreHour": ("VIX", "Hour", 0.6, 1.4),
    }
    strategy.cross_zscore_meta = {
        key: make_cross_meta(*spec) for key, spec in cross_specs.items()
    }

    def make_return_meta(symbol, timeframe, value):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": None,
            "line": DummyLine(value),
            "lookback": 1,
            "last_dt": None,
            "last_value": None,
        }

    return_specs = {
        "enableSIReturnDay": ("SI", "Day", 0.25),
        "enable6JReturnHour": ("6J", "Hour", -0.15),
        "enable6NReturnDay": ("6N", "Day", 0.33),
        "enableNQReturnHour": ("NQ", "Hour", 0.41),
        "enable6MReturnDay": ("6M", "Day", -0.27),
        "enable6EReturnDay": ("6E", "Day", 0.11),
        "enableHGReturnDay": ("HG", "Day", 0.52),
        "enableVIXReturnDay": ("VIX", "Day", -0.38),
    }
    strategy.return_meta = {
        key: make_return_meta(*spec) for key, spec in return_specs.items()
    }

    snapshot = strategy.collect_filter_values()
    normalized = {
        normalize_column_name(key): value for key, value in snapshot.items()
    }

    metadata_path = SRC / "models" / "6A_best.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        features = json.load(handle)["Features"]

    expected = {
        feat
        for feat in features
        if feat in {"open_close", "daily_atr_percentile"}
        or feat.endswith("_z_score")
        or feat.endswith("_z_pipeline")
        or feat.endswith("_return")
        or feat.endswith("_return_pipeline")
    }

    assert expected <= set(normalized.keys())
