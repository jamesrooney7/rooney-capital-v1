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

from models import loader as loader_module  # noqa: E402
from models.loader import load_model_bundle  # noqa: E402
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
    strategy.has_vix = True
    strategy.vix_data = None
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


def test_collect_filter_values_matches_model_bundle(monkeypatch):
    symbol = "6A"

    monkeypatch.setattr(loader_module, "_is_git_lfs_pointer", lambda path: False)
    monkeypatch.setattr(
        loader_module.joblib,
        "load",
        lambda path: {"model": DummyModel([0.1, 0.9]), "features": ()},
    )

    bundle = load_model_bundle(symbol)
    feature_order = []
    seen_features = set()
    for feature in bundle.features:
        normalized_feature = normalize_column_name(feature)
        if normalized_feature not in seen_features:
            feature_order.append(normalized_feature)
            seen_features.add(normalized_feature)

    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    class DummyPercentileTracker:
        def update(self, *args, **kwargs):
            return 0.0

    timeframe_lookup = {"daily": "Day", "hourly": "Hour"}

    cross_meta: dict[str, dict[str, object]] = {}
    return_meta: dict[str, dict[str, object]] = {}
    for feature in feature_order:
        parts = feature.rsplit("_", 2)
        if len(parts) != 3:
            continue
        base, timeframe, suffix = parts
        timeframe_key = timeframe_lookup.get(timeframe)
        if timeframe_key is None:
            continue
        symbol_key = base.upper()
        if suffix in {"z_score", "z_pipeline"}:
            param_key = f"enable{symbol_key}ZScore{timeframe_key}"
            cross_meta.setdefault(
                param_key,
                {
                    "symbol": symbol_key,
                    "timeframe": timeframe_key,
                    "data": None,
                    "line": DummyLine(0.5),
                    "denom": DummyLine(1.25),
                },
            )
        elif suffix in {"return", "return_pipeline"}:
            param_key = f"enable{symbol_key}Return{timeframe_key}"
            return_meta.setdefault(
                param_key,
                {
                    "symbol": symbol_key,
                    "timeframe": timeframe_key,
                    "data": None,
                    "line": DummyLine(0.25),
                    "lookback": 1,
                    "last_dt": None,
                    "last_value": None,
                },
            )

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.percentile_tracker = DummyPercentileTracker()

    dt_num = bt.date2num(datetime(2024, 1, 2, 8, 30))
    strategy.hourly = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = True
    strategy.vix_data = None
    strategy.vix_median = DummyLine(None)
    strategy.dom_threshold = None
    strategy.datr_pct_pct = 55.0
    strategy.hatr_pct_pct = 45.0

    filter_columns = [
        FilterColumn(feature, "", "", f"feature_{feature}") for feature in feature_order
    ]
    strategy.filter_columns = filter_columns
    strategy.filter_keys = {column.parameter for column in filter_columns}
    strategy.filter_column_keys = {column.column_key for column in filter_columns}
    strategy.filter_columns_by_param = {
        column.parameter: [column] for column in filter_columns
    }
    strategy.column_to_param = {
        column.column_key: column.parameter for column in filter_columns
    }

    strategy.cross_zscore_meta = cross_meta
    strategy.return_meta = return_meta

    snapshot = strategy.collect_filter_values()
    normalized_snapshot = {
        normalize_column_name(key): value for key, value in snapshot.items()
    }

    normalized_keys = set(normalized_snapshot.keys())
    missing = seen_features - normalized_keys

    assert not missing, f"Missing features in snapshot: {sorted(missing)}"
