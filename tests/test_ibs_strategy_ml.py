import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace

import backtrader as bt
import pandas as pd
import pytest

# Ensure the project "src" directory is on the import path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import loader as loader_module  # noqa: E402
from models.loader import load_model_bundle  # noqa: E402
from strategy.ibs_strategy import IbsStrategy, _metadata_feature_key  # noqa: E402
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
    assert isinstance(strategy.ml_model.seen, pd.DataFrame)
    assert list(strategy.ml_model.seen.columns) == ["prev_bar"]
    assert strategy.ml_model.seen.iloc[0, 0] == pytest.approx(0.42)


def test_evaluate_ml_score_preserves_feature_names(monkeypatch):
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.ml_features = ["prev_bar"]
    strategy.ml_model = DummyModel([0.3, 0.7])
    strategy.ml_model.feature_names_in_ = ["prev_bar"]
    strategy.cross_zscore_meta = set()
    strategy.return_meta = set()

    def fake_collect(self, intraday_ago=0):
        assert intraday_ago == 0
        return {"Prev Bar %": 0.24}

    strategy.collect_filter_values = MethodType(fake_collect, strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = strategy._evaluate_ml_score()

    assert score == pytest.approx(0.7)
    assert not any("feature names" in str(w.message).lower() for w in caught)


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
            "feature_key": _metadata_feature_key(symbol, timeframe, "z_score"),
            "pipeline_feature_key": _metadata_feature_key(
                symbol, timeframe, "z_pipeline"
            ),
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
            "feature_key": _metadata_feature_key(symbol, timeframe, "return"),
            "pipeline_feature_key": _metadata_feature_key(
                symbol, timeframe, "return_pipeline"
            ),
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


def test_collect_filter_values_populates_alias_features():
    class DummyLine:
        def __init__(self, current, prev=None, prev2=None):
            self.current = current
            self.prev = prev if prev is not None else current
            self.prev2 = prev2 if prev2 is not None else self.prev

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            if idx >= 0:
                return self.current
            if idx == -1:
                return self.prev
            if idx == -2:
                return self.prev2
            return self.prev2

    class DummyData:
        def __init__(self, timeframe, **lines):
            self._timeframe = timeframe
            self._lines = lines

        def __len__(self):
            return 10

        def __getattr__(self, name):
            try:
                return self._lines[name]
            except KeyError as exc:  # pragma: no cover - defensive
                raise AttributeError(name) from exc

    class DummyPercentileTracker:
        def __init__(self, mapping):
            self.mapping = mapping

        def update(self, key, value, marker):
            return self.mapping.get(key, 0.5)

    strategy = IbsStrategy.__new__(IbsStrategy)
    percentile_values = {
        "atrz": 0.61,
        "datrz": 0.31,
        "volz": 0.27,
        "dvolz": 0.41,
        "mom3_z": 0.73,
        "daily_rsi": 0.35,
        "rsi": 0.57,
        "rsi2": 0.29,
        "ibs": 0.5,
    }
    strategy.percentile_tracker = DummyPercentileTracker(percentile_values)

    dt_num = bt.date2num(datetime(2024, 1, 2, 8, 45))
    hourly = DummyData(
        "Hourly",
        datetime=DummyLine(dt_num),
        close=DummyLine(105.0, 100.0, 95.0),
        high=DummyLine(110.0, 108.0, 107.0),
        low=DummyLine(100.0, 98.0, 96.0),
        open=DummyLine(101.0, 99.0, 97.0),
    )
    daily = DummyData(
        "Daily",
        close=DummyLine(120.0, 110.0, 100.0),
        high=DummyLine(130.0, 120.0, 110.0),
        low=DummyLine(110.0, 100.0, 90.0),
    )

    strategy.hourly = hourly
    strategy.daily = daily
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_data = None
    strategy.vix_median = DummyLine(None)
    strategy.dom_threshold = None
    strategy.datr_pct_pct = 62.0
    strategy.hatr_pct_pct = 48.0
    strategy.prev_ibs_val = None
    strategy.prev_daily_ibs_val = None
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.p = SimpleNamespace(
        atrTF="Hourly",
        dAtrTF="Daily",
        volTF="Hourly",
        dVolTF="Daily",
        mom3TF="Daily",
        distZTF="Daily",
        dailyRSITF="Daily",
        zScoreTF="Daily",
    )

    atr_data = DummyData("Hourly", close=DummyLine(0.0))
    strategy.atr_z = DummyLine(1.5)
    strategy.atr_z_data = atr_data
    strategy.datr_z = DummyLine(1.2)
    strategy.datr_z_data = DummyData("Daily", close=DummyLine(0.0))
    strategy.vol_z = DummyLine(0.8)
    strategy.vol_z_data = atr_data
    strategy.dvol_z = DummyLine(0.6)
    strategy.dvol_z_data = DummyData("Daily", close=DummyLine(0.0))
    strategy.mom3_z = DummyLine(0.4)
    strategy.mom3_z_data = DummyData("Daily", close=DummyLine(0.0))
    strategy.dist_z = DummyLine(-0.2)
    strategy.dist_z_data = DummyData("Daily", close=DummyLine(0.0))
    strategy.rsi = DummyLine(55.0)
    strategy.rsi2 = DummyLine(52.0)
    strategy.daily_rsi = DummyLine(45.0)
    strategy.daily_rsi_data = daily
    strategy.price_z = DummyLine(0.2)
    strategy.price_z_data = DummyData("Daily", close=DummyLine(0.0))

    params = [
        "enableATRZ",
        "enableDATRZ",
        "enableVolZ",
        "enableDVolZ",
        "enableMom3",
        "enableDistZ",
        "enableDailyRSI",
        "enableRSIEntry",
        "enableRSIEntry2",
        "enableIBSEntry",
        "enablePrevDayPct",
        "enableZScore",
    ]
    filter_columns = [FilterColumn(param, "", "", param) for param in params]
    strategy.filter_columns = filter_columns
    strategy.filter_keys = {col.parameter for col in filter_columns}
    strategy.filter_column_keys = {col.column_key for col in filter_columns}
    strategy.filter_columns_by_param = {col.parameter: [col] for col in filter_columns}
    strategy.column_to_param = {col.column_key: col.parameter for col in filter_columns}

    snapshot = strategy.collect_filter_values()
    normalized = {normalize_column_name(key): value for key, value in snapshot.items()}

    assert normalized["atr_z_percentile"] == pytest.approx(0.61)
    assert normalized["daily_atr_z_percentile"] == pytest.approx(0.31)
    assert normalized["volume_z_percentile"] == pytest.approx(0.27)
    assert normalized["daily_volume_z_percentile"] == pytest.approx(0.41)
    assert normalized["momentum_z_entry_daily"] == pytest.approx(0.4)
    assert normalized["momentum_z_percentile"] == pytest.approx(0.73)
    assert normalized["distance_z_entry_daily"] == pytest.approx(-0.2)
    assert normalized["prev_day_pctxvalue"] == pytest.approx(10.0)
    assert normalized["secondary_rsi_entry_daily"] == pytest.approx(45.0)
    assert normalized["secondary_rsi_percentile"] == pytest.approx(0.29)
    assert normalized["ibsxatrz"] == pytest.approx(0.75)
    assert normalized["ibsxvolz"] == pytest.approx(0.4)
    assert normalized["rsixatrz"] == pytest.approx(82.5)
    assert normalized["rsixvolz"] == pytest.approx(44.0)
    assert normalized["price_usd"] == pytest.approx(105.0)
    assert normalized["price_z_score_daily"] == pytest.approx(0.2)


def test_cross_return_updates_ml_feature_collector():
    class DummyLine:
        def __init__(self, current):
            self.current = current

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.current

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.ml_feature_collector = {}

    initial_dt = bt.date2num(datetime(2024, 1, 2, 9, 0))
    data = SimpleNamespace(datetime=DummyLine(initial_dt))

    meta = {
        "symbol": "CL",
        "timeframe": "Hour",
        "data": data,
        "line": DummyLine(0.42),
        "lookback": 1,
        "last_dt": None,
        "last_value": None,
        "feature_key": "cl_hourly_return",
        "pipeline_feature_key": "cl_hourly_return_pipeline",
    }

    value = strategy._calc_return_value(meta)
    assert value == pytest.approx(0.42)
    assert strategy.ml_feature_collector["cl_hourly_return"] == pytest.approx(0.42)
    assert strategy.ml_feature_collector["cl_hourly_return_pipeline"] == pytest.approx(0.42)

    # Update the underlying line to simulate a new bar
    meta["line"].current = 0.58
    data.datetime.current = bt.date2num(datetime(2024, 1, 2, 10, 0))

    new_value = strategy._calc_return_value(meta)
    assert new_value == pytest.approx(0.58)
    assert strategy.ml_feature_collector["cl_hourly_return"] == pytest.approx(0.58)
    assert strategy.ml_feature_collector["cl_hourly_return_pipeline"] == pytest.approx(0.58)
