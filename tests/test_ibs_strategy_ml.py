import json
import logging
import sys
import math
import numpy as np
import warnings
from datetime import datetime, timedelta
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
from strategy import ibs_strategy as ibs_module  # noqa: E402
from strategy.ibs_strategy import (  # noqa: E402
    IbsStrategy,
    _metadata_feature_key,
    clamp_period,
)
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


def test_ml_features_request_cross_zscore_populates_snapshot():
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    class DummyPercentileTracker:
        def update(self, *_args, **_kwargs):
            return None

    dt_num = bt.date2num(datetime(2024, 1, 2, 8, 30))

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.percentile_tracker = DummyPercentileTracker()
    strategy.hourly = SimpleNamespace(
        datetime=DummyLine(dt_num),
        close=DummyLine(100.0),
    )
    strategy.daily = SimpleNamespace(close=DummyLine(100.0))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_data = None
    strategy.vix_median = DummyLine(None)
    strategy.dom_threshold = None
    strategy.datr_pct_pct = 55.0
    strategy.hatr_pct_pct = 45.0
    strategy.filter_columns = []
    strategy.filter_keys = set()
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy._cross_feature_enable_lookup = {}

    meta = {
        "symbol": "TLT",
        "timeframe": "Day",
        "data": None,
        "line": DummyLine(0.75),
        "denom": DummyLine(1.5),
        "feature_key": _metadata_feature_key("TLT", "Day", "z_score"),
        "pipeline_feature_key": _metadata_feature_key("TLT", "Day", "z_pipeline"),
    }
    strategy._register_cross_feature_meta(
        strategy.cross_zscore_meta, "enableTLTZScoreDay", meta
    )

    strategy.ml_features = ("tlt_daily_z_score",)
    strategy._normalized_ml_features = (
        normalize_column_name("tlt_daily_z_score"),
    )
    strategy.ml_feature_param_keys = strategy._derive_ml_feature_param_keys()

    assert "enableTLTZScoreDay" in strategy.ml_feature_param_keys

    snapshot = strategy.collect_filter_values()
    normalized = {normalize_column_name(key): value for key, value in snapshot.items()}

    assert normalized["tlt_daily_z_score"] == pytest.approx(0.75)
    assert normalized["tlt_daily_z_pipeline"] == pytest.approx(1.5)


def test_cross_zscore_recovers_after_missing_feed(monkeypatch):
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.p = SimpleNamespace()
    strategy._periods = []
    strategy.max_period = 0
    strategy.cross_zscore_cache = {}
    strategy.cross_data_cache = {}
    strategy.cross_data_missing = set()
    strategy._ml_feature_snapshot = {}
    strategy.ml_feature_collector = {}
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}

    published: dict[str, list[float | None]] = {}

    def fake_publish(self, key, value, _propagating=False):
        if isinstance(key, str):
            published.setdefault(key, []).append(value)

    strategy._publish_ml_feature = MethodType(fake_publish, strategy)

    symbol = "6E"
    timeframe = "Hour"
    enable_param = "enable6EZScoreHour"
    feature_key = _metadata_feature_key(symbol, timeframe, "z_score")
    pipeline_key = _metadata_feature_key(symbol, timeframe, "z_pipeline")
    meta: dict[str, object] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "feed_suffix": "hour",
        "len_param": "sixEZLenHour",
        "window_param": "sixEZWindowHour",
        "len_aliases": ("sixEZLenHour",),
        "window_aliases": ("sixEZWindowHour",),
        "feature_key": feature_key,
        "pipeline_feature_key": pipeline_key,
        "enable_param": enable_param,
        "line": None,
        "data": None,
        "denom": None,
    }
    strategy.cross_zscore_meta[enable_param] = meta

    setattr(strategy.p, "sixEZLenHour", 20)
    setattr(strategy.p, "sixEZWindowHour", 252)

    feed = SimpleNamespace(close=object())
    sentinel_line = object()
    sentinel_denom = object()
    get_calls = {"count": 0}
    build_calls = {"count": 0}
    added_periods: list[int] = []

    def fake_get(self, sym, suffix, enable):
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            return None
        assert sym == symbol
        assert suffix == "hour"
        assert enable == enable_param
        return feed

    def fake_build(
        self,
        sym,
        tf,
        suffix,
        length,
        window,
        data_feed=None,
    ):
        build_calls["count"] += 1
        assert sym == symbol
        assert tf == timeframe
        assert suffix == "hour"
        assert data_feed is feed
        assert length == clamp_period(20)
        assert window == clamp_period(252)
        return {
            "line": sentinel_line,
            "mean": object(),
            "std": object(),
            "denom": sentinel_denom,
            "data": data_feed,
        }

    def fake_addminperiod(self, period):
        added_periods.append(period)

    monkeypatch.setattr(ibs_module, "timeframed_line_val", lambda line, **_: 1.23 if line is sentinel_line else (2.34 if line is sentinel_denom else None))

    strategy._get_cross_feed = MethodType(fake_get, strategy)
    strategy._build_cross_zscore_pipeline = MethodType(fake_build, strategy)
    strategy.addminperiod = MethodType(fake_addminperiod, strategy)

    result_first = strategy._record_cross_zscore_snapshot(meta)
    assert result_first == (None, None)
    assert meta["line"] is None
    assert build_calls["count"] == 0

    result_second = strategy._record_cross_zscore_snapshot(meta)
    assert result_second == (pytest.approx(1.23), pytest.approx(2.34))
    assert meta["line"] is sentinel_line
    assert meta["denom"] is sentinel_denom
    assert meta["data"] is feed
    assert build_calls["count"] == 1
    assert get_calls["count"] >= 2
    assert strategy._periods[-2:] == [clamp_period(20), clamp_period(252)]
    assert strategy.max_period == clamp_period(252)
    assert added_periods == [clamp_period(252)]
    assert published[feature_key][-1] == pytest.approx(1.23)
    assert published[pipeline_key][-1] == pytest.approx(2.34)


def test_derive_ml_feature_keys_adds_enable_atrz_for_ibsxatrz():
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.ml_features = ("ibsxatrz",)
    strategy.filter_keys = set()
    strategy.filter_columns = []
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}

    derived = strategy._derive_ml_feature_param_keys()

    assert "enableATRZ" in derived
    assert "ibs" in derived


def test_derive_ml_feature_keys_adds_enable_volz_for_ibsxvolz():
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.ml_features = ("ibsxvolz",)
    strategy.filter_keys = set()
    strategy.filter_columns = []
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}

    derived = strategy._derive_ml_feature_param_keys()

    assert "enableVolZ" in derived
    assert "ibs" in derived


def test_collect_filter_values_exposes_ibsx_combos_from_full_strategy():
    hourly_index = pd.date_range("2024-01-01", periods=200, freq="h")
    hourly_base = 100 + np.linspace(0, 5, len(hourly_index))
    hourly_df = pd.DataFrame(
        {
            "open": hourly_base,
            "high": hourly_base + 0.5,
            "low": hourly_base - 0.5,
            "close": hourly_base + 0.25,
            "volume": 1_000 + np.arange(len(hourly_index)) * 10,
        },
        index=hourly_index,
    )

    daily_index = pd.date_range("2023-12-01", periods=60, freq="D")
    daily_base = 110 + np.linspace(0, 3, len(daily_index))
    daily_df = pd.DataFrame(
        {
            "open": daily_base,
            "high": daily_base + 1.0,
            "low": daily_base - 1.0,
            "close": daily_base + 0.5,
            "volume": 2_000 + np.arange(len(daily_index)) * 5,
        },
        index=daily_index,
    )

    tlt_index = pd.date_range("2023-12-01", periods=60, freq="D")
    tlt_base = 90 + np.linspace(0, 2, len(tlt_index))
    tlt_df = pd.DataFrame(
        {
            "open": tlt_base,
            "high": tlt_base + 0.75,
            "low": tlt_base - 0.75,
            "close": tlt_base + 0.25,
            "volume": 1_500 + np.arange(len(tlt_index)) * 3,
        },
        index=tlt_index,
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=hourly_df,
            timeframe=bt.TimeFrame.Minutes,
            compression=60,
        ),
        name="ES_hour",
    )
    cerebro.adddata(bt.feeds.PandasData(dataname=daily_df), name="ES_day")
    cerebro.adddata(bt.feeds.PandasData(dataname=tlt_df), name="TLT_day")

    cerebro.addstrategy(
        IbsStrategy,
        symbol="ES",
        ml_features=("ibsxatrz", "ibsxvolz"),
        atrLen=3,
        atrWindow=5,
        volLen=3,
        volWindow=5,
    )

    strategies = cerebro.run(maxcpus=1, runonce=False)
    strategy = strategies[0]

    snapshot = strategy.collect_filter_values()
    normalized = {
        normalize_column_name(key): value for key, value in snapshot.items()
    }

    ibsxatrz = normalized.get("ibsxatrz")
    ibsxvolz = normalized.get("ibsxvolz")

    assert ibsxatrz is not None and not math.isnan(ibsxatrz)
    assert ibsxvolz is not None and not math.isnan(ibsxvolz)


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


def test_collect_filter_values_populates_ml_features_without_columns():
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    class DummyPercentileTracker:
        def update(self, *args, **kwargs):
            return 0.5

    dt_num = bt.date2num(datetime(2024, 1, 2, 8, 15))

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.percentile_tracker = DummyPercentileTracker()
    strategy.hourly = SimpleNamespace(
        datetime=DummyLine(dt_num),
        close=DummyLine(105.0),
        high=DummyLine(106.0),
        low=DummyLine(104.0),
    )
    strategy.daily = SimpleNamespace(
        close=DummyLine(100.0),
        high=DummyLine(110.0),
        low=DummyLine(95.0),
    )
    strategy.signal_data = SimpleNamespace(close=DummyLine(105.0))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_data = None
    strategy.vix_median = DummyLine(None)
    strategy.dom_threshold = None
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.filter_columns = []
    strategy.filter_keys = set()
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}
    strategy.ml_feature_collector = {}
    strategy.ml_features = (
        "open_close",
        "prev_day_pctxvalue",
        "secondary_rsi_entry_daily",
    )
    strategy.prev_day_pct = MethodType(lambda self: 10.0, strategy)
    strategy.daily_rsi = DummyLine(45.0)
    strategy.daily_rsi_data = SimpleNamespace()
    strategy.p = SimpleNamespace(dailyRSITF="Daily")
    strategy.ml_feature_param_keys = strategy._derive_ml_feature_param_keys()

    snapshot = strategy.collect_filter_values()
    normalized = {normalize_column_name(key): value for key, value in snapshot.items()}

    assert normalized["open_close"] == 1
    assert normalized["prev_day_pctxvalue"] == pytest.approx(10.0)
    assert normalized["secondary_rsi_entry_daily"] == pytest.approx(45.0)


@pytest.mark.parametrize("mode", ["ml", "filters"])
def test_collect_filter_values_populates_requested_optional_percentile_features(mode):
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

    class EchoPercentileTracker:
        def update(self, *_args, **_kwargs):
            return _args[1]

    strategy = IbsStrategy.__new__(IbsStrategy)
    dt_num = bt.date2num(datetime(2024, 1, 3, 9, 30))
    hourly = DummyData(
        "Hourly",
        datetime=DummyLine(dt_num),
        close=DummyLine(102.0),
        high=DummyLine(104.0),
        low=DummyLine(100.0),
    )
    daily = DummyData(
        "Daily",
        close=DummyLine(100.0, 98.0, 96.0),
        high=DummyLine(110.0, 108.0, 106.0),
        low=DummyLine(90.0, 88.0, 86.0),
    )

    strategy.percentile_tracker = EchoPercentileTracker()
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
    strategy.datr_pct_pct = 55.0
    strategy.hatr_pct_pct = 45.0
    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy.ml_feature_collector = {}
    strategy._ml_feature_snapshot = {}

    strategy.atr_z = DummyLine(1.1)
    strategy.atr_z_data = DummyData("Hourly", close=DummyLine(0.0))
    strategy.vol_z = DummyLine(0.9)
    strategy.vol_z_data = DummyData("Hourly", close=DummyLine(0.0))
    strategy.rsi = DummyLine(45.0)
    strategy.daily_rsi = DummyLine(40.0)
    strategy.daily_rsi_data = daily

    strategy.ma_spread_fast = DummyLine(105.0)
    strategy.ma_spread_slow = DummyLine(100.0)
    strategy.ma_spread_atr = DummyLine(2.0)
    strategy.ma_spread_data = DummyData("Hourly", close=DummyLine(0.0))

    strategy.donch_prox_data = DummyData("Hourly", close=DummyLine(100.0))
    strategy.donch_prox_high = DummyLine(110.0, 110.0, 110.0)
    strategy.donch_prox_low = DummyLine(90.0, 90.0, 90.0)
    strategy.donch_prox_atr = DummyLine(10.0)

    strategy.bbw = SimpleNamespace(
        lines=SimpleNamespace(
            top=DummyLine(110.0),
            bot=DummyLine(90.0),
            mid=DummyLine(100.0),
        )
    )
    strategy.bbw_data = DummyData("Hourly", close=DummyLine(100.0))

    strategy.filter_columns = []
    strategy.filter_keys = set()
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}

    features = (
        "atrz_pct",
        "volz_pct",
        "rsi_pct",
        "daily_rsi_pct",
        "ma_spread_ribbon_tightness",
        "donchian_proximity_to_nearest_band",
        "bollinger_bandwidth_daily",
    )

    if mode == "ml":
        strategy.ml_features = features
        strategy.ml_feature_param_keys = strategy._derive_ml_feature_param_keys()
    else:
        params = [
            "enableATRZ",
            "enableVolZ",
            "enableRSIEntry",
            "enableDailyRSI",
            "enableMASpread",
            "enableDonchProx",
            "enableBBW",
        ]
        filter_columns = [FilterColumn(param, "", "", param) for param in params]
        strategy.filter_columns = filter_columns
        strategy.filter_keys = {col.parameter for col in filter_columns}
        strategy.filter_column_keys = {col.column_key for col in filter_columns}
        strategy.filter_columns_by_param = {col.parameter: [col] for col in filter_columns}
        strategy.column_to_param = {col.column_key: col.parameter for col in filter_columns}
        strategy.ml_features = ()
        strategy.ml_feature_param_keys = set()

    strategy.p = SimpleNamespace(
        atrTF="Hourly",
        volTF="Hourly",
        maSpreadTF="Hourly",
        maSpreadUseATR=True,
        donchProxTF="Hourly",
        bbwTF="Hourly",
        dailyRSITF="Daily",
    )

    snapshot = strategy.collect_filter_values()
    normalized = {normalize_column_name(key): value for key, value in snapshot.items()}

    assert normalized["atrz"] == pytest.approx(1.1)
    assert normalized["atrz_pct"] == pytest.approx(1.1)
    assert normalized["volz"] == pytest.approx(0.9)
    assert normalized["volz_pct"] == pytest.approx(0.9)
    assert normalized["rsi"] == pytest.approx(45.0)
    assert normalized["rsi_pct"] == pytest.approx(45.0)
    assert normalized["daily_rsi"] == pytest.approx(40.0)
    assert normalized["daily_rsi_pct"] == pytest.approx(40.0)
    assert normalized["ma_spread_ribbon_tightness"] == pytest.approx(2.5)
    assert normalized["donchian_proximity_to_nearest_band"] == pytest.approx(1.0)
    assert normalized["bollinger_bandwidth_daily"] == pytest.approx(0.2)

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


def test_collect_filter_values_updates_ml_collector(monkeypatch):
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    def fake_line_val(line, ago=0):
        if line is None:
            return None
        return getattr(line, "value", line)

    def fake_timeframed_line_val(line, *, data=None, timeframe=None, daily_ago=-1, intraday_ago=0):
        return fake_line_val(line)

    monkeypatch.setattr(ibs_module, "line_val", fake_line_val)
    monkeypatch.setattr(ibs_module, "timeframed_line_val", fake_timeframed_line_val)

    class EchoPercentileTracker:
        def update(self, *_args, **_kwargs):
            return _args[1] if len(_args) > 1 else None

    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.percentile_tracker = EchoPercentileTracker()
    dt_num = bt.date2num(datetime(2024, 1, 2, 9, 0))
    daily_dt = bt.date2num(datetime(2024, 1, 1))

    hourly = SimpleNamespace(
        datetime=DummyLine(dt_num),
        close=DummyLine(105.0),
        high=DummyLine(110.0),
        low=DummyLine(100.0),
    )
    strategy.hourly = hourly
    strategy.prev_bar_pct_data = hourly
    strategy.daily = SimpleNamespace(datetime=DummyLine(daily_dt), close=DummyLine(100.0))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_median = None
    strategy.vix_data = None
    strategy.dom_threshold = None

    strategy.vol_z = DummyLine(0.27)
    strategy.vol_z_data = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.atr_z = DummyLine(0.5)
    strategy.atr_z_data = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.rsi = DummyLine(0.4)

    strategy.prev_day_pct = MethodType(lambda self: 0.34, strategy)
    strategy.prev_bar_pct = MethodType(lambda self: 0.12, strategy)
    strategy.ibs = MethodType(lambda self: 0.55, strategy)

    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy._ml_feature_snapshot = {}
    strategy.ml_feature_collector = {}

    params = [
        "enablePrevDayPct",
        "enablePrevBarPct",
        "enableIBSEntry",
        "enableATRZ",
        "enableVolZ",
        "enableRSIEntry",
    ]
    filter_columns = [FilterColumn(param, "", "", param) for param in params]
    strategy.filter_columns = filter_columns
    strategy.filter_keys = {col.parameter for col in filter_columns}
    strategy.filter_column_keys = {col.column_key for col in filter_columns}
    strategy.filter_columns_by_param = {col.parameter: [col] for col in filter_columns}
    strategy.column_to_param = {col.column_key: col.parameter for col in filter_columns}

    strategy.p = SimpleNamespace(
        prev_bar_pct_tf="Hour",
        atrTF="Hour",
        volTF="Hour",
    )

    strategy.collect_filter_values()
    collector = strategy.ml_feature_collector

    expected_ibs = 0.55
    expected_volz = 0.27
    expected_atrz = 0.5
    expected_rsi = 0.4

    assert collector["price_usd"] == pytest.approx(105.0)
    assert collector["prev_day_pctxvalue"] == pytest.approx(0.34)
    assert collector["prev_bar_pct"] == pytest.approx(0.12)
    assert collector["prev_bar_pct_pct"] == pytest.approx(0.12)
    assert collector["ibs_pct"] == pytest.approx(expected_ibs)
    assert collector["ibs_percentile"] == pytest.approx(expected_ibs)
    assert collector["volz_pct"] == pytest.approx(expected_volz)
    assert collector["volume_z_percentile"] == pytest.approx(expected_volz)
    assert collector["atrz_pct"] == pytest.approx(expected_atrz)
    assert collector["atr_z_percentile"] == pytest.approx(expected_atrz)
    assert collector["rsi_pct"] == pytest.approx(expected_rsi)
    assert collector["ibsxvolz"] == pytest.approx(expected_ibs * expected_volz)
    assert collector["ibsxatrz"] == pytest.approx(expected_ibs * expected_atrz)
    assert collector["rsixvolz"] == pytest.approx(expected_rsi * expected_volz)
    assert collector["rsixatrz"] == pytest.approx(expected_rsi * expected_atrz)

    for key, value in collector.items():
        if value is not None:
            assert not isinstance(value, float) or not math.isnan(value)


def test_collect_filter_values_populates_ibsxatrz_for_ml_features(monkeypatch):
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    def fake_line_val(line, ago=0):
        if line is None:
            return None
        return getattr(line, "value", line)

    def fake_timeframed_line_val(
        line, *, data=None, timeframe=None, daily_ago=-1, intraday_ago=0
    ):
        return fake_line_val(line)

    monkeypatch.setattr(ibs_module, "line_val", fake_line_val)
    monkeypatch.setattr(ibs_module, "timeframed_line_val", fake_timeframed_line_val)

    class EchoPercentileTracker:
        def update(self, *_args, **_kwargs):
            return _args[1] if len(_args) > 1 else None

    strategy = IbsStrategy.__new__(IbsStrategy)
    dt_num = bt.date2num(datetime(2024, 1, 2, 9, 0))

    strategy.percentile_tracker = EchoPercentileTracker()
    strategy.hourly = SimpleNamespace(
        datetime=DummyLine(dt_num),
        close=DummyLine(105.0),
        high=DummyLine(110.0),
        low=DummyLine(100.0),
    )
    strategy.prev_bar_pct_data = strategy.hourly
    strategy.daily = SimpleNamespace(datetime=DummyLine(dt_num), close=DummyLine(100.0))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_median = None
    strategy.vix_data = None
    strategy.dom_threshold = None

    strategy.atr_z = DummyLine(0.5)
    strategy.atr_z_data = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.ibs = MethodType(lambda self: 0.55, strategy)

    strategy.prev_day_pct = MethodType(lambda self: 0.0, strategy)
    strategy.prev_bar_pct = MethodType(lambda self: 0.0, strategy)

    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy._ml_feature_snapshot = {}
    strategy.ml_feature_collector = {}
    strategy.filter_columns = []
    strategy.filter_keys = set()
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}

    strategy.p = SimpleNamespace(prev_bar_pct_tf="Hour", atrTF="Hour")

    strategy.ml_features = ("ibsxatrz",)
    strategy.ml_feature_param_keys = strategy._derive_ml_feature_param_keys()

    strategy.collect_filter_values()

    collector = strategy.ml_feature_collector
    expected = 0.55 * 0.5
    assert collector["ibsxatrz"] == pytest.approx(expected)


def test_collect_filter_values_populates_ibsxvolz_for_ml_features(monkeypatch):
    class DummyLine:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return self.value

    def fake_line_val(line, ago=0):
        if line is None:
            return None
        return getattr(line, "value", line)

    def fake_timeframed_line_val(
        line, *, data=None, timeframe=None, daily_ago=-1, intraday_ago=0
    ):
        return fake_line_val(line)

    monkeypatch.setattr(ibs_module, "line_val", fake_line_val)
    monkeypatch.setattr(ibs_module, "timeframed_line_val", fake_timeframed_line_val)

    class EchoPercentileTracker:
        def update(self, *_args, **_kwargs):
            return _args[1] if len(_args) > 1 else None

    strategy = IbsStrategy.__new__(IbsStrategy)
    dt_num = bt.date2num(datetime(2024, 1, 2, 9, 0))

    strategy.percentile_tracker = EchoPercentileTracker()
    strategy.hourly = SimpleNamespace(
        datetime=DummyLine(dt_num),
        close=DummyLine(105.0),
        high=DummyLine(110.0),
        low=DummyLine(100.0),
    )
    strategy.prev_bar_pct_data = strategy.hourly
    strategy.daily = SimpleNamespace(datetime=DummyLine(dt_num), close=DummyLine(100.0))
    strategy.signal_data = SimpleNamespace(close=DummyLine(float("nan")))
    strategy.last_pivot_high = None
    strategy.last_pivot_low = None
    strategy.prev_pivot_high = None
    strategy.prev_pivot_low = None
    strategy.has_vix = False
    strategy.vix_median = None
    strategy.vix_data = None
    strategy.dom_threshold = None

    strategy.vol_z = DummyLine(0.27)
    strategy.vol_z_data = SimpleNamespace(datetime=DummyLine(dt_num))
    strategy.ibs = MethodType(lambda self: 0.55, strategy)

    strategy.prev_day_pct = MethodType(lambda self: 0.0, strategy)
    strategy.prev_bar_pct = MethodType(lambda self: 0.0, strategy)

    strategy.cross_zscore_meta = {}
    strategy.return_meta = {}
    strategy._ml_feature_snapshot = {}
    strategy.ml_feature_collector = {}
    strategy.filter_columns = []
    strategy.filter_keys = set()
    strategy.filter_column_keys = set()
    strategy.filter_columns_by_param = {}
    strategy.column_to_param = {}

    strategy.p = SimpleNamespace(prev_bar_pct_tf="Hour", volTF="Hour")

    strategy.ml_features = ("ibsxvolz",)
    strategy.ml_feature_param_keys = strategy._derive_ml_feature_param_keys()

    strategy.collect_filter_values()

    collector = strategy.ml_feature_collector
    expected = 0.55 * 0.27
    assert collector["volz_pct"] == pytest.approx(0.27)
    assert collector["ibsxvolz"] == pytest.approx(expected)


def _build_strategy_for_logging(features, snapshot):
    class DummyLine:
        def __init__(self, values):
            self.values = list(values)

        def __len__(self):
            return len(self.values)

        def __getitem__(self, idx):
            if idx < 0:
                idx = len(self.values) + idx
            return self.values[idx]

    class DummyDateLine:
        def __init__(self, dt):
            self.dt = dt

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return 0.0

        def date(self, idx=None):
            return self.dt.date()

        def datetime(self, idx=None):
            return self.dt

    class DummyData:
        def __init__(self, dt_line, price_line):
            self.datetime = dt_line
            self.close = price_line
            self.open = price_line
            self.high = price_line
            self.low = price_line

        def __len__(self):
            return len(self.close)

    class DummyParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getattr__(self, name):  # pragma: no cover - default fallback
            return False

    strategy = IbsStrategy.__new__(IbsStrategy)
    now = datetime(2024, 1, 2, 12, 0)
    hourly = DummyData(DummyDateLine(now), DummyLine([101.0]))
    daily = DummyData(DummyDateLine(now), DummyLine([100.0]))

    strategy.hourly = hourly
    strategy.daily = daily
    strategy.signal_data = SimpleNamespace(close=DummyLine([float("nan")]))
    strategy.update_pivots = MethodType(lambda self: None, strategy)
    strategy.datr_pct_last_date = None
    strategy.hatr_pct_last_dt = None
    strategy._update_datr_pct = MethodType(lambda self: 0.0, strategy)
    strategy._update_hatr_pct = MethodType(lambda self: 0.0, strategy)
    strategy.datr_pct_pct = 0.0
    strategy.hatr_pct_pct = 0.0
    strategy.order = None
    strategy.cancel = MethodType(lambda self, _order: None, strategy)
    strategy.ibs = MethodType(lambda self: 0.5, strategy)
    strategy.prev_daily_ibs = MethodType(lambda self: 0.4, strategy)
    strategy.prev_daily_ibs_val = None
    strategy.prev_ibs_val = None
    strategy.twrc_data = None
    strategy.twrc_last_dt = None
    strategy.twrc_fast = None
    strategy.twrc_base = None
    strategy.pair_z_denom = None
    strategy.atr_z_denom = None
    strategy.vol_z_denom = None
    strategy.dist_z_denom = None
    strategy.mom3_z_denom = None
    strategy.price_z_denom = None
    strategy.datr_z_denom = None
    strategy.dvol_z_denom = None
    strategy.tratr_denom = None
    strategy.can_execute = False
    strategy.bar_executed = None
    strategy.pending_exit = {}
    strategy.trades_log = []
    strategy.current_signal = None
    strategy._ml_last_score = 0.42
    strategy.collect_filter_values = MethodType(
        lambda self, intraday_ago=0: snapshot,
        strategy,
    )

    strategy.p = DummyParams(
        symbol="ES",
        trade_start=now.date() - timedelta(days=1),
        ml_threshold=0.5,
        ml_features=features,
        enable_ibs_exit=False,
    )

    return strategy


def test_next_logs_missing_ml_features_in_warning(caplog):
    features = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    snapshot = {
        "feat_a": 1.0,
        "feat_b": None,
        "feat_c": None,
        "feat_d": None,
        "feat_e": None,
        "ml_passed": False,
    }
    strategy = _build_strategy_for_logging(features, snapshot)

    caplog.clear()
    caplog.set_level(logging.WARNING, logger=ibs_module.logger.name)
    strategy.next()

    warning_records = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warning_records, "Expected a warning record"
    message = warning_records[-1].message
    assert "Missing: feat_b, feat_c, feat_d, feat_e" in message


def test_next_logs_full_missing_feature_list_in_debug(caplog):
    features = [f"feat_{idx}" for idx in range(15)]
    snapshot = {features[0]: 1.0, "ml_passed": False}
    for feature in features[1:]:
        snapshot[feature] = None

    strategy = _build_strategy_for_logging(features, snapshot)

    caplog.clear()
    caplog.set_level(logging.DEBUG, logger=ibs_module.logger.name)
    strategy.next()

    warning_records = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warning_records, "Expected a warning record"
    warning_message = warning_records[-1].message
    expected_remaining = len(features) - 1 - 8
    assert expected_remaining > 0
    assert f"... (+{expected_remaining} more)" in warning_message

    debug_records = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    assert debug_records, "Expected debug log for missing features"
    debug_message = debug_records[-1].message
    for feature in features[1:]:
        assert feature in debug_message


def test_cross_zscore_recovers_after_missing_feed():
    strategy = IbsStrategy.__new__(IbsStrategy)
    strategy.cross_data_cache = {}
    strategy.cross_data_missing = set()
    strategy.cross_zscore_cache = {}
    strategy.available_data_names = set()
    strategy.cross_zscore_meta = {}
    strategy.filter_keys = set()
    strategy._ml_feature_snapshot = {}
    strategy.ml_feature_collector = {}

    strategy._publish_ml_feature = MethodType(
        lambda self, key, numeric, _propagating=False: None, strategy
    )
    strategy._record_cross_zscore_snapshot = MethodType(
        lambda self, meta, intraday_ago=0: (None, None), strategy
    )

    strategy.p = SimpleNamespace(enable6CZScoreHour=False)

    feed_name = "6C_hour"
    feed = SimpleNamespace(_name=feed_name, close=object())
    state = {"ready": False}

    def fake_getdatabyname(self, name):
        if not state["ready"]:
            raise KeyError(name)
        return feed

    strategy.getdatabyname = MethodType(fake_getdatabyname, strategy)

    def fake_build(
        self,
        symbol,
        timeframe,
        feed_suffix,
        length,
        window,
        data_feed=None,
    ):
        if data_feed is None:
            enable_param = f"enable{symbol}ZScore{timeframe}"
            data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
        if data_feed is None:
            return None
        pipeline = {
            "line": f"{symbol}_{timeframe}_line",
            "mean": f"{symbol}_{timeframe}_mean",
            "std": f"{symbol}_{timeframe}_std",
            "denom": f"{symbol}_{timeframe}_denom",
            "len": length,
            "window": window,
            "data": data_feed,
        }
        self.cross_zscore_cache[(symbol, timeframe)] = pipeline
        return pipeline

    strategy._build_cross_zscore_pipeline = MethodType(fake_build, strategy)

    enable_param = "enable6CZScoreHour"
    meta = {
        "symbol": "6C",
        "timeframe": "Hour",
        "feed_suffix": "hour",
        "len_param": "len_param",
        "window_param": "window_param",
        "len_aliases": ("len_param",),
        "window_aliases": ("window_param",),
        "feature_key": _metadata_feature_key("6C", "Hour", "z_score"),
        "pipeline_feature_key": _metadata_feature_key("6C", "Hour", "z_pipeline"),
        "line": None,
        "data": None,
        "mean": None,
        "std": None,
        "denom": None,
    }
    strategy.cross_zscore_meta[enable_param] = meta

    missing = strategy._get_cross_feed("6C", "hour", enable_param)
    assert missing is None
    assert ("6C", "hour") not in strategy.cross_data_cache
    assert ("6C", "hour") in strategy.cross_data_missing

    state["ready"] = True
    strategy.available_data_names.add(feed_name)

    pipeline = strategy._build_cross_zscore_pipeline("6C", "Hour", "hour", 20, 252)
    assert pipeline is not None

    meta.update(
        {
            "line": pipeline["line"],
            "data": pipeline["data"],
            "mean": pipeline["mean"],
            "std": pipeline["std"],
            "denom": pipeline["denom"],
            "len": 20,
            "window": 252,
        }
    )
    strategy._record_cross_zscore_snapshot(meta)

    assert strategy.cross_data_cache[("6C", "hour")] is feed
    assert ("6C", "hour") not in strategy.cross_data_missing
    assert meta["data"] is feed
    assert meta["line"] == "6C_Hour_line"
    assert strategy.cross_zscore_cache[("6C", "Hour")]["data"] is feed
