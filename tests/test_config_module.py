import importlib
import sys
from pathlib import Path

import backtrader as bt
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
STRATEGY_ROOT = SRC_ROOT / "strategy"
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))


def _reload_config(monkeypatch, **env):
    keys = [
        "PINE_COMMISSION_PER_SIDE",
        "COMMISSION_PER_SIDE",
        "PINE_PAIR_MAP",
        "PAIR_MAP",
        "PINE_PAIR_MAP_PATH",
        "PAIR_MAP_PATH",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("config", None)
    return importlib.import_module("config")


def _reload_strategy(monkeypatch, **env):
    config = _reload_config(monkeypatch, **env)
    sys.modules.pop("strategy.ibs_strategy", None)
    module = importlib.import_module("strategy.ibs_strategy")
    return config, module.IbsStrategy


def _make_ohlc(start: str, periods: int, freq: str) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq)
    base = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "open": 100 + base,
            "high": 100.5 + base,
            "low": 99.5 + base,
            "close": 100.25 + base,
            "volume": 1_000,
            "openinterest": 0,
        },
        index=index,
    )


def test_config_defaults(monkeypatch):
    config = _reload_config(monkeypatch)
    assert config.COMMISSION_PER_SIDE == pytest.approx(
        config.DEFAULT_COMMISSION_PER_SIDE
    )
    assert dict(config.PAIR_MAP) == dict(config.DEFAULT_PAIR_MAP)


def test_config_env_overrides(monkeypatch):
    config = _reload_config(
        monkeypatch,
        PINE_COMMISSION_PER_SIDE="2.5",
        PINE_PAIR_MAP="ES:YM,NQ:ES",
    )
    assert config.COMMISSION_PER_SIDE == 2.5
    assert config.PAIR_MAP["ES"] == "YM"
    assert config.PAIR_MAP["NQ"] == "ES"


def test_strategy_initialises_with_default_config(monkeypatch):
    config, IbsStrategy = _reload_strategy(monkeypatch)

    cerebro = bt.Cerebro()
    es_hour = bt.feeds.PandasData(dataname=_make_ohlc("2023-01-01", 400, "H"))
    es_day = bt.feeds.PandasData(
        dataname=_make_ohlc("2022-01-01", 400, "D"), timeframe=bt.TimeFrame.Days
    )
    tlt_day = bt.feeds.PandasData(
        dataname=_make_ohlc("2022-01-01", 400, "D"), timeframe=bt.TimeFrame.Days
    )

    cerebro.adddata(es_hour, name="ES_hour")
    cerebro.adddata(es_day, name="ES_day")
    cerebro.adddata(tlt_day, name="TLT_day")

    cerebro.addstrategy(
        IbsStrategy,
        symbol="ES",
        pair_symbol="",  # rely on config.PAIR_MAP
        enable_ibs_entry=False,
        enable_ibs_exit=False,
        enable_stop=False,
        enable_tp=False,
        enable_auto_close=False,
    )

    strategies = cerebro.run(maxcpus=1)
    strategy = strategies[0]
    expected_pair = config.PAIR_MAP.get("ES", "")
    assert strategy.p.pair_symbol == expected_pair

