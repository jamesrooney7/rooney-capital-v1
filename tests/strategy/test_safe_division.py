import sys
from pathlib import Path

import pandas as pd
import pytest

import backtrader as bt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy.safe_div import SafeDivision


class _ZeroDenominatorStrategy(bt.Strategy):
    params = dict(fallback=7.0)

    def __init__(self):
        zero_line = bt.LineNum(0.0)
        self.safe_division = SafeDivision(self.data.close, zero_line, zero=self.p.fallback)
        self.outputs = []

    def next(self):
        # Record the computed value so the test can inspect it after the run.
        self.outputs.append(self.safe_division[0])


def _run_strategy(dataframe, strategy, *, runonce=True):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)
    cerebro.addstrategy(strategy)
    result = cerebro.run(runonce=runonce)
    return result[0]


@pytest.mark.parametrize("runonce", [True, False])
def test_safe_division_handles_zero_denominator_without_error(runonce):
    dataframe = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    strategy = _run_strategy(dataframe, _ZeroDenominatorStrategy, runonce=runonce)

    assert strategy.outputs[0] == pytest.approx(strategy.p.fallback)
