import sys
from pathlib import Path
from types import MethodType

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
