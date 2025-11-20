"""
Strategy #32: ROC (Rate of Change)

Momentum strategy using Rate of Change indicator:
- Enter when ROC crosses above threshold (positive momentum)
- ROC measures % price change over N periods
- Exit when ROC crosses below threshold (momentum fading)

Classic momentum oscillator for trend identification.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 0.9-1.6
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma


class ROCStrategy(BaseStrategy):
    """
    ROC (Rate of Change) Strategy (Catalogue #32).

    Entry:
    - ROC crosses above threshold (positive momentum)
    - Price above moving average (trend filter)
    - Indicates building upward momentum

    Exit:
    - ROC crosses below threshold (momentum fading)
    - OR price crosses below MA (trend broken)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - roc_period: [10, 20, 30] (lookback for ROC calculation)
    - roc_threshold: [2, 4, 6] (% threshold for entry/exit)
    - roc_ma_length: [50, 100, 200] (trend filter MA)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=32,
            name="ROCStrategy",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'roc_period': [10, 20, 30],
            'roc_threshold': [2, 4, 6],
            'roc_ma_length': [50, 100, 200]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['roc_ma_length']) + max(self.param_grid['roc_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROC and trend filter.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ROC and MA
        """
        period = self.params.get('roc_period', 20)
        ma_length = self.params.get('roc_ma_length', 100)

        # Rate of Change: ((Close - Close[n]) / Close[n]) * 100
        data['roc'] = ((data['Close'] - data['Close'].shift(period)) /
                       data['Close'].shift(period)) * 100

        # Trend filter MA
        data['ma_trend'] = calculate_sma(data['Close'], ma_length)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when ROC turns positive.

        Args:
            data: OHLCV dataframe with ROC
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('roc_threshold', 4)

        # Entry conditions:
        # 1. ROC crosses above threshold (positive momentum)
        roc_above = data['roc'] > threshold
        roc_was_below = data['roc'].shift(1) <= threshold

        # 2. Price above MA (trend filter)
        above_ma = data['Close'] > data['ma_trend']

        entry = roc_above & roc_was_below & above_ma

        return entry

    def exit_logic(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        current_idx: int
    ) -> TradeExit:
        """
        Exit when ROC fades or trend breaks.

        Args:
            data: OHLCV dataframe with ROC
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        threshold = params.get('roc_threshold', 4)
        current_bar = data.iloc[current_idx]

        # Exit 1: ROC crosses below threshold (momentum fading)
        if current_bar['roc'] < threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Price crosses below MA (trend broken)
        if current_bar['Close'] < current_bar['ma_trend']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
