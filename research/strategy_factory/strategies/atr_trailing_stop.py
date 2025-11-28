"""
Strategy #28: ATR Trailing Stop

Trend-following using ATR-based trailing stop:
- Enter on momentum signal (price above MA, strong move)
- Trail stop using ATR multiples
- Exit when stop hit or momentum reverses

Dynamic stop placement based on volatility.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 0.9-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma, calculate_atr


class ATRTrailingStop(BaseStrategy):
    """
    ATR Trailing Stop Strategy (Catalogue #28).

    Entry:
    - Price breaks above N-period high (momentum)
    - Price above moving average (trend filter)

    Exit:
    - ATR trailing stop hit (price falls below highest high - X*ATR)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - atr_period: [14, 21, 28] (ATR calculation period)
    - atr_mult: [2.0, 3.0, 4.0] (ATR multiplier for stop distance)
    - atr_entry_length: [10, 20, 30] (breakout period for entry)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=28,
            name="ATRTrailingStop",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'atr_period': [14, 21, 28],
            'atr_mult': [2.0, 3.0, 4.0],
            'atr_entry_length': [10, 20, 30]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['atr_period']) + max(self.param_grid['atr_entry_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR and breakout levels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ATR and entry levels
        """
        atr_period = self.params.get('atr_period', 14)
        entry_length = self.params.get('atr_entry_length', 20)

        # Calculate ATR
        data['atr_ind'] = calculate_atr(data, atr_period)

        # Entry level = N-period high
        data['entry_high'] = data['High'].rolling(window=entry_length).max()

        # Moving average for trend filter
        data['ma_50'] = calculate_sma(data['Close'], 50)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on breakout above recent high.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry conditions:
        # 1. Price breaks above N-period high
        breakout = data['Close'] > data['entry_high'].shift(1)

        # 2. Price above MA (trend filter)
        above_ma = data['Close'] > data['ma_50']

        # 3. Previous close was below the high (fresh breakout)
        was_below = data['Close'].shift(1) <= data['entry_high'].shift(2)

        entry = breakout & above_ma & was_below

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
        Exit when ATR trailing stop is hit.

        Args:
            data: OHLCV dataframe with ATR
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        atr_mult = params.get('atr_mult', 3.0)

        # Calculate trailing stop level
        # Stop = highest high since entry - (ATR * multiplier)
        data_slice = data.iloc[entry_idx:current_idx+1]
        highest_high = data_slice['High'].max()

        current_bar = data.iloc[current_idx]
        stop_level = highest_high - (current_bar['atr_ind'] * atr_mult)

        # Exit if price closes below stop
        if current_bar['Close'] < stop_level:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
