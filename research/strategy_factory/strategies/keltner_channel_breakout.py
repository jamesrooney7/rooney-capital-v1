"""
Strategy #2: Keltner Channel Breakout

Classic volatility breakout strategy:
- Enter when price breaks above upper Keltner Channel
- Exit when price touches middle band (EMA)

Keltner Channels = EMA ± (ATR × multiplier)

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_ema


class KeltnerChannelBreakout(BaseStrategy):
    """
    Keltner Channel Breakout Strategy (Catalogue #2).

    Entry:
    - Price breaks above upper Keltner Channel
    - Upper = EMA(length) + ATR(atr_length) × multiplier

    Exit:
    - Price touches middle band (EMA)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - kc_length: [10, 20, 30, 40]
    - kc_atr_length: [10, 14, 20]
    - kc_multiplier: [1.5, 2.0, 2.5, 3.0]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=2,
            name="KeltnerChannelBreakout",
            archetype="breakout",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'kc_length': [10, 20, 30, 40],
            'kc_atr_length': [10, 14, 20],
            'kc_multiplier': [1.5, 2.0, 2.5, 3.0]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_length = max(self.param_grid['kc_length'])
        max_atr = max(self.param_grid['kc_atr_length'])
        return max(max_length, max_atr) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with kc_upper, kc_middle, kc_lower columns
        """
        kc_length = self.params.get('kc_length', 20)
        kc_atr_length = self.params.get('kc_atr_length', 14)
        kc_multiplier = self.params.get('kc_multiplier', 2.0)

        # Middle band = EMA
        data['kc_middle'] = calculate_ema(data['Close'], kc_length)

        # Calculate ATR for channel width
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=kc_atr_length).mean()

        # Upper and lower bands
        data['kc_upper'] = data['kc_middle'] + (atr * kc_multiplier)
        data['kc_lower'] = data['kc_middle'] - (atr * kc_multiplier)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price breaks above upper Keltner Channel.

        Args:
            data: OHLCV dataframe with Keltner Channels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close breaks above upper channel
        # Current bar: Close > upper
        # Previous bar: Close <= upper
        breakout = data['Close'] > data['kc_upper']
        was_below = data['Close'].shift(1) <= data['kc_upper'].shift(1)

        entry = breakout & was_below

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
        Exit when price touches middle band.

        Args:
            data: OHLCV dataframe with Keltner Channels
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when price touches or crosses below middle band
        if current_bar['Close'] <= current_bar['kc_middle']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
