"""
Strategy #18: EMA Ribbon

Trend following using multiple EMAs:
- Enter when fast EMA crosses above slow EMA ribbon
- Exit when fast EMA crosses below slow EMA ribbon

Uses 3-5 EMAs to identify strong trends.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-6,000
- Raw Sharpe: 0.2-0.5
- ML Sharpe: 0.9-1.6
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_ema


class EMARibbon(BaseStrategy):
    """
    EMA Ribbon Strategy (Catalogue #18).

    Entry:
    - Fast EMA crosses above slowest EMA in ribbon
    - All ribbon EMAs are aligned (ascending order)

    Exit:
    - Fast EMA crosses below slowest EMA
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ribbon_fast: [8, 13, 21]
    - ribbon_slow: [34, 55, 89]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=18,
            name="EMARibbon",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ribbon_fast': [8, 13, 21],
            'ribbon_slow': [34, 55, 89]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ribbon_slow']) + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA ribbon.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with fast and slow EMA columns
        """
        fast_length = self.params.get('ribbon_fast', 13)
        slow_length = self.params.get('ribbon_slow', 55)

        # Calculate EMAs
        data['ema_fast'] = calculate_ema(data['Close'], fast_length)
        data['ema_slow'] = calculate_ema(data['Close'], slow_length)

        # Middle EMAs for ribbon alignment check
        mid1 = int((fast_length + slow_length) / 3)
        mid2 = int((fast_length + slow_length) * 2 / 3)
        data['ema_mid1'] = calculate_ema(data['Close'], mid1)
        data['ema_mid2'] = calculate_ema(data['Close'], mid2)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when fast EMA crosses above slow EMA.

        Args:
            data: OHLCV dataframe with EMAs
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Fast crosses above slow
        cross_above = data['ema_fast'] > data['ema_slow']
        was_below = data['ema_fast'].shift(1) <= data['ema_slow'].shift(1)

        # Ribbon alignment: EMAs in ascending order (bullish)
        ribbon_aligned = (
            (data['ema_fast'] > data['ema_mid1']) &
            (data['ema_mid1'] > data['ema_mid2']) &
            (data['ema_mid2'] > data['ema_slow'])
        )

        entry = cross_above & was_below & ribbon_aligned

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
        Exit when fast EMA crosses below slow EMA.

        Args:
            data: OHLCV dataframe with EMAs
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when fast crosses below slow
        if current_bar['ema_fast'] < current_bar['ema_slow']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
