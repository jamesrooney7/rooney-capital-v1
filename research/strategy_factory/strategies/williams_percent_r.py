"""
Strategy #12: Williams %R

Momentum oscillator measuring overbought/oversold:
- Enter when %R crosses above oversold (-80)
- Exit when %R crosses below overbought (-20)

Similar to Stochastic but inverted scale.

Expected Performance (ES 2010-2024):
- Trade Count: 7,000-11,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.1-1.9
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class WilliamsPercentR(BaseStrategy):
    """
    Williams %R Strategy (Catalogue #12).

    Entry:
    - %R crosses above oversold level (-80)
    - %R = (High_n - Close) / (High_n - Low_n) × -100

    Exit:
    - %R crosses below overbought level (-20)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - wr_length: [10, 14, 20, 28]
    - wr_oversold: [-90, -80, -70]
    - wr_overbought: [-30, -20, -10]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=12,
            name="WilliamsPercentR",
            archetype="momentum",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'wr_length': [10, 14, 20, 28],
            'wr_oversold': [-90, -80, -70],
            'wr_overbought': [-30, -20, -10]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['wr_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with williams_r column
        """
        length = self.params.get('wr_length', 14)

        # Calculate %R
        high_n = data['High'].rolling(window=length).max()
        low_n = data['Low'].rolling(window=length).min()

        williams_r = ((high_n - data['Close']) / (high_n - low_n + 1e-10)) * -100

        data['williams_r'] = williams_r

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when %R crosses above oversold.

        Args:
            data: OHLCV dataframe with Williams %R
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        oversold = params.get('wr_oversold', -80)

        # Entry: %R crosses above oversold
        above_oversold = data['williams_r'] > oversold
        was_below = data['williams_r'].shift(1) <= oversold

        entry = above_oversold & was_below

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
        Exit when %R crosses below overbought.

        Args:
            data: OHLCV dataframe with Williams %R
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        overbought = params.get('wr_overbought', -20)
        current_bar = data.iloc[current_idx]

        # Exit when %R crosses below overbought
        if current_bar['williams_r'] < overbought:
            if current_idx > 0:
                prev_wr = data.iloc[current_idx - 1]['williams_r']
                if prev_wr >= overbought:
                    return TradeExit(
                        exit=True,
                        exit_type='signal',
                        exit_price=current_bar['Close']
                    )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
