"""
Strategy #41: 3-Bar Low Strategy

Asymmetric mean reversion strategy:
- Enter when close < lowest low of previous 3 bars
- Exit when close > highest high of previous 7 bars
- Optional moving average filter

Botnet101 strategy - faster signals than 5-bar low.

Expected Performance:
- Faster signals than 5-bar low
- Works on any timeframe from intraday to daily
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ThreeBarLow(BaseStrategy):
    """
    3-Bar Low Strategy (Catalogue #41).

    Entry:
    - Close < lowest low of previous N bars (default 3)
    - Optional: Price above moving average filter

    Exit:
    - Close > highest high of previous M bars (default 7)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - lookback_entry: [2, 3, 4, 5]
    - lookback_exit: [5, 7, 10]
    - ma_period: [10, 20, 50]
    - use_ma_filter: [True, False]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=41,
            name="ThreeBarLow",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'lookback_entry': [2, 3, 4, 5],
            'lookback_exit': [5, 7, 10],
            'ma_period': [10, 20, 50],
            'use_ma_filter': [True, False]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ma_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lowest low for entry and highest high for exit.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with lowest_low, highest_high, and ma columns
        """
        lookback_entry = self.params.get('lookback_entry', 3)
        lookback_exit = self.params.get('lookback_exit', 7)
        ma_period = self.params.get('ma_period', 20)

        # Lowest low of previous N bars for entry (exclude current bar)
        data['lowest_low'] = data['Low'].shift(1).rolling(window=lookback_entry).min()

        # Highest high of previous M bars for exit (exclude current bar)
        data['highest_high'] = data['High'].shift(1).rolling(window=lookback_exit).max()

        # Moving average filter (SMA)
        data['ma'] = data['Close'].rolling(window=ma_period).mean()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when close < lowest low.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        use_ma_filter = params.get('use_ma_filter', False)

        # Entry: Close < lowest low of previous N bars
        entry = data['Close'] < data['lowest_low']

        # Optional MA filter: only enter if price above MA
        if use_ma_filter:
            entry = entry & (data['Close'] > data['ma'])

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
        Exit when close > highest high of previous N bars.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when close > highest high of previous N bars
        if current_bar['Close'] > current_bar['highest_high']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
