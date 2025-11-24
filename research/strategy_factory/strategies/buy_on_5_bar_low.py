"""
Strategy #40: Buy on 5 Bar Low Strategy

Simple mean reversion strategy:
- Enter when close < lowest low of previous N bars (default 5)
- Exit when close > high[1], indicating price strength

Botnet101 strategy - works on any timeframe.

Expected Performance:
- High frequency of signals in volatile markets
- Best for stocks/ETFs/futures
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class BuyOn5BarLow(BaseStrategy):
    """
    Buy on 5 Bar Low Strategy (Catalogue #40).

    Entry:
    - Close < lowest low of previous N bars (oversold reversal signal)

    Exit:
    - Close > high[1] (showing price strength)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - lookback: [3, 5, 7, 10]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=40,
            name="BuyOn5BarLow",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'lookback': [3, 5, 7, 10]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['lookback']) + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lowest low over lookback period.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with lowest_low column
        """
        lookback = self.params.get('lookback', 5)

        # Lowest low of previous N bars (shift by 1 to exclude current bar)
        data['lowest_low'] = data['Low'].shift(1).rolling(window=lookback).min()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when close < lowest low.

        Args:
            data: OHLCV dataframe with lowest_low
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close < lowest low of previous N bars
        entry = data['Close'] < data['lowest_low']

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
        Exit when close > high[1].

        Args:
            data: OHLCV dataframe
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Get previous bar's high
        if current_idx > 0:
            prev_high = data.iloc[current_idx - 1]['High']

            # Exit when close > high[1]
            if current_bar['Close'] > prev_high:
                return TradeExit(
                    exit=True,
                    exit_type='signal',
                    exit_price=current_bar['Close']
                )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
