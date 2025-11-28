"""
Strategy #42: Gap Down Reversal Strategy

Gap reversal strategy:
- Enter when previous bar is bearish, current bar gaps down, then reverses bullish
- Exit when close > highest high of previous N bars

Botnet101 strategy - capitalizes on gap down exhaustion.

Expected Performance:
- Requires true gap below prior low
- Best on timeframes with gaps (daily, 1HR+)
- Works in high volatility
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class GapDownReversal(BaseStrategy):
    """
    Gap Down Reversal Strategy (Catalogue #42).

    Entry:
    - Previous bar is bearish (close[1] < open[1])
    - Current bar gaps down (open < low[1])
    - Current bar closes bullish (close > open)

    Exit:
    - Close > highest high of previous N bars
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - exit_lookback: [5, 7, 10]
    - min_gap_pct: [0.0, 0.1, 0.25] (minimum gap size as % of price)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=42,
            name="GapDownReversal",
            archetype="gap_reversal",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'exit_lookback': [5, 7, 10],
            'min_gap_pct': [0.0, 0.1, 0.25]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['exit_lookback']) + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate highest high for exit and gap conditions.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with highest_high and gap flags
        """
        exit_lookback = self.params.get('exit_lookback', 7)

        # Highest high of previous N bars for exit
        data['highest_high'] = data['High'].shift(1).rolling(window=exit_lookback).max()

        # Previous bar OHLC
        data['prev_open'] = data['Open'].shift(1)
        data['prev_high'] = data['High'].shift(1)
        data['prev_low'] = data['Low'].shift(1)
        data['prev_close'] = data['Close'].shift(1)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on gap down reversal.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        min_gap_pct = params.get('min_gap_pct', 0.0) / 100.0

        # Condition 1: Previous bar is bearish
        prev_bearish = data['prev_close'] < data['prev_open']

        # Condition 2: Current bar gaps down (open < prior low)
        gap_down = data['Open'] < data['prev_low']

        # Optional: Minimum gap size filter
        if min_gap_pct > 0:
            gap_size = (data['prev_low'] - data['Open']) / data['prev_low']
            gap_down = gap_down & (gap_size >= min_gap_pct)

        # Condition 3: Current bar closes bullish (close > open)
        bullish_close = data['Close'] > data['Open']

        # Entry: All three conditions met
        entry = prev_bearish & gap_down & bullish_close

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
