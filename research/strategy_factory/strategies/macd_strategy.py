"""
Strategy #19: MACD Strategy

Most popular momentum indicator:
- Enter when MACD line crosses above signal line
- Exit when MACD line crosses below signal line

Widely used but prone to whipsaws in ranging markets.

Expected Performance (ES 2010-2024):
- Trade Count: 4,000-6,000
- Raw Sharpe: 0.2-0.4
- ML Sharpe: 0.8-1.5
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_macd


class MACDStrategy(BaseStrategy):
    """
    MACD Crossover Strategy (Catalogue #19).

    Entry:
    - MACD line crosses above signal line

    Exit:
    - MACD line crosses below signal line
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - macd_fast: [8, 12, 16]
    - macd_slow: [21, 26, 31]
    - macd_signal: [7, 9, 11]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=19,
            name="MACD",
            archetype="momentum",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'macd_fast': [8, 12, 16],
            'macd_slow': [21, 26, 31],
            'macd_signal': [7, 9, 11]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_slow = max(self.param_grid['macd_slow'])
        max_signal = max(self.param_grid['macd_signal'])
        return max_slow + max_signal + 10  # +10 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD, signal line, and histogram.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with macd_line, macd_signal, macd_histogram columns
        """
        macd_fast = self.params.get('macd_fast', 12)
        macd_slow = self.params.get('macd_slow', 26)
        macd_signal_period = self.params.get('macd_signal', 9)

        macd_line, signal_line, histogram = calculate_macd(
            data['Close'],
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal_period
        )

        data['macd_line'] = macd_line
        data['macd_signal'] = signal_line
        data['macd_histogram'] = histogram

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when MACD crosses above signal line.

        Args:
            data: OHLCV dataframe with MACD
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: MACD line crosses above signal line
        # Current bar: MACD > signal
        # Previous bar: MACD <= signal
        macd_above_signal = data['macd_line'] > data['macd_signal']
        macd_was_below = data['macd_line'].shift(1) <= data['macd_signal'].shift(1)

        entry = macd_above_signal & macd_was_below

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
        Exit when MACD crosses below signal line.

        Args:
            data: OHLCV dataframe with MACD
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when MACD crosses below signal line
        if current_bar['macd_line'] < current_bar['macd_signal']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
