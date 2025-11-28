"""
Strategy #16: Turtle Trading

Famous breakout system by Richard Dennis:
- Enter on 20-day high breakout (System 1) or 55-day (System 2)
- Exit on 10-day low (System 1) or 20-day low (System 2)

One of the most tested and documented trading systems.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.2-0.4
- ML Sharpe: 0.8-1.5
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class TurtleTrading(BaseStrategy):
    """
    Turtle Trading System (Catalogue #16).

    Entry:
    - Price breaks above N-day high (20 or 55 days)

    Exit:
    - Price breaks below N-day low (10 or 20 days)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - turtle_entry_length: [20, 30, 40, 55]
    - turtle_exit_length: [10, 15, 20]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=16,
            name="TurtleTrading",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'turtle_entry_length': [20, 30, 40, 55],
            'turtle_exit_length': [10, 15, 20]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['turtle_entry_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate entry and exit levels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with entry_high and exit_low columns
        """
        entry_length = self.params.get('turtle_entry_length', 20)
        exit_length = self.params.get('turtle_exit_length', 10)

        # Entry level = N-day high
        data['entry_high'] = data['High'].rolling(window=entry_length).max()

        # Exit level = N-day low
        data['exit_low'] = data['Low'].rolling(window=exit_length).min()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on breakout above entry high.

        Args:
            data: OHLCV dataframe with entry/exit levels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close breaks above entry high
        breakout = data['Close'] > data['entry_high'].shift(1)

        # Avoid re-entry on same breakout
        was_below = data['Close'].shift(1) <= data['entry_high'].shift(2)

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
        Exit on breakdown below exit low.

        Args:
            data: OHLCV dataframe with entry/exit levels
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when price breaks below exit low
        if current_bar['Close'] < current_bar['exit_low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
