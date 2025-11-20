"""
Strategy #15: Price Channel Breakout (Donchian Channel)

Classic breakout strategy:
- Enter when price breaks above N-day high
- Exit when price breaks below N-day low

Turtle Trading strategy popularized by Richard Dennis.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-3,000
- Raw Sharpe: 0.2-0.4
- ML Sharpe: 0.7-1.3
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class PriceChannelBreakout(BaseStrategy):
    """
    Price Channel Breakout Strategy (Catalogue #15).

    Entry:
    - Price breaks above highest high of last N bars
    - (with optional breakout percentage buffer)

    Exit:
    - Price breaks below lowest low of last N bars
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - channel_length: [15, 20, 25, 30]
    - channel_breakout_pct: [0.0, 0.25, 0.5]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=15,
            name="PriceChannelBreakout",
            archetype="breakout",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'channel_length': [15, 20, 25, 30],
            'channel_breakout_pct': [0.0, 0.25, 0.5]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_length = max(self.param_grid['channel_length'])
        return max_length + 5  # +5 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price channel (highest high and lowest low).

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with channel_high and channel_low columns
        """
        channel_length = self.params.get('channel_length', 20)
        breakout_pct = self.params.get('channel_breakout_pct', 0.0) / 100.0

        # Rolling highest high and lowest low
        data['channel_high'] = data['High'].rolling(window=channel_length).max()
        data['channel_low'] = data['Low'].rolling(window=channel_length).min()

        # Apply breakout percentage buffer
        channel_range = data['channel_high'] - data['channel_low']
        data['channel_high_adj'] = data['channel_high'] + (channel_range * breakout_pct)
        data['channel_low_adj'] = data['channel_low'] - (channel_range * breakout_pct)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price breaks above channel high.

        Args:
            data: OHLCV dataframe with channel
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close > channel_high_adj
        entry = data['Close'] > data['channel_high_adj']

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
        Exit when price breaks below channel low.

        Args:
            data: OHLCV dataframe with channel
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        current_price = current_bar['Close']

        # Exit when price breaks below channel low
        if current_price < current_bar['channel_low_adj']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_price
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
