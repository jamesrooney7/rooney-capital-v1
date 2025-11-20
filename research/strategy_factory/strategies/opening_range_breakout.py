"""
Strategy #25: Opening Range Breakout

Classic intraday breakout strategy:
- Define opening range as first N minutes (e.g., 30 min)
- Enter when price breaks above/below this range
- Exit at EOD or stops

Popular with day traders for capturing momentum.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 0.9-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import time
from .base import BaseStrategy, TradeExit


class OpeningRangeBreakout(BaseStrategy):
    """
    Opening Range Breakout Strategy (Catalogue #25).

    Entry:
    - Price breaks above opening range high (with optional buffer)

    Exit:
    - End of day
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - or_duration_minutes: [15, 30, 60]
    - or_breakout_pct: [0.0, 0.1, 0.2]  (percentage buffer)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=25,
            name="OpeningRangeBreakout",
            archetype="breakout",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'or_duration_minutes': [15, 30, 60],
            'or_breakout_pct': [0.0, 0.1, 0.2]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 5  # Need a few bars to establish opening range

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate opening range for each day.

        Args:
            data: OHLCV dataframe with datetime index

        Returns:
            DataFrame with or_high, or_low, or_defined columns
        """
        or_duration = self.params.get('or_duration_minutes', 30)
        breakout_pct = self.params.get('or_breakout_pct', 0.0) / 100.0

        # Add date and time columns
        data['date'] = data.index.date
        data['time'] = data.index.time

        # Identify opening range period
        # Assuming market opens at 9:30 AM ET for ES
        # Opening range = first N minutes (e.g., 9:30-10:00 for 30 min)
        market_open = time(9, 30)  # 9:30 AM

        # Calculate OR end time
        or_end_minutes = or_duration
        or_end_hour = 9 + (30 + or_end_minutes) // 60
        or_end_minute = (30 + or_end_minutes) % 60
        or_end_time = time(or_end_hour, or_end_minute)

        # Mark bars in opening range
        data['in_or'] = (data['time'] >= market_open) & (data['time'] < or_end_time)

        # Calculate OR high and low for each day
        data['or_high'] = data.groupby('date')['High'].transform(
            lambda x: x[data.loc[x.index, 'in_or']].max() if data.loc[x.index, 'in_or'].any() else np.nan
        )
        data['or_low'] = data.groupby('date')['Low'].transform(
            lambda x: x[data.loc[x.index, 'in_or']].min() if data.loc[x.index, 'in_or'].any() else np.nan
        )

        # Forward fill OR levels for the rest of the day
        data['or_high'] = data.groupby('date')['or_high'].ffill()
        data['or_low'] = data.groupby('date')['or_low'].ffill()

        # Apply breakout percentage buffer
        or_range = data['or_high'] - data['or_low']
        data['or_high_adj'] = data['or_high'] + (or_range * breakout_pct)
        data['or_low_adj'] = data['or_low'] - (or_range * breakout_pct)

        # Mark when OR is defined (after OR period ends)
        data['or_defined'] = data['time'] >= or_end_time

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price breaks above opening range.

        Args:
            data: OHLCV dataframe with OR levels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry conditions:
        # 1. OR is defined (after OR period)
        # 2. Close > or_high_adj (breakout)
        # 3. Not in OR period (don't trade during OR formation)

        entry = (
            data['or_defined'] &
            (data['Close'] > data['or_high_adj']) &
            (~data['in_or'])
        )

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
        No strategy-specific exit (relies on EOD exit).

        Args:
            data: OHLCV dataframe
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        # Could add: exit if price breaks back below OR low
        # For simplicity, relying on EOD exit in base class

        return TradeExit(exit=False, exit_type='none')
