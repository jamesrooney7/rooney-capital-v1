"""
Strategy #50: 3 Down 3 Up Strategy

Symmetric entry/exit strategy:
- Enter after N consecutive down closes (default 3)
- Exit after N consecutive up closes (default 3)
- Optional 200 EMA filter

Botnet101 strategy - works on any timeframe.

Expected Performance:
- Best in oscillating/volatile markets
- Simple consecutive close counter
- Optional trend filter improves reliability
- Typical hold period: Short-term
- NOT optimized per author
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ThreeDownThreeUp(BaseStrategy):
    """
    3 Down 3 Up Strategy (Catalogue #50).

    Entry:
    - N consecutive down closes (close < previous close)
    - Optional: Price above 200 EMA filter

    Exit:
    - N consecutive up closes (close > previous close)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - consecutive_down_entry: [2, 3, 4]
    - consecutive_up_exit: [2, 3, 4]
    - ema_period: [100, 150, 200]
    - use_ema_filter: [True, False]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=50,
            name="ThreeDownThreeUp",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'consecutive_down_entry': [2, 3, 4],
            'consecutive_up_exit': [2, 3, 4],
            'ema_period': [100, 150, 200],
            'use_ema_filter': [True, False]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ema_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consecutive up/down closes and EMA.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with consecutive_down, consecutive_up, ema columns
        """
        ema_period = self.params.get('ema_period', 200)

        # EMA filter
        data['ema'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

        # Down close and up close
        data['down_close'] = data['Close'] < data['Close'].shift(1)
        data['up_close'] = data['Close'] > data['Close'].shift(1)

        # Count consecutive down closes
        consecutive_down = []
        count = 0
        for val in data['down_close']:
            if val:
                count += 1
            else:
                count = 0
            consecutive_down.append(count)
        data['consecutive_down'] = consecutive_down

        # Count consecutive up closes
        consecutive_up = []
        count = 0
        for val in data['up_close']:
            if val:
                count += 1
            else:
                count = 0
            consecutive_up.append(count)
        data['consecutive_up'] = consecutive_up

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals after N consecutive down closes.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        consecutive_down_entry = params.get('consecutive_down_entry', 3)
        use_ema_filter = params.get('use_ema_filter', False)

        # Entry: Consecutive down closes >= threshold
        entry = data['consecutive_down'] >= consecutive_down_entry

        # Optional EMA filter: only enter if price above EMA
        if use_ema_filter:
            entry = entry & (data['Close'] > data['ema'])

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
        Exit after N consecutive up closes.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        consecutive_up_exit = params.get('consecutive_up_exit', 3)
        current_bar = data.iloc[current_idx]

        # Exit after N consecutive up closes
        if current_bar['consecutive_up'] >= consecutive_up_exit:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
