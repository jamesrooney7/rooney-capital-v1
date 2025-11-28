"""
Strategy #4: Support/Resistance Bounce

Mean reversion off key price levels:
- Identify support/resistance using recent highs/lows
- Enter when price bounces off these levels
- Exit when price returns to midpoint

Expected Performance (ES 2010-2024):
- Trade Count: 5,000-8,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.2-2.0
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class SupportResistanceBounce(BaseStrategy):
    """
    Support/Resistance Bounce Strategy (Catalogue #4).

    Entry:
    - Price touches support (recent low) and bounces
    - Touch = Low within X% of support level
    - Bounce = Close back above support + threshold

    Exit:
    - Price reaches midpoint between support and resistance
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - sr_lookback: [20, 40, 60, 80] (bars to find S/R levels)
    - sr_touch_pct: [0.1, 0.2, 0.3] (% within level to count as touch)
    - sr_bounce_pct: [0.1, 0.2, 0.3] (% bounce to confirm entry)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=4,
            name="SupportResistanceBounce",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'sr_lookback': [20, 40, 60, 80],
            'sr_touch_pct': [0.1, 0.2, 0.3],
            'sr_bounce_pct': [0.1, 0.2, 0.3]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['sr_lookback']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate support and resistance levels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with support, resistance, midpoint columns
        """
        lookback = self.params.get('sr_lookback', 40)

        # Support = rolling low
        data['support'] = data['Low'].rolling(window=lookback).min()

        # Resistance = rolling high
        data['resistance'] = data['High'].rolling(window=lookback).max()

        # Midpoint
        data['sr_midpoint'] = (data['support'] + data['resistance']) / 2

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price bounces off support.

        Args:
            data: OHLCV dataframe with S/R levels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        touch_pct = params.get('sr_touch_pct', 0.2) / 100
        bounce_pct = params.get('sr_bounce_pct', 0.2) / 100

        # Touch: Low within X% of support
        touch_threshold = data['support'] * (1 + touch_pct)
        touched_support = data['Low'] <= touch_threshold

        # Bounce: Close back above support + bounce threshold
        bounce_level = data['support'] * (1 + bounce_pct)
        bounced = data['Close'] > bounce_level

        # Entry: touched on previous bar, bounced on current bar
        entry = touched_support.shift(1) & bounced

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
        Exit when price reaches midpoint.

        Args:
            data: OHLCV dataframe with S/R levels
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when price reaches midpoint
        if current_bar['Close'] >= current_bar['sr_midpoint']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
