"""
Strategy #26: Inside Bar Breakout

Price action pattern - volatility contraction followed by expansion:
- Enter when price breaks out of inside bar range
- Exit on opposite breakout or time-based

Inside bar = bar with high < prev high AND low > prev low (consolidation).

Expected Performance (ES 2010-2024):
- Trade Count: 4,000-7,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.1-1.9
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class InsideBarBreakout(BaseStrategy):
    """
    Inside Bar Breakout Strategy (Catalogue #26).

    Entry:
    - Identify inside bar (High < prev High AND Low > prev Low)
    - Enter on breakout above inside bar high

    Exit:
    - Breakdown below inside bar low
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ib_min_inside_bars: [1, 2, 3] (consecutive inside bars required)
    - ib_breakout_pct: [0.0, 0.1, 0.2] (% above high to confirm breakout)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=26,
            name="InsideBarBreakout",
            archetype="breakout",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ib_min_inside_bars': [1, 2, 3],
            'ib_breakout_pct': [0.0, 0.1, 0.2]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify inside bars and breakout levels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with inside_bar, ib_high, ib_low columns
        """
        # Inside bar: High < prev High AND Low > prev Low
        inside_bar = (
            (data['High'] < data['High'].shift(1)) &
            (data['Low'] > data['Low'].shift(1))
        )

        data['inside_bar'] = inside_bar

        # Breakout levels (from most recent inside bar or mother bar)
        data['ib_high'] = data['High'].shift(1)
        data['ib_low'] = data['Low'].shift(1)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on inside bar breakout.

        Args:
            data: OHLCV dataframe with inside bar info
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        min_inside_bars = params.get('ib_min_inside_bars', 1)
        breakout_pct = params.get('ib_breakout_pct', 0.0) / 100

        # Check if we had N consecutive inside bars
        inside_count = data['inside_bar'].rolling(window=min_inside_bars).sum()
        had_inside_bars = inside_count.shift(1) >= min_inside_bars

        # Breakout: Close above previous bar's high (with optional buffer)
        breakout_level = data['ib_high'] * (1 + breakout_pct)
        breakout = data['Close'] > breakout_level

        entry = had_inside_bars & breakout

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
        Exit on breakdown below inside bar low.

        Args:
            data: OHLCV dataframe with inside bar info
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Exit when price breaks below entry bar's low
        if current_bar['Close'] < entry_bar['ib_low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
