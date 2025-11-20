"""
Strategy #3: Three Bar Reversal

Classic price action pattern:
- Enter after 3 consecutive bars in one direction followed by reversal
- Simple momentum exhaustion signal

Expected Performance (ES 2010-2024):
- Trade Count: 5,000-8,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ThreeBarReversal(BaseStrategy):
    """
    Three Bar Reversal Strategy (Catalogue #3).

    Entry:
    - 3 consecutive down bars (lower closes)
    - Current bar closes up (reversal)
    - Optionally: reversal bar must close above X% from low

    Exit:
    - 3 consecutive up bars (momentum exhausted)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - tbr_reversal_pct: [0.0, 0.2, 0.5, 1.0] (% from low for reversal confirmation)
    - tbr_exit_bars: [2, 3, 4] (consecutive bars to exit)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=3,
            name="ThreeBarReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'tbr_reversal_pct': [0.0, 0.2, 0.5, 1.0],
            'tbr_exit_bars': [2, 3, 4]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consecutive down bars and reversal strength.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with down_streak and reversal_strength columns
        """
        # Down bar = Close < previous Close
        down_bar = data['Close'] < data['Close'].shift(1)

        # Count consecutive down bars
        down_streak = down_bar.astype(int).groupby((~down_bar).cumsum()).cumsum()
        data['down_streak'] = down_streak

        # Up bar = Close > previous Close
        up_bar = data['Close'] > data['Close'].shift(1)
        up_streak = up_bar.astype(int).groupby((~up_bar).cumsum()).cumsum()
        data['up_streak'] = up_streak

        # Reversal strength = % from bar low to close
        data['reversal_strength'] = ((data['Close'] - data['Low']) / data['Low']) * 100

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals after 3 down bars + reversal.

        Args:
            data: OHLCV dataframe with streak info
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        reversal_pct = params.get('tbr_reversal_pct', 0.2)

        # Entry conditions:
        # 1. Had 3+ consecutive down bars (shifted by 1)
        had_down_streak = data['down_streak'].shift(1) >= 3

        # 2. Current bar is up (reversal)
        current_up = data['Close'] > data['Close'].shift(1)

        # 3. Reversal strength sufficient
        strong_reversal = data['reversal_strength'] >= reversal_pct

        entry = had_down_streak & current_up & strong_reversal

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
        Exit after N consecutive up bars (momentum exhausted).

        Args:
            data: OHLCV dataframe with streak info
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        exit_bars = params.get('tbr_exit_bars', 3)
        current_bar = data.iloc[current_idx]

        # Exit when we see N consecutive up bars
        if current_bar['up_streak'] >= exit_bars:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
