"""
Strategy #23: Gap Fill

Intraday mean reversion on gaps:
- Enter when gap > threshold (expecting reversion)
- Exit when gap 50% filled (or at target)

Common pattern in liquid markets - gaps tend to fill.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 0.9-1.6
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class GapFill(BaseStrategy):
    """
    Gap Fill Strategy (Catalogue #23).

    Entry:
    - Gap > gap_threshold% (Open vs previous Close)
    - For down gaps (gap < 0): Enter long expecting fill

    Exit:
    - Gap filled by gap_fill_target% (e.g., 50% filled)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - gap_threshold: [0.5, 1.0, 1.5, 2.0]  (percentage)
    - gap_fill_target: [0.3, 0.5, 0.7]  (fraction of gap to fill)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=23,
            name="GapFill",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'gap_threshold': [0.5, 1.0, 1.5, 2.0],
            'gap_fill_target': [0.3, 0.5, 0.7]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 2  # Just need previous close

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap size and fill level.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with gap_pct, prev_close, gap_fill_price columns
        """
        # Previous close
        data['prev_close'] = data['Close'].shift(1)

        # Gap percentage: (Open - prev_close) / prev_close
        data['gap_pct'] = ((data['Open'] - data['prev_close']) / data['prev_close']) * 100

        # Gap fill target price (for exit calculation)
        # For down gap: prev_close - (gap_fill_target × gap_size)
        # For up gap: prev_close + (gap_fill_target × gap_size)
        gap_fill_target = self.params.get('gap_fill_target', 0.5)
        gap_size = data['Open'] - data['prev_close']
        data['gap_fill_price'] = data['prev_close'] + (gap_fill_target * gap_size)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when gap exceeds threshold.

        Args:
            data: OHLCV dataframe with gap_pct
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        gap_threshold = params.get('gap_threshold', 1.0)

        # Entry: Gap down > threshold (negative gap)
        # We enter long expecting price to rise back towards previous close
        entry = data['gap_pct'] < -gap_threshold

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
        Exit when gap is partially filled.

        Args:
            data: OHLCV dataframe with gap info
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        entry_bar = data.iloc[entry_idx]
        current_bar = data.iloc[current_idx]

        # Get gap fill target price (calculated at entry)
        gap_fill_price = entry_bar['gap_fill_price']
        current_price = current_bar['Close']

        # Exit when price reaches or exceeds gap fill target
        # (for down gap trade, gap_fill_price is above entry)
        if current_price >= gap_fill_price:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_price
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
