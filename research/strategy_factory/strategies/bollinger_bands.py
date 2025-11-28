"""
Strategy #1: Bollinger Bands Mean Reversion

Classic Bollinger Bands reversal strategy:
- Enter when price closes below lower band
- Exit when price closes above middle band

Well-documented strategy with consistent performance across markets.

Expected Performance (ES 2010-2024):
- Trade Count: 12,000+
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-2.0+
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_bollinger_bands


class BollingerBands(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy (Catalogue #1).

    Entry:
    - Close < Lower Bollinger Band

    Exit:
    - Close > Middle Band (SMA)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - bb_length: [15, 20, 25, 30]
    - bb_stddev: [1.5, 2.0, 2.5, 3.0]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=1,
            name="BollingerBands",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'bb_length': [15, 20, 25, 30],
            'bb_stddev': [1.5, 2.0, 2.5, 3.0]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_length = max(self.param_grid['bb_length'])
        return max_length + 10  # +10 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with bb_upper, bb_middle, bb_lower columns added
        """
        bb_length = self.params.get('bb_length', 20)
        bb_stddev = self.params.get('bb_stddev', 2.0)

        upper, middle, lower = calculate_bollinger_bands(
            data['Close'],
            period=bb_length,
            std_dev=bb_stddev
        )

        data['bb_upper'] = upper
        data['bb_middle'] = middle
        data['bb_lower'] = lower

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price closes below lower band.

        Args:
            data: OHLCV dataframe with Bollinger Bands
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close below lower Bollinger Band
        entry = data['Close'] < data['bb_lower']

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
        Exit when price closes above middle band.

        Args:
            data: OHLCV dataframe with Bollinger Bands
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        current_price = current_bar['Close']
        bb_middle = current_bar['bb_middle']

        # Exit when price closes above middle band
        if current_price > bb_middle:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_price
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
