"""
Strategy #17: Moving Average Crossover

Classic dual moving average crossover:
- Enter when fast MA crosses above slow MA
- Exit when fast MA crosses below slow MA

One of the most widely studied and traded strategies.

Expected Performance (ES 2010-2024):
- Trade Count: Variable (depends on parameters)
- Raw Sharpe: 0.2-0.4
- ML Sharpe: 0.8-1.5
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma


class MACross(BaseStrategy):
    """
    Moving Average Crossover Strategy (Catalogue #17).

    Entry:
    - Fast MA crosses above Slow MA

    Exit:
    - Fast MA crosses below Slow MA
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ma_fast: [5, 10, 15, 20]
    - ma_slow: [30, 50, 75, 100]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=17,
            name="MACross",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ma_fast': [5, 10, 15, 20],
            'ma_slow': [30, 50, 75, 100]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_slow = max(self.param_grid['ma_slow'])
        return max_slow + 10  # +10 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fast and slow moving averages.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ma_fast and ma_slow columns added
        """
        ma_fast = self.params.get('ma_fast', 10)
        ma_slow = self.params.get('ma_slow', 50)

        # Only calculate if fast < slow (valid configuration)
        if ma_fast >= ma_slow:
            # Invalid config - set to NaN
            data['ma_fast'] = np.nan
            data['ma_slow'] = np.nan
        else:
            data['ma_fast'] = calculate_sma(data['Close'], period=ma_fast)
            data['ma_slow'] = calculate_sma(data['Close'], period=ma_slow)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when fast MA crosses above slow MA.

        Args:
            data: OHLCV dataframe with MAs
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        ma_fast = params.get('ma_fast', 10)
        ma_slow = params.get('ma_slow', 50)

        # Skip invalid configurations
        if ma_fast >= ma_slow:
            return pd.Series(False, index=data.index)

        # Entry: Fast MA crosses above Slow MA
        # Current bar: fast > slow
        # Previous bar: fast <= slow
        fast_above_slow = data['ma_fast'] > data['ma_slow']
        fast_was_below = data['ma_fast'].shift(1) <= data['ma_slow'].shift(1)

        entry = fast_above_slow & fast_was_below

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
        Exit when fast MA crosses below slow MA.

        Args:
            data: OHLCV dataframe with MAs
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when fast MA crosses below slow MA
        if current_bar['ma_fast'] < current_bar['ma_slow']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
