"""
Strategy #47: Consecutive Bearish Candle Strategy

Simple consecutive down bar counter:
- Enter after N consecutive bars where close[i] < close[i-1]
- Exit when close > high[1]
- More flexible than 4-bar momentum

Botnet101 strategy - timeframe agnostic.

Expected Performance:
- Best in volatile markets with frequent swings
- Simple consecutive close counter
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ConsecutiveBearishCandle(BaseStrategy):
    """
    Consecutive Bearish Candle Strategy (Catalogue #47).

    Entry:
    - N consecutive bars where close < previous close
    - Identifies momentum exhaustion

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - consecutive_lookback: [2, 3, 4, 5]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=47,
            name="ConsecutiveBearishCandle",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'consecutive_lookback': [2, 3, 4, 5]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['consecutive_lookback']) + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consecutive down closes.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with down_close and consecutive_down columns
        """
        # Current close < previous close
        data['down_close'] = data['Close'] < data['Close'].shift(1)

        # Count consecutive down closes
        consecutive = []
        count = 0
        for val in data['down_close']:
            if val:
                count += 1
            else:
                count = 0
            consecutive.append(count)

        data['consecutive_down'] = consecutive

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
        consecutive_lookback = params.get('consecutive_lookback', 3)

        # Entry: Consecutive down closes >= threshold
        entry = data['consecutive_down'] >= consecutive_lookback

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
        Exit when close > high[1].

        Args:
            data: OHLCV dataframe
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Get previous bar's high
        if current_idx > 0:
            prev_high = data.iloc[current_idx - 1]['High']

            # Exit when close > high[1]
            if current_bar['Close'] > prev_high:
                return TradeExit(
                    exit=True,
                    exit_type='signal',
                    exit_price=current_bar['Close']
                )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
