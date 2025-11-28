"""
Strategy #46: 4 Bar Momentum Reversal Strategy

Momentum exhaustion strategy:
- Enter when close < reference close (N bars ago) for N consecutive bars
- Exit when close > high[1], confirming reversal
- Requires sustained bearish move before entry

Botnet101 strategy - works on any timeframe from intraday to daily.

Expected Performance:
- Momentum exhaustion
- Reference close lookback customizable
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class FourBarMomentumReversal(BaseStrategy):
    """
    4 Bar Momentum Reversal Strategy (Catalogue #46).

    Entry:
    - For N consecutive bars, close < close[N bars ago]
    - This indicates sustained bearish momentum

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - lookback: [3, 4, 5] (N bars)
    - buy_threshold: [3, 4, 5] (consecutive bars threshold)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=46,
            name="FourBarMomentumReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'lookback': [3, 4, 5],
            'buy_threshold': [3, 4, 5]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['lookback']) + max(self.param_grid['buy_threshold']) + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate reference close and consecutive condition counter.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ref_close and consecutive_below columns
        """
        lookback = self.params.get('lookback', 4)

        # Reference close (N bars ago)
        data['ref_close'] = data['Close'].shift(lookback)

        # Current close < reference close
        data['below_ref'] = data['Close'] < data['ref_close']

        # Count consecutive bars where close < ref_close
        # Reset counter when condition fails
        consecutive = []
        count = 0
        for val in data['below_ref']:
            if val:
                count += 1
            else:
                count = 0
            consecutive.append(count)

        data['consecutive_below'] = consecutive

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals after N consecutive bars below reference.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        buy_threshold = params.get('buy_threshold', 4)

        # Entry: Consecutive bars below reference >= threshold
        entry = data['consecutive_below'] >= buy_threshold

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
