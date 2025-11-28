"""
Strategy #51: Consecutive Close<Low[1] Mean Reversion

Contrarian strategy for overextended bearish moves:
- Enter when consecutive bars of close < low[1] reaches threshold
- Exit when close > high[1]
- Counts aggressive down bars where each close breaks prior low

Botnet101 strategy - works on any timeframe from intraday to daily.

Expected Performance:
- Identifies overextended bearish moves
- Optional EMA filter
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ConsecutiveCloseLow(BaseStrategy):
    """
    Consecutive Close<Low[1] Mean Reversion Strategy (Catalogue #51).

    Entry:
    - N consecutive bars where close < low[1]
    - Indicates aggressive selling pressure
    - Optional: Price above EMA filter

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - threshold: [2, 3, 4]
    - ema_period: [100, 150, 200]
    - use_ema_filter: [True, False]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=51,
            name="ConsecutiveCloseLow",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'threshold': [2, 3, 4],
            'ema_period': [100, 150, 200],
            'use_ema_filter': [True, False]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ema_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consecutive close < low[1] and EMA.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with consecutive_close_low and ema columns
        """
        ema_period = self.params.get('ema_period', 200)

        # EMA filter
        data['ema'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

        # Close < low[1] (aggressive down bar)
        data['close_below_prev_low'] = data['Close'] < data['Low'].shift(1)

        # Count consecutive close < low[1]
        consecutive = []
        count = 0
        for val in data['close_below_prev_low']:
            if val:
                count += 1
            else:
                count = 0
            consecutive.append(count)

        data['consecutive_close_low'] = consecutive

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals after N consecutive close < low[1].

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('threshold', 3)
        use_ema_filter = params.get('use_ema_filter', False)

        # Entry: Consecutive close < low[1] >= threshold
        entry = data['consecutive_close_low'] >= threshold

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
