"""
Strategy #54: 10 Bar High Breakout with Low IBS

Breakout strategy with pullback confirmation:
- Enter when high > highest(high, N) breaking out AND low IBS
- Low IBS suggests pullback/consolidation before breakout
- Exit when close > high[1]

Botnet101 strategy - combines momentum with mean reversion filter.

Expected Performance:
- New N-bar high with low IBS suggests consolidation before breakout
- Works on any timeframe
- Optional EMA filter
- Typical hold period: Medium-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_ibs(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Internal Bar Strength (IBS).

    IBS = (Close - Low) / (High - Low)

    Args:
        data: OHLCV dataframe

    Returns:
        IBS series
    """
    range_hl = data['High'] - data['Low']
    range_hl = range_hl.replace(0, np.nan)
    ibs = (data['Close'] - data['Low']) / range_hl
    return ibs.fillna(0.5)


class TenBarHighBreakout(BaseStrategy):
    """
    10 Bar High Breakout with Low IBS Strategy (Catalogue #54).

    Entry:
    - High > highest high of previous N bars (breakout)
    - IBS <= threshold (pullback confirmation)
    - Optional: Price above EMA filter

    Exit:
    - Close > high[1] (continuation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - lookback_period: [7, 10, 15, 20]
    - ibs_threshold: [0.1, 0.15, 0.2, 0.25]
    - ema_period: [100, 150, 200]
    - use_ema_filter: [True, False]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=54,
            name="TenBarHighBreakout",
            archetype="breakout",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'lookback_period': [7, 10, 15, 20],
            'ibs_threshold': [0.1, 0.15, 0.2, 0.25],
            'ema_period': [100, 150, 200],
            'use_ema_filter': [True, False]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ema_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate highest high, IBS, and EMA.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with highest_high, ibs, ema columns
        """
        lookback_period = self.params.get('lookback_period', 10)
        ema_period = self.params.get('ema_period', 200)

        # Highest high of previous N bars (exclude current bar)
        data['highest_high'] = data['High'].shift(1).rolling(window=lookback_period).max()

        # IBS
        data['ibs'] = calculate_ibs(data)

        # EMA filter
        data['ema'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on breakout with low IBS.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        ibs_threshold = params.get('ibs_threshold', 0.15)
        use_ema_filter = params.get('use_ema_filter', False)

        # Condition 1: High breaks above highest high (breakout)
        breakout = data['High'] > data['highest_high']

        # Condition 2: Low IBS (pullback/consolidation)
        low_ibs = data['ibs'] <= ibs_threshold

        # Entry: Both conditions met
        entry = breakout & low_ibs

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

            # Exit when close > high[1] (continuation)
            if current_bar['Close'] > prev_high:
                return TradeExit(
                    exit=True,
                    exit_type='signal',
                    exit_price=current_bar['Close']
                )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
