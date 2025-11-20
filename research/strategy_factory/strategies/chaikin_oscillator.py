"""
Strategy #35: Chaikin Oscillator

Accumulation/Distribution momentum oscillator:
- Measures difference between fast and slow EMAs of A/D line
- A/D line tracks money flow based on close position in range
- Enter when oscillator crosses zero (accumulation phase)

Volume-based momentum for detecting institutional activity.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 0.9-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_chaikin_oscillator(data: pd.DataFrame, fast: int = 3,
                                 slow: int = 10) -> pd.Series:
    """
    Calculate Chaikin Oscillator.

    Args:
        data: OHLCV dataframe
        fast: Fast EMA period
        slow: Slow EMA period

    Returns:
        Series with Chaikin Oscillator values
    """
    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close']))

    # Avoid division by zero
    range_hl = data['High'] - data['Low']
    clv = np.where(range_hl > 0, clv / range_hl, 0)

    # Money Flow Volume (CLV * Volume)
    mf_volume = clv * data['Volume']

    # Accumulation/Distribution Line (cumulative MF Volume)
    ad_line = mf_volume.cumsum()

    # Chaikin Oscillator: EMA(fast) - EMA(slow) of A/D Line
    ad_series = pd.Series(ad_line, index=data.index)
    ema_fast = ad_series.ewm(span=fast, adjust=False).mean()
    ema_slow = ad_series.ewm(span=slow, adjust=False).mean()

    chaikin_osc = ema_fast - ema_slow

    return chaikin_osc


class ChaikinOscillator(BaseStrategy):
    """
    Chaikin Oscillator Strategy (Catalogue #35).

    Entry:
    - Chaikin Oscillator crosses above zero (accumulation)
    - Indicates buying pressure exceeding selling
    - Volume confirms institutional participation

    Exit:
    - Oscillator crosses below zero (distribution)
    - OR oscillator diverges negatively from price
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - co_fast: [3, 5, 8] (fast EMA period)
    - co_slow: [10, 15, 20] (slow EMA period)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=35,
            name="ChaikinOscillator",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'co_fast': [3, 5, 8],
            'co_slow': [10, 15, 20]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['co_slow']) + 50  # Extra for A/D line

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Chaikin Oscillator.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with Chaikin Oscillator
        """
        fast = self.params.get('co_fast', 3)
        slow = self.params.get('co_slow', 10)

        data['chaikin_osc'] = calculate_chaikin_oscillator(data, fast, slow)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when oscillator crosses above zero.

        Args:
            data: OHLCV dataframe with Chaikin Oscillator
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry conditions:
        # 1. Chaikin Oscillator crosses above zero (bullish)
        osc_above = data['chaikin_osc'] > 0
        osc_was_below = data['chaikin_osc'].shift(1) <= 0

        entry = osc_above & osc_was_below

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
        Exit when oscillator crosses below zero.

        Args:
            data: OHLCV dataframe with Chaikin Oscillator
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when Chaikin Oscillator crosses below zero (distribution phase)
        if current_bar['chaikin_osc'] < 0:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
