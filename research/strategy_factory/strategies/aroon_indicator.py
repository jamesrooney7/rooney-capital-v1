"""
Strategy #33: Aroon Indicator

Trend strength using Aroon Up/Down oscillator:
- Aroon Up: Time since N-period high
- Aroon Down: Time since N-period low
- Enter when Aroon Up crosses above Aroon Down

Measures trend strength by tracking new highs/lows.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 0.9-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_aroon(data: pd.DataFrame, period: int = 25) -> tuple:
    """
    Calculate Aroon Up and Aroon Down.

    Args:
        data: OHLCV dataframe
        period: Lookback period

    Returns:
        Tuple of (aroon_up, aroon_down) Series
    """
    high = data['High']
    low = data['Low']

    # Days since highest high in period
    aroon_up = high.rolling(window=period).apply(
        lambda x: float(period - x.argmax()) / period * 100,
        raw=True
    )

    # Days since lowest low in period
    aroon_down = low.rolling(window=period).apply(
        lambda x: float(period - x.argmin()) / period * 100,
        raw=True
    )

    return aroon_up, aroon_down


class AroonIndicator(BaseStrategy):
    """
    Aroon Indicator Strategy (Catalogue #33).

    Entry:
    - Aroon Up crosses above Aroon Down (bullish)
    - Aroon Up above threshold (strong uptrend)
    - New highs being made recently

    Exit:
    - Aroon Down crosses above Aroon Up (bearish crossover)
    - OR Aroon Up falls below threshold (trend weakening)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - aroon_period: [14, 25, 50] (lookback period)
    - aroon_threshold: [50, 70, 90] (minimum Aroon Up for entry)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=33,
            name="AroonIndicator",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'aroon_period': [14, 25, 50],
            'aroon_threshold': [50, 70, 90]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['aroon_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Aroon Up and Down.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with Aroon indicators
        """
        period = self.params.get('aroon_period', 25)

        aroon_up, aroon_down = calculate_aroon(data, period)

        data['aroon_up'] = aroon_up
        data['aroon_down'] = aroon_down

        # Aroon oscillator (difference)
        data['aroon_osc'] = data['aroon_up'] - data['aroon_down']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on Aroon Up crossover.

        Args:
            data: OHLCV dataframe with Aroon
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('aroon_threshold', 70)

        # Entry conditions:
        # 1. Aroon Up crosses above Aroon Down
        up_cross = (data['aroon_up'] > data['aroon_down']) & \
                   (data['aroon_up'].shift(1) <= data['aroon_down'].shift(1))

        # 2. Aroon Up above threshold (strong uptrend)
        strong_up = data['aroon_up'] >= threshold

        entry = up_cross & strong_up

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
        Exit when Aroon reverses or weakens.

        Args:
            data: OHLCV dataframe with Aroon
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        threshold = params.get('aroon_threshold', 70)
        current_bar = data.iloc[current_idx]

        # Exit 1: Aroon Down crosses above Aroon Up (trend reversal)
        if current_bar['aroon_down'] > current_bar['aroon_up']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Aroon Up falls below threshold (trend weakening)
        if current_bar['aroon_up'] < threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
