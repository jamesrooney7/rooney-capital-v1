"""
Strategy #9: ADX Trend Strength

Trend-following using Average Directional Index:
- Enter when ADX shows strong trend (>25) and +DI crosses above -DI
- Exit when trend weakens (ADX falls) or directional indicator reverses

Wilder's ADX system for measuring trend strength.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_adx(data: pd.DataFrame, period: int = 14) -> tuple:
    """
    Calculate ADX, +DI, and -DI.

    Args:
        data: OHLCV dataframe
        period: ADX period

    Returns:
        Tuple of (adx, plus_di, minus_di) Series
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth using Wilder's method (exponential moving average)
    alpha = 1 / period

    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()

    # Calculate directional indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    # Avoid division by zero
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx, plus_di, minus_di


class ADXTrendStrength(BaseStrategy):
    """
    ADX Trend Strength Strategy (Catalogue #9).

    Entry:
    - ADX above threshold (strong trend)
    - +DI crosses above -DI (bullish trend)

    Exit:
    - ADX falls below threshold (trend weakening)
    - OR +DI crosses below -DI (trend reversal)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - adx_period: [14, 21, 28]
    - adx_threshold: [20, 25, 30]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=9,
            name="ADXTrendStrength",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'adx_period': [14, 21, 28],
            'adx_threshold': [20, 25, 30]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['adx_period']) * 2 + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX and directional indicators.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ADX, +DI, -DI columns
        """
        period = self.params.get('adx_period', 14)

        adx, plus_di, minus_di = calculate_adx(data, period)

        data['adx'] = adx
        data['plus_di'] = plus_di
        data['minus_di'] = minus_di

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on strong trend with bullish direction.

        Args:
            data: OHLCV dataframe with ADX indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('adx_threshold', 25)

        # Entry conditions:
        # 1. ADX above threshold (strong trend)
        strong_trend = data['adx'] > threshold

        # 2. +DI crosses above -DI (bullish)
        plus_di_prev = data['plus_di'].shift(1).fillna(0)
        minus_di_prev = data['minus_di'].shift(1).fillna(0)
        di_cross = (data['plus_di'] > data['minus_di']) & (plus_di_prev <= minus_di_prev)

        entry = strong_trend & di_cross

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
        Exit when trend weakens or reverses.

        Args:
            data: OHLCV dataframe with ADX
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        threshold = params.get('adx_threshold', 25)
        current_bar = data.iloc[current_idx]

        # Exit 1: ADX falls below threshold (trend weakening)
        if current_bar['adx'] < threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: +DI crosses below -DI (trend reversal)
        if current_bar['plus_di'] < current_bar['minus_di']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
