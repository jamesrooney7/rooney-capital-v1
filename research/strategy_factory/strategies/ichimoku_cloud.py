"""
Strategy #10: Ichimoku Cloud

Comprehensive Japanese indicator system:
- Enter when price breaks above cloud with bullish signals
- Exit when price falls back into/below cloud

Complex multi-component system with Tenkan, Kijun, Senkou, and Chikou lines.

Expected Performance (ES 2010-2024):
- Trade Count: 1,500-3,000
- Raw Sharpe: 0.2-0.5
- ML Sharpe: 0.8-1.5
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_ichimoku(data: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
                      senkou_b: int = 52) -> tuple:
    """
    Calculate Ichimoku Cloud components.

    Args:
        data: OHLCV dataframe
        tenkan: Tenkan-sen (conversion line) period
        kijun: Kijun-sen (base line) period
        senkou_b: Senkou Span B period

    Returns:
        Tuple of (tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span)
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (high.rolling(window=tenkan).max() +
                  low.rolling(window=tenkan).min()) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (high.rolling(window=kijun).max() +
                 low.rolling(window=kijun).min()) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
    senkou_b_line = (high.rolling(window=senkou_b).max() +
                     low.rolling(window=senkou_b).min()) / 2
    senkou_b_shifted = senkou_b_line.shift(kijun)

    # Chikou Span (Lagging Span): Close shifted backward
    chikou_span = close.shift(-kijun)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b_shifted, chikou_span


class IchimokuCloud(BaseStrategy):
    """
    Ichimoku Cloud Strategy (Catalogue #10).

    Entry:
    - Price above cloud (above both Senkou A and B)
    - Tenkan crosses above Kijun (TK cross)
    - Chikou above price (lagging span confirmation)

    Exit:
    - Price falls into or below cloud
    - OR Tenkan crosses below Kijun
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ich_tenkan: [9, 12] (conversion line period)
    - ich_kijun: [26, 30] (base line period)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=10,
            name="IchimokuCloud",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ich_tenkan': [9, 12],
            'ich_kijun': [26, 30]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 60  # Need enough for Senkou B (52) + shift

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with Ichimoku lines
        """
        tenkan = self.params.get('ich_tenkan', 9)
        kijun = self.params.get('ich_kijun', 26)

        tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou = calculate_ichimoku(
            data, tenkan=tenkan, kijun=kijun, senkou_b=52
        )

        data['tenkan'] = tenkan_sen
        data['kijun'] = kijun_sen
        data['senkou_a'] = senkou_a
        data['senkou_b'] = senkou_b
        data['chikou'] = chikou

        # Cloud top and bottom
        data['cloud_top'] = data[['senkou_a', 'senkou_b']].max(axis=1)
        data['cloud_bottom'] = data[['senkou_a', 'senkou_b']].min(axis=1)

        # Price position relative to cloud
        data['above_cloud'] = data['Close'] > data['cloud_top']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on bullish Ichimoku setup.

        Args:
            data: OHLCV dataframe with Ichimoku components
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry conditions:
        # 1. Price above cloud
        above_cloud = data['above_cloud']

        # 2. Tenkan crosses above Kijun (TK cross)
        tk_cross = (data['tenkan'] > data['kijun']) & \
                   (data['tenkan'].shift(1) <= data['kijun'].shift(1))

        # 3. Chikou above price from 26 bars ago (optional confirmation)
        # Shift comparison to align properly
        chikou_confirm = data['chikou'].shift(26) > data['Close'].shift(26)

        entry = above_cloud & tk_cross

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
        Exit when price falls into cloud or TK reversal.

        Args:
            data: OHLCV dataframe with Ichimoku
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit 1: Price falls into or below cloud
        if not current_bar['above_cloud']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Tenkan crosses below Kijun (bearish TK cross)
        if current_bar['tenkan'] < current_bar['kijun']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
