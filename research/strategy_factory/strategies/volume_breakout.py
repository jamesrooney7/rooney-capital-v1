"""
Strategy #29: Volume Breakout

Momentum strategy using volume confirmation:
- Enter on price breakout with above-average volume
- Volume surge indicates institutional participation
- Exit when momentum fades or volume dries up

Classic tape reading approach combining price and volume.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma


class VolumeBreakout(BaseStrategy):
    """
    Volume Breakout Strategy (Catalogue #29).

    Entry:
    - Price breaks above N-period high
    - Volume exceeds X times average volume
    - Confirms strong institutional buying

    Exit:
    - Price falls below entry bar low (failed breakout)
    - OR volume drops below threshold (participation fading)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - vb_lookback: [10, 20, 30] (period for price breakout)
    - vb_vol_mult: [1.5, 2.0, 2.5] (volume multiplier vs average)
    - vb_vol_period: [20, 30, 50] (period for average volume)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=29,
            name="VolumeBreakout",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'vb_lookback': [10, 20, 30],
            'vb_vol_mult': [1.5, 2.0, 2.5],
            'vb_vol_period': [20, 30, 50]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['vb_vol_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume and breakout indicators.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with volume metrics
        """
        lookback = self.params.get('vb_lookback', 20)
        vol_period = self.params.get('vb_vol_period', 30)

        # Price breakout level
        data['breakout_high'] = data['High'].rolling(window=lookback).max()

        # Average volume
        data['avg_volume'] = data['Volume'].rolling(window=vol_period).mean()

        # Volume ratio
        data['vol_ratio'] = data['Volume'] / data['avg_volume']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on price breakout with volume surge.

        Args:
            data: OHLCV dataframe with volume metrics
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        vol_mult = params.get('vb_vol_mult', 2.0)

        # Entry conditions:
        # 1. Price breaks above recent high
        price_breakout = data['Close'] > data['breakout_high'].shift(1)

        # 2. Volume exceeds threshold (strong participation)
        volume_surge = data['vol_ratio'] >= vol_mult

        # 3. Previous close was below the high (fresh breakout)
        was_below = data['Close'].shift(1) <= data['breakout_high'].shift(2)

        entry = price_breakout & volume_surge & was_below

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
        Exit when breakout fails or volume fades.

        Args:
            data: OHLCV dataframe with volume
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Exit 1: Price falls below entry bar low (failed breakout)
        if current_bar['Close'] < entry_bar['Low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Volume drops significantly (participation fading)
        if current_bar['vol_ratio'] < 0.8:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
