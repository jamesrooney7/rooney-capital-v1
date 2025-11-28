"""
Strategy #52: ATR Buy the Dip Mean Reversion

ATR-based oversold detection:
- Enter when close < smoothed ATR lower trigger
- Lower trigger = SMA(close - ATR×multiplier, smoothing_period)
- Exit when close > high[1]

Botnet101 strategy - timeframe agnostic.

Expected Performance:
- Dynamic threshold based on volatility
- Enters on extreme dips below volatility-adjusted level
- Optional EMA filter
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        data: OHLCV dataframe
        period: ATR period

    Returns:
        ATR series
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    # True Range is the max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is exponential moving average of TR
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr


class ATRBuyDip(BaseStrategy):
    """
    ATR Buy the Dip Mean Reversion Strategy (Catalogue #52).

    Entry:
    - Close < smoothed lower trigger
    - Lower trigger = SMA(close - ATR×multiplier, smoothing_period)
    - Optional: Price above EMA filter

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - atr_period: [14, 20, 28]
    - atr_multiplier: [0.5, 1.0, 1.5, 2.0]
    - smoothing_period: [5, 10, 15]
    - ema_period: [100, 150, 200]
    - use_ema_filter: [True, False]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=52,
            name="ATRBuyDip",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'atr_period': [14, 20, 28],
            'atr_multiplier': [0.5, 1.0, 1.5, 2.0],
            'smoothing_period': [5, 10, 15],
            'ema_period': [100, 150, 200],
            'use_ema_filter': [True, False]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ema_period']) + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR, lower trigger, and EMA.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with atr, lower_trigger, ema columns
        """
        atr_period = self.params.get('atr_period', 20)
        atr_multiplier = self.params.get('atr_multiplier', 1.0)
        smoothing_period = self.params.get('smoothing_period', 10)
        ema_period = self.params.get('ema_period', 200)

        # Calculate ATR
        data['atr'] = calculate_atr(data, period=atr_period)

        # Raw lower trigger: close - ATR×multiplier
        raw_trigger = data['Close'] - (data['atr'] * atr_multiplier)

        # Smoothed lower trigger
        data['lower_trigger'] = raw_trigger.rolling(window=smoothing_period).mean()

        # EMA filter
        data['ema'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when close < smoothed lower trigger.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        use_ema_filter = params.get('use_ema_filter', False)

        # Entry: Close < lower trigger (oversold)
        entry = data['Close'] < data['lower_trigger']

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
