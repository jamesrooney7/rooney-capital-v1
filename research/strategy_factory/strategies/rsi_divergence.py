"""
Strategy #22: RSI Divergence

Advanced mean reversion using divergence:
- Enter when price makes lower low but RSI makes higher low (bullish divergence)
- Exit when RSI reaches overbought or divergence negated

Classic divergence trading strategy.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.4-0.7
- ML Sharpe: 1.2-2.0
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_rsi


class RSIDivergence(BaseStrategy):
    """
    RSI Divergence Strategy (Catalogue #22).

    Entry:
    - Bullish divergence: Price lower low + RSI higher low
    - Lookback window to find divergence

    Exit:
    - RSI reaches overbought level
    - OR divergence negated (new lower low)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - rsi_div_length: [14, 21, 28]
    - rsi_div_lookback: [5, 10, 20] (bars to search for divergence)
    - rsi_div_overbought: [65, 70, 75]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=22,
            name="RSIDivergence",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'rsi_div_length': [14, 21, 28],
            'rsi_div_lookback': [5, 10, 20],
            'rsi_div_overbought': [65, 70, 75]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_rsi = max(self.param_grid['rsi_div_length'])
        max_lookback = max(self.param_grid['rsi_div_lookback'])
        return max_rsi + max_lookback + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and identify divergences.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with rsi and divergence columns
        """
        rsi_length = self.params.get('rsi_div_length', 14)
        lookback = self.params.get('rsi_div_lookback', 10)

        # Calculate RSI
        data['rsi'] = calculate_rsi(data['Close'], rsi_length)

        # Find local lows in price and RSI
        data['price_low'] = data['Low'].rolling(window=lookback).min()
        data['rsi_low'] = data['rsi'].rolling(window=lookback).min()

        # Bullish divergence: price lower low + RSI higher low
        price_lower_low = data['Low'] < data['price_low'].shift(1)
        rsi_higher_low = data['rsi'] > data['rsi_low'].shift(1)

        data['bullish_divergence'] = price_lower_low & rsi_higher_low

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on bullish divergence.

        Args:
            data: OHLCV dataframe with RSI divergence
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Bullish divergence detected
        entry = data['bullish_divergence']

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
        Exit when RSI reaches overbought or divergence negated.

        Args:
            data: OHLCV dataframe with RSI
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        overbought = params.get('rsi_div_overbought', 70)
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Exit 1: RSI reaches overbought
        if current_bar['rsi'] >= overbought:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Divergence negated (new lower low below entry)
        if current_bar['Low'] < entry_bar['Low'] * 0.995:  # 0.5% lower
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
