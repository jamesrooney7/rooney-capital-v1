"""
Strategy #11: Stochastic RSI

Momentum oscillator combining Stochastic and RSI:
- Enter when StochRSI crosses above oversold level
- Exit when StochRSI crosses below overbought level

Faster and more sensitive than regular RSI.

Expected Performance (ES 2010-2024):
- Trade Count: 8,000-12,000
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_rsi


class StochasticRSI(BaseStrategy):
    """
    Stochastic RSI Strategy (Catalogue #11).

    Entry:
    - StochRSI crosses above oversold level (e.g., 20)
    - StochRSI = (RSI - RSI_low) / (RSI_high - RSI_low) × 100

    Exit:
    - StochRSI crosses below overbought level (e.g., 80)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - stochrsi_length: [14, 21, 28]
    - stochrsi_stoch_length: [14, 21]
    - stochrsi_oversold: [10, 20, 30]
    - stochrsi_overbought: [70, 80, 90]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=11,
            name="StochasticRSI",
            archetype="momentum",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'stochrsi_length': [14, 21, 28],
            'stochrsi_stoch_length': [14, 21],
            'stochrsi_oversold': [10, 20, 30],
            'stochrsi_overbought': [70, 80, 90]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_rsi = max(self.param_grid['stochrsi_length'])
        max_stoch = max(self.param_grid['stochrsi_stoch_length'])
        return max_rsi + max_stoch + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic RSI.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with stochrsi column
        """
        rsi_length = self.params.get('stochrsi_length', 14)
        stoch_length = self.params.get('stochrsi_stoch_length', 14)

        # Calculate RSI
        rsi = calculate_rsi(data['Close'], rsi_length)

        # Calculate Stochastic of RSI
        rsi_low = rsi.rolling(window=stoch_length).min()
        rsi_high = rsi.rolling(window=stoch_length).max()

        stochrsi = ((rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)) * 100

        data['stochrsi'] = stochrsi

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when StochRSI crosses above oversold.

        Args:
            data: OHLCV dataframe with StochRSI
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        oversold = params.get('stochrsi_oversold', 20)

        # Entry: StochRSI crosses above oversold
        above_oversold = data['stochrsi'] > oversold
        was_below = data['stochrsi'].shift(1) <= oversold

        entry = above_oversold & was_below

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
        Exit when StochRSI crosses below overbought.

        Args:
            data: OHLCV dataframe with StochRSI
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        overbought = params.get('stochrsi_overbought', 80)
        current_bar = data.iloc[current_idx]

        # Exit when StochRSI crosses below overbought
        if current_bar['stochrsi'] < overbought:
            # Check if it was above overbought recently
            if current_idx > 0:
                prev_stochrsi = data.iloc[current_idx - 1]['stochrsi']
                if prev_stochrsi >= overbought:
                    return TradeExit(
                        exit=True,
                        exit_type='signal',
                        exit_price=current_bar['Close']
                    )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
