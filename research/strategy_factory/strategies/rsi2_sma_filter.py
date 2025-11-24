"""
Strategy #36: RSI(2) + 200 SMA Filter

Larry Connors' classic strategy:
- Only trade long when price > 200 SMA (trend filter)
- Enter when RSI(2) < 5 (extreme oversold)
- Exit when RSI(2) > 70 (overbought)

Combines trend following with mean reversion.

Expected Performance (ES 2010-2024):
- Trade Count: 15,000+
- Raw Sharpe: 0.4-0.6
- ML Sharpe: 1.2-2.5+
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_rsi, calculate_sma


class RSI2SMAFilter(BaseStrategy):
    """
    RSI(2) + 200 SMA Filter Strategy (Catalogue #36).

    Entry:
    - RSI(period) < oversold_threshold
    - AND Close > SMA(filter_length) [trend filter]

    Exit:
    - RSI(period) > overbought_threshold
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - rsi_length: [2, 3, 4]
    - rsi_oversold: [3, 5, 10]
    - rsi_overbought: [65, 70, 75]
    - sma_filter: [150, 200, 250]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=36,
            name="RSI2_SMAFilter",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'rsi_length': [2, 3, 4],
            'rsi_oversold': [3, 5, 10],
            'rsi_overbought': [65, 70, 75],
            'sma_filter': [150, 200, 250]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_sma = max(self.param_grid['sma_filter'])
        return max_sma + 10  # +10 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and SMA filter.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with 'rsi' and 'sma_filter' columns added
        """
        rsi_length = self.params.get('rsi_length', 2)
        sma_filter_length = self.params.get('sma_filter', 200)

        data['rsi'] = calculate_rsi(data['Close'], period=rsi_length)
        data['sma_filter'] = calculate_sma(data['Close'], period=sma_filter_length)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when RSI is oversold AND price above SMA.

        Args:
            data: OHLCV dataframe with 'rsi' and 'sma_filter'
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        rsi_oversold = params.get('rsi_oversold', 5)

        # Entry condition 1: RSI oversold
        rsi_condition = data['rsi'] < rsi_oversold

        # Entry condition 2: Price above SMA (trend filter)
        trend_condition = data['Close'] > data['sma_filter']

        # Entry: Both conditions met
        entry = rsi_condition & trend_condition

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
        Exit when RSI crosses above overbought threshold.

        Args:
            data: OHLCV dataframe with 'rsi'
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        rsi_overbought = params.get('rsi_overbought', 70)

        current_bar = data.iloc[current_idx]
        current_rsi = current_bar['rsi']

        # Exit when RSI crosses above overbought
        if current_rsi > rsi_overbought:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
