"""
Strategy #48: Consecutive Bars Above/Below EMA Buy the Dip Strategy

MA-based dip buying:
- Enter when close remains below MA for N consecutive bars
- Exit when close > high[1]
- Customizable MA type (SMA or EMA) and length

Botnet101 strategy - works on any timeframe when price oscillates around short-term MA.

Expected Performance:
- Counts consecutive bars below MA
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class ConsecutiveBarsEMA(BaseStrategy):
    """
    Consecutive Bars Below EMA Buy the Dip Strategy (Catalogue #48).

    Entry:
    - Close below MA for N consecutive bars
    - MA type: SMA or EMA (customizable)

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - consecutive_bars_threshold: [2, 3, 4]
    - ma_type: ['SMA', 'EMA']
    - ma_length: [5, 10, 20]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=48,
            name="ConsecutiveBarsEMA",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'consecutive_bars_threshold': [2, 3, 4],
            'ma_type': ['SMA', 'EMA'],
            'ma_length': [5, 10, 20]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['ma_length']) + max(self.param_grid['consecutive_bars_threshold']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average and consecutive bars below MA.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ma and consecutive_below_ma columns
        """
        ma_type = self.params.get('ma_type', 'SMA')
        ma_length = self.params.get('ma_length', 5)

        # Calculate moving average
        if ma_type == 'EMA':
            data['ma'] = data['Close'].ewm(span=ma_length, adjust=False).mean()
        else:  # SMA
            data['ma'] = data['Close'].rolling(window=ma_length).mean()

        # Current close < MA
        data['below_ma'] = data['Close'] < data['ma']

        # Count consecutive bars below MA
        consecutive = []
        count = 0
        for val in data['below_ma']:
            if val:
                count += 1
            else:
                count = 0
            consecutive.append(count)

        data['consecutive_below_ma'] = consecutive

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals after N consecutive bars below MA.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        consecutive_bars_threshold = params.get('consecutive_bars_threshold', 3)

        # Entry: Consecutive bars below MA >= threshold
        entry = data['consecutive_below_ma'] >= consecutive_bars_threshold

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
