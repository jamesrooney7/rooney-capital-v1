"""
Strategy #20: Fibonacci Retracement

Mean reversion using Fibonacci levels:
- Identify swing high/low over lookback period
- Enter when price retraces to Fib level (38.2%, 50%, 61.8%)
- Exit when price returns to swing high or Fib level broken

Classic support/resistance levels based on Fibonacci ratios.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class FibonacciRetracement(BaseStrategy):
    """
    Fibonacci Retracement Strategy (Catalogue #20).

    Entry:
    - Identify swing high and low over lookback period
    - Enter when price retraces to Fibonacci level from high
    - Levels: 38.2%, 50%, 61.8%

    Exit:
    - Price reaches swing high (target)
    - OR price breaks below deeper Fib level (failure)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - fib_lookback: [20, 30, 50] (period to find swing high/low)
    - fib_level: [0.382, 0.500, 0.618] (retracement level for entry)
    - fib_tolerance: [0.01, 0.02, 0.03] (% tolerance around level)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=20,
            name="FibonacciRetracement",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'fib_lookback': [20, 30, 50],
            'fib_level': [0.382, 0.500, 0.618],
            'fib_tolerance': [0.01, 0.02, 0.03]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['fib_lookback']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with swing points and Fib levels
        """
        lookback = self.params.get('fib_lookback', 30)

        # Find swing high and low over lookback period
        data['swing_high'] = data['High'].rolling(window=lookback).max()
        data['swing_low'] = data['Low'].rolling(window=lookback).min()

        # Calculate range
        data['swing_range'] = data['swing_high'] - data['swing_low']

        # Fibonacci retracement levels (from high)
        data['fib_382'] = data['swing_high'] - 0.382 * data['swing_range']
        data['fib_500'] = data['swing_high'] - 0.500 * data['swing_range']
        data['fib_618'] = data['swing_high'] - 0.618 * data['swing_range']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price touches Fib level.

        Args:
            data: OHLCV dataframe with Fib levels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        fib_level = params.get('fib_level', 0.500)
        tolerance = params.get('fib_tolerance', 0.02)

        # Get the appropriate Fib level column
        if fib_level == 0.382:
            level_col = 'fib_382'
        elif fib_level == 0.618:
            level_col = 'fib_618'
        else:  # 0.500
            level_col = 'fib_500'

        fib_price = data[level_col]

        # Entry conditions:
        # 1. Price is near the Fib level (within tolerance)
        price_at_level = (
            (data['Low'] <= fib_price * (1 + tolerance)) &
            (data['High'] >= fib_price * (1 - tolerance))
        )

        # 2. We have a valid swing range (not too small)
        valid_range = data['swing_range'] > data['swing_range'].rolling(50).mean() * 0.3

        # 3. Price bouncing off level (low below, close above)
        bouncing = data['Close'] > fib_price

        entry = price_at_level & valid_range & bouncing

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
        Exit when price reaches swing high or breaks deeper level.

        Args:
            data: OHLCV dataframe with Fib levels
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Exit 1: Price reaches swing high (target achieved)
        if current_bar['Close'] >= entry_bar['swing_high'] * 0.98:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Price breaks below swing low (pattern failure)
        if current_bar['Close'] < entry_bar['swing_low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
