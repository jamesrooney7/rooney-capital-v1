"""
Strategy #13: CCI (Commodity Channel Index)

Momentum oscillator identifying cyclical trends:
- Enter when CCI crosses above oversold (-100)
- Exit when CCI crosses below overbought (+100)

Widely used in commodity and futures trading.

Expected Performance (ES 2010-2024):
- Trade Count: 5,000-8,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class CCIStrategy(BaseStrategy):
    """
    CCI (Commodity Channel Index) Strategy (Catalogue #13).

    Entry:
    - CCI crosses above oversold level (-100)
    - CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)

    Exit:
    - CCI crosses below overbought level (+100)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - cci_length: [14, 20, 30, 40]
    - cci_oversold: [-150, -100, -50]
    - cci_overbought: [50, 100, 150]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=13,
            name="CCIStrategy",
            archetype="momentum",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'cci_length': [14, 20, 30, 40],
            'cci_oversold': [-150, -100, -50],
            'cci_overbought': [50, 100, 150]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['cci_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CCI.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with cci column
        """
        length = self.params.get('cci_length', 20)

        # Typical Price
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3

        # Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=length).mean()

        # Mean Absolute Deviation
        mad = typical_price.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        # CCI = (TP - SMA) / (0.015 × MAD)
        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        data['cci'] = cci

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when CCI crosses above oversold.

        Args:
            data: OHLCV dataframe with CCI
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        oversold = params.get('cci_oversold', -100)

        # Entry: CCI crosses above oversold
        above_oversold = data['cci'] > oversold
        was_below = data['cci'].shift(1) <= oversold

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
        Exit when CCI crosses below overbought.

        Args:
            data: OHLCV dataframe with CCI
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        overbought = params.get('cci_overbought', 100)
        current_bar = data.iloc[current_idx]

        # Exit when CCI crosses below overbought
        if current_bar['cci'] < overbought:
            if current_idx > 0:
                prev_cci = data.iloc[current_idx - 1]['cci']
                if prev_cci >= overbought:
                    return TradeExit(
                        exit=True,
                        exit_type='signal',
                        exit_price=current_bar['Close']
                    )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
