"""
Strategy #24: VWAP Reversion

Intraday mean reversion strategy:
- Enter when price deviates >N standard deviations from VWAP
- Exit when price returns to VWAP

Popular with institutional traders for execution algorithms.

Expected Performance (ES 2010-2024):
- Trade Count: 8,000+
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_vwap


class VWAPReversion(BaseStrategy):
    """
    VWAP Mean Reversion Strategy (Catalogue #24).

    Entry:
    - Price < VWAP - (N × std_dev)

    Exit:
    - Price returns to VWAP
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - vwap_std_threshold: [1.5, 2.0, 2.5, 3.0]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=24,
            name="VWAPReversion",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'vwap_std_threshold': [1.5, 2.0, 2.5, 3.0]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 20  # Need some bars to calculate VWAP std

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP and standard deviation bands.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with vwap, vwap_std columns added
        """
        # Calculate VWAP (reset daily for intraday data)
        data['vwap'] = calculate_vwap(data)

        # Calculate rolling standard deviation of (Price - VWAP)
        data['price_vwap_diff'] = data['Close'] - data['vwap']
        data['vwap_std'] = data['price_vwap_diff'].rolling(window=20).std()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price deviates from VWAP.

        Args:
            data: OHLCV dataframe with VWAP
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        vwap_std_threshold = params.get('vwap_std_threshold', 2.0)

        # Entry: Price below VWAP by N standard deviations
        lower_band = data['vwap'] - (vwap_std_threshold * data['vwap_std'])
        entry = data['Close'] < lower_band

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
        Exit when price returns to VWAP.

        Args:
            data: OHLCV dataframe with VWAP
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        current_price = current_bar['Close']
        vwap = current_bar['vwap']

        # Exit when price returns to VWAP (or above)
        if current_price >= vwap:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_price
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
