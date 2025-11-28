"""
Strategy #45: Internal Bar Strength (IBS) Strategy

Pure IBS mean reversion strategy:
- Enter when IBS < threshold (close near bar's low)
- Exit when IBS > threshold (close near bar's high)
- Simple calculation, no warmup needed

Botnet101 strategy - best in ranging markets.

Expected Performance:
- Works on any timeframe
- Simple and fast
- Typical hold period: Short-term
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_ibs(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Internal Bar Strength (IBS).

    IBS = (Close - Low) / (High - Low)

    Measures where the close is within the bar's range.
    0 = closed at low, 1 = closed at high

    Args:
        data: OHLCV dataframe

    Returns:
        IBS series
    """
    range_hl = data['High'] - data['Low']
    # Avoid division by zero
    range_hl = range_hl.replace(0, np.nan)
    ibs = (data['Close'] - data['Low']) / range_hl
    return ibs.fillna(0.5)  # Fill doji bars (no range) with neutral value


class IBSStrategy(BaseStrategy):
    """
    Pure IBS Mean Reversion Strategy (Catalogue #45).

    Entry:
    - IBS < threshold (default 0.2) - close near bar's low

    Exit:
    - IBS > threshold (default 0.8) - close near bar's high
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ibs_buy_threshold: [0.1, 0.15, 0.2, 0.25]
    - ibs_sell_threshold: [0.7, 0.75, 0.8, 0.85]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=45,
            name="IBSStrategy",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ibs_buy_threshold': [0.1, 0.15, 0.2, 0.25],
            'ibs_sell_threshold': [0.7, 0.75, 0.8, 0.85]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 5  # Minimal warmup needed

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IBS.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ibs column
        """
        data['ibs'] = calculate_ibs(data)
        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when IBS < threshold.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        ibs_buy_threshold = params.get('ibs_buy_threshold', 0.2)

        # Entry: IBS < threshold (close near low)
        entry = data['ibs'] < ibs_buy_threshold

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
        Exit when IBS > threshold.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        ibs_sell_threshold = params.get('ibs_sell_threshold', 0.8)
        current_bar = data.iloc[current_idx]

        # Exit when IBS > threshold (close near high)
        if current_bar['ibs'] > ibs_sell_threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
