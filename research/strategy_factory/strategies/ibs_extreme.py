"""
Strategy #53: Internal Bar Strength (IBS) Extreme Oversold

Extreme IBS mean reversion strategy:
- Enter when IBS <= threshold (default 0.1) - close near bar's low
- Exit when IBS >= threshold (default 0.7) - close near bar's high
- More extreme version of IBS strategy #45

Botnet101 strategy - works on any timeframe for stocks/ETFs/futures.

Expected Performance:
- Uses lower threshold (0.1 vs 0.2) for fewer but higher-conviction signals
- Works on any timeframe
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


class IBSExtreme(BaseStrategy):
    """
    IBS Extreme Oversold Strategy (Catalogue #53).

    Entry:
    - IBS <= threshold (default 0.1) - extreme oversold
    - Close very near bar's low

    Exit:
    - IBS >= threshold (default 0.7) - close near bar's high
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - ibs_entry_threshold: [0.05, 0.1, 0.15]
    - ibs_exit_threshold: [0.65, 0.7, 0.75, 0.8]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=53,
            name="IBSExtreme",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'ibs_entry_threshold': [0.05, 0.1, 0.15],
            'ibs_exit_threshold': [0.65, 0.7, 0.75, 0.8]
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
        Generate entry signals when IBS <= extreme threshold.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        ibs_entry_threshold = params.get('ibs_entry_threshold', 0.1)

        # Entry: IBS <= threshold (extreme oversold)
        entry = data['ibs'] <= ibs_entry_threshold

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
        Exit when IBS >= threshold.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        ibs_exit_threshold = params.get('ibs_exit_threshold', 0.7)
        current_bar = data.iloc[current_idx]

        # Exit when IBS >= threshold (close near high)
        if current_bar['ibs'] >= ibs_exit_threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
