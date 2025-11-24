"""
Strategy #49: Average High-Low Range + IBS Reversal Strategy

Volatility + IBS mean reversion:
- Enter when close below (high - 2.5×avg_HL_range) for N bars AND IBS < threshold
- Exit when close > high[1]
- Avg_HL = SMA(high-low, 20) - average high-low range

Botnet101 strategy - identifies extreme deviations.

Expected Performance:
- Combines volatility measure with IBS
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

    Args:
        data: OHLCV dataframe

    Returns:
        IBS series
    """
    range_hl = data['High'] - data['Low']
    range_hl = range_hl.replace(0, np.nan)
    ibs = (data['Close'] - data['Low']) / range_hl
    return ibs.fillna(0.5)


class AvgHLRangeIBS(BaseStrategy):
    """
    Average High-Low Range + IBS Reversal Strategy (Catalogue #49).

    Entry:
    - Close below (high - multiplier × avg_HL_range) for N bars
    - IBS < threshold
    - avg_HL_range = SMA(high-low, length)

    Exit:
    - Close > high[1] (reversal confirmation)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - length: [15, 20, 25]
    - range_multiplier: [2.0, 2.5, 3.0]
    - bars_below_threshold: [1, 2, 3]
    - ibs_buy_threshold: [0.1, 0.2, 0.3]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=49,
            name="AvgHLRangeIBS",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'length': [15, 20, 25],
            'range_multiplier': [2.0, 2.5, 3.0],
            'bars_below_threshold': [1, 2, 3],
            'ibs_buy_threshold': [0.1, 0.2, 0.3],
            'use_discretionary_exits': [True, False]  # Test with/without discretionary exit
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average high-low range, threshold, and IBS.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with avg_hl_range, lower_threshold, ibs, consecutive_below columns
        """
        length = self.params.get('length', 20)
        range_multiplier = self.params.get('range_multiplier', 2.5)

        # Average high-low range
        data['hl_range'] = data['High'] - data['Low']
        data['avg_hl_range'] = data['hl_range'].rolling(window=length).mean()

        # Lower threshold: high - multiplier × avg_HL_range
        data['lower_threshold'] = data['High'] - (range_multiplier * data['avg_hl_range'])

        # IBS
        data['ibs'] = calculate_ibs(data)

        # Close below threshold
        data['below_threshold'] = data['Close'] < data['lower_threshold']

        # Count consecutive bars below threshold
        consecutive = []
        count = 0
        for val in data['below_threshold']:
            if val:
                count += 1
            else:
                count = 0
            consecutive.append(count)

        data['consecutive_below'] = consecutive

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when below threshold and low IBS.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        bars_below_threshold = params.get('bars_below_threshold', 2)
        ibs_buy_threshold = params.get('ibs_buy_threshold', 0.2)

        # Condition 1: Consecutive bars below threshold >= threshold
        below_condition = data['consecutive_below'] >= bars_below_threshold

        # Condition 2: IBS < threshold
        ibs_condition = data['ibs'] < ibs_buy_threshold

        # Entry: Both conditions met
        entry = below_condition & ibs_condition

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
        use_discretionary_exits = params.get('use_discretionary_exits', True)
        current_bar = data.iloc[current_idx]

        # Only apply discretionary exit if enabled
        if use_discretionary_exits:
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

        # No strategy-specific exit (rely on stop/target/time)
        return TradeExit(exit=False, exit_type='none')
