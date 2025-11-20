"""
Strategy #31: Pivot Point Reversal

Classic floor trader pivot points for support/resistance:
- Calculate daily pivot, support (S1/S2), resistance (R1/R2)
- Enter when price bounces off support levels
- Exit when price reaches pivot or resistance

Traditional day trading approach using prior day's range.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-6,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class PivotPointReversal(BaseStrategy):
    """
    Pivot Point Reversal Strategy (Catalogue #31).

    Entry:
    - Price touches support level (S1 or S2)
    - Then bounces (closes above level)
    - Anticipates mean reversion to pivot

    Exit:
    - Price reaches pivot point (primary target)
    - OR price breaks below support level (failed bounce)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - pp_level: ['S1', 'S2'] (which support level to trade)
    - pp_tolerance: [0.001, 0.002, 0.003] (% tolerance for "at level")
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=31,
            name="PivotPointReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'pp_level': ['S1', 'S2'],
            'pp_tolerance': [0.001, 0.002, 0.003]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 50  # Need enough data to calculate daily pivots

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pivot points based on prior day's H/L/C.

        Args:
            data: OHLCV dataframe with datetime index

        Returns:
            DataFrame with pivot levels
        """
        # Need datetime column for daily grouping
        if 'datetime' not in data.columns:
            data['datetime'] = data.index

        # Extract date for daily pivots
        data['date'] = pd.to_datetime(data['datetime']).dt.date

        # Calculate daily high, low, close (prior day)
        daily_hlc = data.groupby('date').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).shift(1)  # Use prior day

        # Merge back to intraday data
        data = data.merge(
            daily_hlc,
            left_on='date',
            right_index=True,
            how='left',
            suffixes=('', '_daily')
        )

        # Calculate pivot points
        # Pivot = (High + Low + Close) / 3
        data['pivot'] = (data['High_daily'] + data['Low_daily'] + data['Close_daily']) / 3

        # Support levels
        # S1 = (2 * Pivot) - High
        data['s1'] = (2 * data['pivot']) - data['High_daily']

        # S2 = Pivot - (High - Low)
        data['s2'] = data['pivot'] - (data['High_daily'] - data['Low_daily'])

        # Resistance levels (for reference)
        # R1 = (2 * Pivot) - Low
        data['r1'] = (2 * data['pivot']) - data['Low_daily']

        # R2 = Pivot + (High - Low)
        data['r2'] = data['pivot'] + (data['High_daily'] - data['Low_daily'])

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price bounces off support.

        Args:
            data: OHLCV dataframe with pivot levels
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        level = params.get('pp_level', 'S1')
        tolerance = params.get('pp_tolerance', 0.002)

        # Select support level
        support_level = data['s1'] if level == 'S1' else data['s2']

        # Entry conditions:
        # 1. Price touched support (low at or below level)
        touched_support = data['Low'] <= support_level * (1 + tolerance)

        # 2. Price bounced (close above support)
        bounced = data['Close'] > support_level

        # 3. Valid pivot data (not NaN)
        valid_data = ~support_level.isna()

        # 4. Price below pivot (mean reversion opportunity)
        below_pivot = data['Close'] < data['pivot']

        entry = touched_support & bounced & valid_data & below_pivot

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
        Exit when price reaches pivot or support breaks.

        Args:
            data: OHLCV dataframe with pivots
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        level = params.get('pp_level', 'S1')
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Exit 1: Price reaches pivot (target achieved)
        if current_bar['Close'] >= current_bar['pivot']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Support level breaks (failed bounce)
        support_level = entry_bar['s1'] if level == 'S1' else entry_bar['s2']
        if current_bar['Close'] < support_level:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
