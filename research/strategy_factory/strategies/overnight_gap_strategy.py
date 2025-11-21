"""
Strategy #38: Overnight Gap Strategy

Trade overnight gaps between sessions:
- Identify gaps between prior close and current open
- Enter when gap is significant (exceeds threshold)
- Can trade gap fill (reversion) or gap continuation (momentum)

Exploits overnight information asymmetry.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class OvernightGapStrategy(BaseStrategy):
    """
    Overnight Gap Strategy (Catalogue #38).

    Entry:
    - Gap down exceeds threshold (open < prior close)
    - Enter long anticipating gap fill (mean reversion)
    - For long-only, only trade gap downs

    Exit:
    - Gap filled (price returns to prior close)
    - OR end of day if gap doesn't fill
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - gap_threshold: [0.003, 0.005, 0.010] (min gap size as % of price)
    - gap_max: [0.015, 0.020, 0.030] (max gap size to avoid crashes)
    - gap_entry_time: [0, 1, 2] (bars after open to wait)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=38,
            name="OvernightGapStrategy",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'gap_threshold': [0.003, 0.005, 0.010],
            'gap_max': [0.015, 0.020, 0.030],
            'gap_entry_time': [0, 1, 2]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 50

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap size and session markers.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with gap metrics
        """
        # Need datetime for session detection
        if 'datetime' in data.columns:
            dt_col = data['datetime']
        else:
            dt_col = data.index

        # Detect session start (first bar of day)
        # Assuming intraday data, detect when date changes
        data['date'] = pd.to_datetime(dt_col).dt.date
        data['is_open'] = data['date'] != data['date'].shift(1)

        # Prior session close (last close of previous day)
        data['prior_close'] = np.where(
            data['is_open'],
            data['Close'].shift(1),
            np.nan
        )
        data['prior_close'] = data['prior_close'].fillna(method='ffill')

        # Gap size (as % of prior close)
        data['gap_pct'] = np.where(
            data['is_open'],
            (data['Open'] - data['prior_close']) / data['prior_close'],
            0
        )

        # Bars since open
        data['bars_since_open'] = data.groupby('date').cumcount()

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on gap down.

        Args:
            data: OHLCV dataframe with gap metrics
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('gap_threshold', 0.005)
        max_gap = params.get('gap_max', 0.020)
        entry_time = params.get('gap_entry_time', 1)

        # Entry conditions:
        # 1. Gap down (negative gap)
        gap_down = data['gap_pct'] < -threshold

        # 2. Gap not too large (avoid crashes)
        gap_reasonable = data['gap_pct'] > -max_gap

        # 3. Wait N bars after open for confirmation
        wait_period = data['bars_since_open'] >= entry_time

        entry = gap_down & gap_reasonable & wait_period

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
        Exit when gap fills (price reaches prior close).

        Args:
            data: OHLCV dataframe with gap info
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        entry_bar = data.iloc[entry_idx]
        current_bar = data.iloc[current_idx]

        # Exit when gap fills (price reaches/exceeds prior close)
        prior_close = entry_bar['prior_close']
        if current_bar['Close'] >= prior_close:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
