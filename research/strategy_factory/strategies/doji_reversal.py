"""
Strategy #14: Doji Reversal

Doji candlestick pattern signaling indecision and potential reversal:
- Enter after doji at support/bottom of downtrend
- Very small body (open ≈ close), shadows indicate rejection

Classic reversal pattern when momentum exhausted.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class DojiReversal(BaseStrategy):
    """
    Doji Reversal Strategy (Catalogue #14).

    Entry:
    - Doji pattern: Body < X% of total range
    - After downtrend (N consecutive down bars)
    - Optionally: require longer shadows for "dragonfly doji"

    Exit:
    - Pattern failure (close below doji low)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - doji_body_pct: [0.05, 0.10, 0.15] (max body size as % of range)
    - doji_require_trend: [0, 2, 3] (consecutive down bars required)
    - doji_shadow_min: [0.3, 0.5, 0.7] (min shadow size as % of range)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=14,
            name="DojiReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'doji_body_pct': [0.05, 0.10, 0.15],
            'doji_require_trend': [0, 2, 3],
            'doji_shadow_min': [0.3, 0.5, 0.7]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate doji pattern metrics.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with doji indicators
        """
        # Body and range calculations
        data['body'] = (data['Close'] - data['Open']).abs()
        data['range'] = data['High'] - data['Low']

        # Avoid division by zero
        data['body_pct'] = np.where(
            data['range'] > 0,
            data['body'] / data['range'],
            1.0
        )

        # Total shadow length
        data['total_shadow'] = data['range'] - data['body']
        data['shadow_pct'] = np.where(
            data['range'] > 0,
            data['total_shadow'] / data['range'],
            0
        )

        # Lower shadow (for dragonfly doji)
        data['lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        data['lower_shadow_pct'] = np.where(
            data['range'] > 0,
            data['lower_shadow'] / data['range'],
            0
        )

        # Down bar streak for trend confirmation
        down_bar = data['Close'] < data['Close'].shift(1)
        down_streak = down_bar.astype(int).groupby((~down_bar).cumsum()).cumsum()
        data['down_streak'] = down_streak.shift(1)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on doji after downtrend.

        Args:
            data: OHLCV dataframe with doji metrics
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        body_pct = params.get('doji_body_pct', 0.10)
        require_trend = params.get('doji_require_trend', 2)
        shadow_min = params.get('doji_shadow_min', 0.5)

        # Doji conditions:
        # 1. Very small body
        small_body = data['body_pct'] <= body_pct

        # 2. Significant shadows (rejection)
        long_shadows = data['shadow_pct'] >= shadow_min

        # 3. Minimum range to avoid noise
        min_range = data['range'] > data['range'].rolling(20).mean() * 0.3

        # 4. Prior downtrend if required
        if require_trend > 0:
            had_downtrend = data['down_streak'] >= require_trend
        else:
            had_downtrend = True

        entry = small_body & long_shadows & min_range & had_downtrend

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
        Exit when pattern fails (close below doji low).

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
        entry_bar = data.iloc[entry_idx]

        # Exit if close below doji's low (pattern failure)
        if current_bar['Close'] < entry_bar['Low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
