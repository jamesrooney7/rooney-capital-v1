"""
Strategy #6: Hammer/Shooting Star

Classic candlestick reversal patterns:
- Hammer: Small body at top, long lower shadow (bullish reversal)
- Shooting star: Small body at bottom, long upper shadow (bearish, skip for long-only)

Entry after hammer pattern at support or in downtrend.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class HammerShootingStar(BaseStrategy):
    """
    Hammer/Shooting Star Strategy (Catalogue #6).

    Entry:
    - Hammer pattern: Long lower shadow (2-3x body), small upper shadow
    - Body at upper part of range
    - Optionally: require prior downtrend

    Exit:
    - Pattern failure (close below hammer low)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - hammer_shadow_ratio: [2.0, 2.5, 3.0] (lower shadow / body ratio)
    - hammer_body_position: [0.6, 0.7, 0.8] (body position in range, higher=top)
    - hammer_require_trend: [0, 2, 3] (consecutive down bars required)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=6,
            name="HammerShootingStar",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'hammer_shadow_ratio': [2.0, 2.5, 3.0],
            'hammer_body_position': [0.6, 0.7, 0.8],
            'hammer_require_trend': [0, 2, 3]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate hammer pattern metrics.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with hammer indicators
        """
        # Body and shadow calculations
        data['body'] = (data['Close'] - data['Open']).abs()
        data['range'] = data['High'] - data['Low']

        # Lower shadow = distance from low to min(open, close)
        data['lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']

        # Upper shadow = distance from max(open, close) to high
        data['upper_shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])

        # Body position in range (0=bottom, 1=top)
        data['body_position'] = np.where(
            data['range'] > 0,
            (np.minimum(data['Open'], data['Close']) - data['Low']) / data['range'],
            0.5
        )

        # Shadow to body ratio
        data['shadow_ratio'] = np.where(
            data['body'] > 0,
            data['lower_shadow'] / data['body'],
            0
        )

        # Down bar streak for trend confirmation
        down_bar = data['Close'] < data['Close'].shift(1)
        down_streak = down_bar.astype(int).groupby((~down_bar).cumsum()).cumsum()
        data['down_streak'] = down_streak.shift(1)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on hammer pattern.

        Args:
            data: OHLCV dataframe with hammer metrics
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        shadow_ratio = params.get('hammer_shadow_ratio', 2.5)
        body_position = params.get('hammer_body_position', 0.7)
        require_trend = params.get('hammer_require_trend', 0)

        # Hammer conditions:
        # 1. Long lower shadow (at least 2-3x body size)
        long_shadow = data['shadow_ratio'] >= shadow_ratio

        # 2. Body near top of range
        body_at_top = data['body_position'] >= body_position

        # 3. Small upper shadow (less than body)
        small_upper = data['upper_shadow'] <= data['body']

        # 4. Minimum range to avoid noise
        min_range = data['range'] > data['range'].rolling(20).mean() * 0.5

        # 5. Prior downtrend if required
        if require_trend > 0:
            had_downtrend = data['down_streak'] >= require_trend
        else:
            had_downtrend = True

        entry = long_shadow & body_at_top & small_upper & min_range & had_downtrend

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
        Exit when pattern fails (close below hammer low).

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

        # Exit if close below hammer's low (pattern failure)
        if current_bar['Close'] < entry_bar['Low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
