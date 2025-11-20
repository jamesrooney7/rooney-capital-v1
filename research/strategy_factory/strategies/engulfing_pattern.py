"""
Strategy #5: Engulfing Pattern

Classic candlestick pattern for reversals:
- Enter after bullish engulfing pattern (down trend reversal)
- Exit when pattern fails or target hit

Engulfing = current bar's body completely engulfs previous bar's body.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.7
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class EngulfingPattern(BaseStrategy):
    """
    Engulfing Pattern Strategy (Catalogue #5).

    Entry:
    - Bullish engulfing: Current up bar engulfs previous down bar
    - Optionally: require prior downtrend (N bars down)

    Exit:
    - Pattern failure (close below engulfing bar low)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - eng_min_body_pct: [0.3, 0.5, 0.7] (min body size as % of range)
    - eng_require_trend: [0, 2, 3] (consecutive bars to confirm trend, 0=none)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=5,
            name="EngulfingPattern",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'eng_min_body_pct': [0.3, 0.5, 0.7],
            'eng_require_trend': [0, 2, 3]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate body sizes and engulfing patterns.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with body metrics and engulfing flags
        """
        # Body size and direction
        data['body'] = data['Close'] - data['Open']
        data['body_abs'] = data['body'].abs()
        data['range'] = data['High'] - data['Low']

        # Avoid division by zero
        data['body_pct'] = np.where(
            data['range'] > 0,
            data['body_abs'] / data['range'],
            0
        )

        # Up/down bars
        data['up_bar'] = data['Close'] > data['Open']
        data['down_bar'] = data['Close'] < data['Open']

        # Bullish engulfing: current up bar body engulfs previous down bar body
        curr_open = data['Open']
        curr_close = data['Close']
        prev_open = data['Open'].shift(1)
        prev_close = data['Close'].shift(1)

        bullish_engulfing = (
            data['up_bar'] &  # Current bar is up
            data['down_bar'].shift(1) &  # Previous bar was down
            (curr_open <= prev_close) &  # Current open below/equal prev close
            (curr_close >= prev_open)    # Current close above/equal prev open
        )

        data['bullish_engulfing'] = bullish_engulfing

        # Count consecutive down bars for trend confirmation
        down_streak = data['down_bar'].astype(int).groupby(
            (~data['down_bar']).cumsum()
        ).cumsum()
        data['down_streak'] = down_streak.shift(1)  # Prior trend

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on bullish engulfing pattern.

        Args:
            data: OHLCV dataframe with engulfing info
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        min_body_pct = params.get('eng_min_body_pct', 0.5)
        require_trend = params.get('eng_require_trend', 0)

        # Entry conditions:
        # 1. Bullish engulfing pattern
        engulfing = data['bullish_engulfing']

        # 2. Current bar has significant body
        strong_body = data['body_pct'] >= min_body_pct

        # 3. Prior downtrend if required
        if require_trend > 0:
            had_downtrend = data['down_streak'] >= require_trend
        else:
            had_downtrend = True

        entry = engulfing & strong_body & had_downtrend

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
        Exit when pattern fails (close below engulfing bar low).

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

        # Exit if close below engulfing bar's low (pattern failure)
        if current_bar['Close'] < entry_bar['Low']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
