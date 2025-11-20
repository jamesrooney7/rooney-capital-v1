"""
Strategy #30: Momentum Fade

Contrarian strategy fading extreme momentum:
- Enter when momentum indicator hits extreme levels
- Anticipates reversal after overextension
- Exit when price returns to equilibrium

Counter-trend strategy betting on exhaustion.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma


class MomentumFade(BaseStrategy):
    """
    Momentum Fade Strategy (Catalogue #30).

    Entry:
    - N-period ROC (Rate of Change) exceeds threshold
    - Indicates overextended move (extreme momentum)
    - Enter counter-trend anticipating reversion

    Exit:
    - ROC returns to neutral zone (momentum exhausted)
    - OR price reaches mean (target achieved)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - mf_period: [5, 10, 20] (ROC calculation period)
    - mf_threshold: [5, 7, 10] (ROC % threshold for extreme)
    - mf_ma_length: [20, 50, 100] (mean for reversion target)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=30,
            name="MomentumFade",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'mf_period': [5, 10, 20],
            'mf_threshold': [5, 7, 10],
            'mf_ma_length': [20, 50, 100]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['mf_ma_length']) + max(self.param_grid['mf_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum (ROC) and mean.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with ROC and MA
        """
        period = self.params.get('mf_period', 10)
        ma_length = self.params.get('mf_ma_length', 50)

        # Rate of Change (momentum)
        data['roc'] = ((data['Close'] - data['Close'].shift(period)) /
                       data['Close'].shift(period)) * 100

        # Moving average for mean reversion target
        data['mean'] = calculate_sma(data['Close'], ma_length)

        # Distance from mean
        data['dist_from_mean'] = ((data['Close'] - data['mean']) / data['mean']) * 100

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals on extreme downward momentum.

        Args:
            data: OHLCV dataframe with ROC
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        threshold = params.get('mf_threshold', 7)

        # Entry conditions:
        # 1. Extreme negative ROC (oversold/overextended down)
        extreme_down = data['roc'] < -threshold

        # 2. Price below mean (confirming oversold)
        below_mean = data['Close'] < data['mean']

        # 3. ROC was not extreme on previous bar (fresh signal)
        was_not_extreme = data['roc'].shift(1) >= -threshold

        entry = extreme_down & below_mean & was_not_extreme

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
        Exit when momentum normalizes or price reaches mean.

        Args:
            data: OHLCV dataframe with ROC
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit 1: ROC returns to neutral (momentum exhausted)
        if current_bar['roc'] > -2:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: Price reaches or crosses above mean
        if current_bar['Close'] >= current_bar['mean']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
