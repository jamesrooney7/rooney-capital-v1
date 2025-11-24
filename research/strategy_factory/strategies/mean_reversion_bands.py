"""
Strategy #27: Mean Reversion Bands

Simple mean reversion using standard deviation bands:
- Enter when price crosses below lower band (oversold)
- Exit when price returns to middle (mean) or upper band

Similar to Bollinger Bands but focuses on band touches and mean reversion.

Expected Performance (ES 2010-2024):
- Trade Count: 4,000-7,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma


class MeanReversionBands(BaseStrategy):
    """
    Mean Reversion Bands Strategy (Catalogue #27).

    Entry:
    - Price closes below lower band (mean - N*std)
    - Indicates oversold condition

    Exit:
    - Price returns to mean (middle band)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - mrb_length: [20, 30, 50] (lookback period for mean/std)
    - mrb_std: [1.5, 2.0, 2.5] (standard deviations for bands)
    - mrb_exit_pct: [0.5, 0.75, 1.0] (% return to mean for exit: 0.5=halfway, 1.0=full mean)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=27,
            name="MeanReversionBands",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'mrb_length': [20, 30, 50],
            'mrb_std': [1.5, 2.0, 2.5],
            'mrb_exit_pct': [0.5, 0.75, 1.0]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['mrb_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion bands.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with middle, upper, and lower bands
        """
        length = self.params.get('mrb_length', 30)
        std_mult = self.params.get('mrb_std', 2.0)

        # Calculate mean and standard deviation
        data['middle'] = calculate_sma(data['Close'], length)
        data['std'] = data['Close'].rolling(window=length).std()

        # Calculate bands
        data['upper_band'] = data['middle'] + std_mult * data['std']
        data['lower_band'] = data['middle'] - std_mult * data['std']

        # Distance from mean (for exit logic)
        data['dist_from_mean'] = data['Close'] - data['middle']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price crosses below lower band.

        Args:
            data: OHLCV dataframe with bands
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry conditions:
        # 1. Current close below lower band
        below_band = data['Close'] < data['lower_band']

        # 2. Previous close was above lower band (fresh cross)
        was_above = data['Close'].shift(1) >= data['lower_band'].shift(1)

        entry = below_band & was_above

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
        Exit when price returns toward mean.

        Args:
            data: OHLCV dataframe with bands
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        exit_pct = params.get('mrb_exit_pct', 0.75)
        current_bar = data.iloc[current_idx]
        entry_bar = data.iloc[entry_idx]

        # Calculate target price based on % return to mean
        # Entry was below mean, target is X% of the way back to mean
        entry_dist = entry_bar['middle'] - entry_price
        target_price = entry_price + (entry_dist * exit_pct)

        # Exit when price reaches target (returned toward mean)
        if current_bar['Close'] >= target_price:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
