"""
Strategy #44: Bollinger Bands Reversal + IBS Strategy

Mean reversion strategy combining volatility and intrabar strength:
- Enter when close < lower BB (2σ) AND IBS < threshold (oversold)
- Exit when IBS > threshold (overbought)
- Double confirmation: volatility (BB) + intrabar strength (IBS)

Botnet101 strategy - best in mean-reverting markets.

Expected Performance:
- Exits on IBS overbought
- Timeframe agnostic
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


class BBIBSReversal(BaseStrategy):
    """
    Bollinger Bands + IBS Reversal Strategy (Catalogue #44).

    Entry:
    - Close < lower Bollinger Band (oversold by volatility)
    - IBS < threshold (close near bar's low)

    Exit:
    - IBS > threshold (close near bar's high, overbought)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - bb_period: [15, 20, 25]
    - bb_mult: [1.5, 2.0, 2.5]
    - ibs_buy_threshold: [0.1, 0.2, 0.3]
    - ibs_sell_threshold: [0.7, 0.8, 0.9]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=44,
            name="BBIBSReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'bb_period': [15, 20, 25],
            'bb_mult': [1.5, 2.0, 2.5],
            'ibs_buy_threshold': [0.1, 0.2, 0.3],
            'ibs_sell_threshold': [0.7, 0.8, 0.9]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['bb_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and IBS.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with bb_middle, bb_lower, bb_upper, ibs columns
        """
        bb_period = self.params.get('bb_period', 20)
        bb_mult = self.params.get('bb_mult', 2.0)

        # Bollinger Bands
        data['bb_middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * bb_mult)
        data['bb_lower'] = data['bb_middle'] - (bb_std * bb_mult)

        # Internal Bar Strength
        data['ibs'] = calculate_ibs(data)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price breaks below lower BB with low IBS.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        ibs_buy_threshold = params.get('ibs_buy_threshold', 0.2)

        # Condition 1: Close < lower Bollinger Band
        below_bb = data['Close'] < data['bb_lower']

        # Condition 2: IBS < threshold (close near low of bar)
        low_ibs = data['ibs'] < ibs_buy_threshold

        # Entry: Both conditions met
        entry = below_bb & low_ibs

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
        Exit when IBS > threshold (overbought).

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

        # Exit when IBS > threshold (close near high of bar)
        if current_bar['ibs'] > ibs_sell_threshold:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
