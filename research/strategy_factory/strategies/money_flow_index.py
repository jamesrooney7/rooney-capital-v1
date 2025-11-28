"""
Strategy #34: Money Flow Index (MFI)

Volume-weighted RSI for overbought/oversold conditions:
- MFI combines price and volume (money flow)
- Enter when MFI crosses above oversold threshold
- Exit when MFI reaches overbought or reverses

RSI variant incorporating volume for institutional flow.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-5,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index.

    Args:
        data: OHLCV dataframe
        period: MFI period

    Returns:
        Series with MFI values
    """
    # Typical price
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3

    # Raw money flow
    money_flow = typical_price * data['Volume']

    # Positive and negative money flow
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

    # Sum over period
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()

    # Money flow ratio
    mf_ratio = positive_mf / negative_mf

    # Money flow index
    mfi = 100 - (100 / (1 + mf_ratio))

    return mfi


class MoneyFlowIndex(BaseStrategy):
    """
    Money Flow Index Strategy (Catalogue #34).

    Entry:
    - MFI crosses above oversold threshold (e.g., 20)
    - Volume confirms buying pressure
    - Anticipates bounce from oversold

    Exit:
    - MFI reaches overbought (e.g., 80)
    - OR MFI reverses below midpoint (50)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - mfi_period: [14, 21, 28] (MFI calculation period)
    - mfi_oversold: [15, 20, 25] (oversold threshold)
    - mfi_overbought: [75, 80, 85] (overbought threshold)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=34,
            name="MoneyFlowIndex",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'mfi_period': [14, 21, 28],
            'mfi_oversold': [15, 20, 25],
            'mfi_overbought': [75, 80, 85]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['mfi_period']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Money Flow Index.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with MFI
        """
        period = self.params.get('mfi_period', 14)

        data['mfi'] = calculate_mfi(data, period)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when MFI crosses above oversold.

        Args:
            data: OHLCV dataframe with MFI
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        oversold = params.get('mfi_oversold', 20)

        # Entry conditions:
        # 1. MFI crosses above oversold threshold
        mfi_above = data['mfi'] > oversold
        mfi_was_below = data['mfi'].shift(1) <= oversold

        entry = mfi_above & mfi_was_below

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
        Exit when MFI reaches overbought or reverses.

        Args:
            data: OHLCV dataframe with MFI
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        overbought = params.get('mfi_overbought', 80)
        current_bar = data.iloc[current_idx]

        # Exit 1: MFI reaches overbought (take profit)
        if current_bar['mfi'] >= overbought:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: MFI falls below 50 (momentum lost)
        if current_bar['mfi'] < 50:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
