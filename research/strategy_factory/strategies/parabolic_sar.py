"""
Strategy #8: Parabolic SAR

Trend-following system using Parabolic Stop and Reverse:
- Enter when SAR flips from above to below price (bullish)
- Exit when SAR flips back (trend reversal)

Classic Wilder indicator for trailing stops.

Expected Performance (ES 2010-2024):
- Trade Count: 3,000-6,000
- Raw Sharpe: 0.2-0.5
- ML Sharpe: 0.9-1.6
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_parabolic_sar(data: pd.DataFrame, af_start: float = 0.02,
                           af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR indicator.

    Args:
        data: OHLCV dataframe
        af_start: Starting acceleration factor
        af_increment: AF increment per new extreme
        af_max: Maximum acceleration factor

    Returns:
        Series with SAR values
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values

    sar = np.zeros(len(data))
    ep = np.zeros(len(data))  # Extreme point
    af = np.zeros(len(data))
    trend = np.zeros(len(data))  # 1=long, -1=short

    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    trend[0] = 1

    for i in range(1, len(data)):
        # Previous values
        prev_sar = sar[i-1]
        prev_ep = ep[i-1]
        prev_af = af[i-1]
        prev_trend = trend[i-1]

        # Calculate new SAR
        sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

        # Check for trend reversal
        if prev_trend == 1:  # Was long
            # Ensure SAR doesn't go above price
            sar[i] = min(sar[i], low[i-1])
            if i > 1:
                sar[i] = min(sar[i], low[i-2])

            # Reversal to short if low breaks SAR
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = prev_ep  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                ep[i] = max(prev_ep, high[i])
                if ep[i] > prev_ep:
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    af[i] = prev_af
        else:  # Was short
            # Ensure SAR doesn't go below price
            sar[i] = max(sar[i], high[i-1])
            if i > 1:
                sar[i] = max(sar[i], high[i-2])

            # Reversal to long if high breaks SAR
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = prev_ep  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                ep[i] = min(prev_ep, low[i])
                if ep[i] < prev_ep:
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    af[i] = prev_af

    return pd.Series(sar, index=data.index)


class ParabolicSAR(BaseStrategy):
    """
    Parabolic SAR Strategy (Catalogue #8).

    Entry:
    - SAR flips from above to below price (bullish reversal)

    Exit:
    - SAR flips back (bearish reversal)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - psar_af_start: [0.02, 0.03] (initial acceleration factor)
    - psar_af_max: [0.2, 0.3] (maximum acceleration factor)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=8,
            name="ParabolicSAR",
            archetype="trend_following",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'psar_af_start': [0.02, 0.03],
            'psar_af_max': [0.2, 0.3]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parabolic SAR.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with SAR column
        """
        af_start = self.params.get('psar_af_start', 0.02)
        af_max = self.params.get('psar_af_max', 0.2)

        data['sar'] = calculate_parabolic_sar(
            data,
            af_start=af_start,
            af_increment=0.02,
            af_max=af_max
        )

        # SAR position relative to price
        data['sar_below'] = data['sar'] < data['Close']

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when SAR flips below price.

        Args:
            data: OHLCV dataframe with SAR
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: SAR flips from above to below price (bullish)
        sar_now_below = data['sar_below']
        sar_was_above = ~data['sar_below'].shift(1)

        entry = sar_now_below & sar_was_above

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
        Exit when SAR flips back above price.

        Args:
            data: OHLCV dataframe with SAR
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when SAR flips above price (trend reversal)
        if not current_bar['sar_below']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
