"""
Strategy #7: Moving Average Envelope

Mean reversion using MA envelopes:
- Enter when price touches lower envelope (MA - X%)
- Exit when price returns to MA centerline

Classic mean reversion indicator.

Expected Performance (ES 2010-2024):
- Trade Count: 6,000-10,000
- Raw Sharpe: 0.4-0.7
- ML Sharpe: 1.3-2.2
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_sma, calculate_ema


class MovingAverageEnvelope(BaseStrategy):
    """
    Moving Average Envelope Strategy (Catalogue #7).

    Entry:
    - Price closes below lower envelope
    - Lower envelope = MA × (1 - envelope_pct)

    Exit:
    - Price returns to MA centerline
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - mae_length: [20, 50, 100, 200]
    - mae_type: ['sma', 'ema']
    - mae_envelope_pct: [1.0, 2.0, 3.0, 4.0]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=7,
            name="MovingAverageEnvelope",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'mae_length': [20, 50, 100, 200],
            'mae_envelope_pct': [1.0, 2.0, 3.0, 4.0]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['mae_length']) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MA envelopes.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with mae_upper, mae_middle, mae_lower columns
        """
        length = self.params.get('mae_length', 50)
        envelope_pct = self.params.get('mae_envelope_pct', 2.0) / 100
        ma_type = self.params.get('mae_type', 'sma')

        # Calculate middle band (MA)
        if ma_type == 'ema':
            data['mae_middle'] = calculate_ema(data['Close'], length)
        else:
            data['mae_middle'] = calculate_sma(data['Close'], length)

        # Calculate envelopes
        data['mae_upper'] = data['mae_middle'] * (1 + envelope_pct)
        data['mae_lower'] = data['mae_middle'] * (1 - envelope_pct)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when price touches lower envelope.

        Args:
            data: OHLCV dataframe with MA envelopes
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        # Entry: Close below lower envelope
        entry = data['Close'] < data['mae_lower']

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
        Exit when price returns to MA centerline.

        Args:
            data: OHLCV dataframe with MA envelopes
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]

        # Exit when price returns to middle band
        if current_bar['Close'] >= current_bar['mae_middle']:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
