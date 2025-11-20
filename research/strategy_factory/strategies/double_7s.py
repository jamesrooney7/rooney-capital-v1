"""
Strategy #37: Double 7s

Larry Connors' percentile rank strategy:
- Enter when 7-day price percentile rank < 5%
- Exit when 7-day price percentile rank > 95%

Pure mean reversion on price extremes.

Expected Performance (ES 2010-2024):
- Trade Count: 10,000+
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-2.0
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


class Double7s(BaseStrategy):
    """
    Double 7s Strategy (Catalogue #37).

    Entry:
    - Price percentile rank over N days < entry_pct%

    Exit:
    - Price percentile rank over N days > exit_pct%
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - percentile_window: [5, 7, 10]
    - entry_pct: [3, 5, 10]
    - exit_pct: [90, 95, 97]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=37,
            name="Double7s",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'percentile_window': [5, 7, 10],
            'entry_pct': [3, 5, 10],
            'exit_pct': [90, 95, 97]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        max_window = max(self.param_grid['percentile_window'])
        return max_window + 5  # +5 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling percentile rank of close price.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with 'pct_rank' column added
        """
        window = self.params.get('percentile_window', 7)

        # Calculate percentile rank over rolling window
        # Percentile rank = % of values in window that are less than current value
        def percentile_rank(series):
            return (series.rank(pct=True).iloc[-1]) * 100

        data['pct_rank'] = data['Close'].rolling(window=window).apply(
            percentile_rank, raw=False
        )

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when percentile rank is low (extreme low).

        Args:
            data: OHLCV dataframe with 'pct_rank'
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        entry_pct = params.get('entry_pct', 5)

        # Entry: Percentile rank < entry threshold (e.g., < 5%)
        entry = data['pct_rank'] < entry_pct

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
        Exit when percentile rank is high (extreme high).

        Args:
            data: OHLCV dataframe with 'pct_rank'
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        exit_pct = params.get('exit_pct', 95)

        current_bar = data.iloc[current_idx]
        current_pct_rank = current_bar['pct_rank']

        # Exit when percentile rank > exit threshold (e.g., > 95%)
        if current_pct_rank > exit_pct:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
