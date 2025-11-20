"""
Strategy #39: Time-of-Day Reversal

Intraday mean reversion at specific times:
- Enter when price is oversold at key reversal time
- Exploits intraday session patterns (lunch lull, late day rally)
- Exit by end of day or when target reached

Captures time-based market microstructure effects.

Expected Performance (ES 2010-2024):
- Trade Count: 2,000-4,000
- Raw Sharpe: 0.3-0.6
- ML Sharpe: 1.0-1.8
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_rsi


class TimeOfDayReversal(BaseStrategy):
    """
    Time-of-Day Reversal Strategy (Catalogue #39).

    Entry:
    - Specific time window (e.g., 10:00-11:00, 13:00-14:00)
    - Price is oversold (RSI < threshold)
    - Anticipates mean reversion rally

    Exit:
    - Target time reached (e.g., 15:30)
    - OR RSI reaches overbought
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - tod_entry_hour: [10, 11, 13] (hour to enter)
    - tod_rsi_threshold: [25, 30, 35] (RSI oversold level)
    - tod_exit_hour: [14, 15] (hour to exit)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=39,
            name="TimeOfDayReversal",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'tod_entry_hour': [10, 11, 13],
            'tod_rsi_threshold': [25, 30, 35],
            'tod_exit_hour': [14, 15]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return 50

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and time-based markers.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with RSI and time info
        """
        # Need datetime for time filtering
        if 'datetime' not in data.columns:
            data['datetime'] = data.index

        # Extract hour
        data['hour'] = pd.to_datetime(data['datetime']).dt.hour

        # Calculate RSI for oversold detection
        data['rsi'] = calculate_rsi(data['Close'], period=14)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals at specific time when oversold.

        Args:
            data: OHLCV dataframe with time and RSI
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        entry_hour = params.get('tod_entry_hour', 11)
        rsi_threshold = params.get('tod_rsi_threshold', 30)

        # Entry conditions:
        # 1. Specific hour window
        in_time_window = data['hour'] == entry_hour

        # 2. RSI oversold
        oversold = data['rsi'] < rsi_threshold

        # 3. RSI was not oversold on previous bar (fresh signal)
        was_not_oversold = data['rsi'].shift(1) >= rsi_threshold

        entry = in_time_window & oversold & was_not_oversold

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
        Exit at target time or when RSI reaches overbought.

        Args:
            data: OHLCV dataframe with time and RSI
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        exit_hour = params.get('tod_exit_hour', 15)
        current_bar = data.iloc[current_idx]

        # Exit 1: Target time reached
        if current_bar['hour'] >= exit_hour:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # Exit 2: RSI reaches overbought (70)
        if current_bar['rsi'] >= 70:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
