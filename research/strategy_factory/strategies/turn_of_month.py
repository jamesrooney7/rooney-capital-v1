"""
Strategy #43: Turn of the Month Strategy on Steroids

Seasonal/calendar anomaly strategy:
- Enter when day of month >= threshold (default 25) AND close < close[1]
- Exit when RSI > overbought threshold (default 65)
- Exploits turn-of-month effect (stocks tend to rise end of month)

Botnet101 strategy - requires daily bars for calendar tracking.

Expected Performance:
- Mean reversion entry with RSI exit
- Typical hold period: Short-term
- Best on daily timeframe
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI indicator.

    Args:
        series: Price series (usually Close)
        period: RSI period

    Returns:
        RSI series
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


class TurnOfMonth(BaseStrategy):
    """
    Turn of the Month Strategy (Catalogue #43).

    Entry:
    - Day of month >= threshold (default 25)
    - Close < close[1] (pullback/dip)

    Exit:
    - RSI > overbought threshold (default 65)
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - day_threshold: [20, 25, 28]
    - rsi_period: [10, 14, 20]
    - rsi_overbought: [60, 65, 70, 75]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=43,
            name="TurnOfMonth",
            archetype="seasonal",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'day_threshold': [20, 25, 28],
            'rsi_period': [10, 14, 20],
            'rsi_overbought': [60, 65, 70, 75]
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        return max(self.param_grid['rsi_period']) * 3 + 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and extract day of month.

        Args:
            data: OHLCV dataframe with datetime index

        Returns:
            DataFrame with rsi and day_of_month columns
        """
        rsi_period = self.params.get('rsi_period', 14)

        # Calculate RSI
        data['rsi'] = calculate_rsi(data['Close'], period=rsi_period)

        # Extract day of month from datetime index
        # Handle both DatetimeIndex and regular index
        if isinstance(data.index, pd.DatetimeIndex):
            data['day_of_month'] = data.index.day
        else:
            # If index is not DatetimeIndex, try to convert or use datetime column
            if 'Datetime' in data.columns:
                data['day_of_month'] = pd.to_datetime(data['Datetime']).dt.day
            elif 'Date' in data.columns:
                data['day_of_month'] = pd.to_datetime(data['Date']).dt.day
            else:
                # Fallback: try to convert index to datetime
                try:
                    data['day_of_month'] = pd.to_datetime(data.index).day
                except:
                    # If all else fails, set to 15 (mid-month) to avoid errors
                    data['day_of_month'] = 15

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals during turn-of-month period with pullback.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        day_threshold = params.get('day_threshold', 25)

        # Condition 1: Day of month >= threshold
        turn_of_month = data['day_of_month'] >= day_threshold

        # Condition 2: Close < close[1] (pullback)
        pullback = data['Close'] < data['Close'].shift(1)

        # Entry: Both conditions met
        entry = turn_of_month & pullback

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
        Exit when RSI > overbought threshold.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        rsi_overbought = params.get('rsi_overbought', 65)
        current_bar = data.iloc[current_idx]

        # Exit when RSI > overbought threshold
        if current_bar['rsi'] > rsi_overbought:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')
