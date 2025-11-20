"""
Base Strategy class for Strategy Factory.

All research strategies inherit from this class and implement:
- entry_logic(): When to enter trades
- exit_logic(): When to exit trades
- calculate_indicators(): Compute required technical indicators
- param_grid: Dictionary of parameters to optimize
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TradeExit:
    """
    Represents an exit condition and its type.
    """
    exit: bool
    exit_type: str  # 'signal', 'stop_loss', 'take_profit', 'time', 'eod'
    exit_price: Optional[float] = None


class BaseStrategy(ABC):
    """
    Base class for all research strategies.

    Each strategy must implement:
    - entry_logic(): Returns boolean Series indicating entry signals
    - exit_logic(): Returns TradeExit indicating when/how to exit
    - calculate_indicators(): Computes required indicators
    - param_grid: Property returning parameter dictionary

    Exit Hierarchy (checked in order):
    1. Strategy-specific exit (from exit_logic)
    2. Stop loss (fixed ATR multiple)
    3. Take profit (fixed ATR multiple)
    4. Time-based exit (max bars held)
    5. End-of-day exit (4pm EST)
    """

    def __init__(
        self,
        strategy_id: int,
        name: str,
        archetype: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy.

        Args:
            strategy_id: Unique ID from strategy catalogue
            name: Strategy name
            archetype: Strategy archetype (mean_reversion, momentum, etc.)
            params: Strategy parameters (from param_grid)
        """
        self.strategy_id = strategy_id
        self.name = name
        self.archetype = archetype
        self.params = params or {}

        # Fixed exit parameters (Phase 1)
        self.stop_loss_atr = self.params.get('stop_loss_atr', 1.0)
        self.take_profit_atr = self.params.get('take_profit_atr', 1.0)
        self.max_bars_held = self.params.get('max_bars_held', 20)
        self.auto_close_time = self.params.get('auto_close_time', '16:00')  # 4pm EST

        # Data
        self.data: Optional[pd.DataFrame] = None
        self.indicators: Optional[pd.DataFrame] = None

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required by strategy.

        This is called once during initialization.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with indicator columns added
        """
        pass

    @abstractmethod
    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        pass

    def exit_logic(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        current_idx: int
    ) -> TradeExit:
        """
        Check if exit condition is met.

        Default implementation returns no exit (strategy-specific exit).
        Subclasses can override to add custom exit logic.

        Args:
            data: OHLCV dataframe with indicators
            params: Strategy parameters
            entry_idx: Index of entry bar
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object with exit flag and type
        """
        # Default: no strategy-specific exit
        return TradeExit(exit=False, exit_type='none')

    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, List[Any]]:
        """
        Parameter grid for optimization.

        Returns:
            Dictionary mapping param names to lists of values.
            Example: {
                'rsi_period': [10, 14, 20, 30],
                'threshold': [20, 25, 30]
            }
        """
        pass

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """
        Minimum number of bars needed before strategy can generate signals.

        Returns:
            Integer bar count
        """
        pass

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting by calculating all indicators.

        Args:
            data: Raw OHLCV dataframe

        Returns:
            DataFrame with indicators added
        """
        # Calculate ATR (used for stops/targets)
        data = self._calculate_atr(data)

        # Calculate strategy-specific indicators
        data = self.calculate_indicators(data)

        self.data = data
        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        Used for stop loss and take profit calculations.

        Args:
            data: OHLCV dataframe
            period: ATR period (default 14)

        Returns:
            DataFrame with 'atr' column added
        """
        high = data['High']
        low = data['Low']
        close = data['Close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA of True Range
        data['atr'] = tr.ewm(span=period, adjust=False).mean()

        return data

    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        direction: int = 1  # 1 for long, -1 for short
    ) -> bool:
        """
        Check if stop loss is hit.

        Args:
            entry_price: Entry price
            current_price: Current price
            atr: Current ATR value
            direction: Trade direction (1=long, -1=short)

        Returns:
            True if stop loss hit
        """
        stop_distance = self.stop_loss_atr * atr

        if direction == 1:  # Long position
            stop_price = entry_price - stop_distance
            return current_price <= stop_price
        else:  # Short position (future implementation)
            stop_price = entry_price + stop_distance
            return current_price >= stop_price

    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        direction: int = 1
    ) -> bool:
        """
        Check if take profit is hit.

        Args:
            entry_price: Entry price
            current_price: Current price
            atr: Current ATR value
            direction: Trade direction (1=long, -1=short)

        Returns:
            True if take profit hit
        """
        target_distance = self.take_profit_atr * atr

        if direction == 1:  # Long position
            target_price = entry_price + target_distance
            return current_price >= target_price
        else:  # Short position
            target_price = entry_price - target_distance
            return current_price <= target_price

    def check_time_exit(
        self,
        entry_idx: int,
        current_idx: int
    ) -> bool:
        """
        Check if maximum holding period exceeded.

        Args:
            entry_idx: Entry bar index
            current_idx: Current bar index

        Returns:
            True if max bars held exceeded
        """
        bars_held = current_idx - entry_idx
        return bars_held >= self.max_bars_held

    def check_eod_exit(
        self,
        current_time: pd.Timestamp
    ) -> bool:
        """
        Check if end-of-day exit time reached.

        Args:
            current_time: Current bar timestamp

        Returns:
            True if EOD time reached
        """
        # Skip if no auto_close_time set
        if not self.auto_close_time:
            return False

        # Ensure auto_close_time is a string
        auto_close_str = str(self.auto_close_time)

        # Parse auto_close_time (e.g., "16:00")
        hour, minute = map(int, auto_close_str.split(':'))
        eod_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return current_time >= eod_time

    def get_exit(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        current_idx: int,
        direction: int = 1
    ) -> TradeExit:
        """
        Check all exit conditions in priority order.

        Priority:
        1. Strategy-specific exit
        2. Stop loss
        3. Take profit
        4. Time-based exit
        5. End-of-day exit

        Args:
            data: OHLCV dataframe
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index
            direction: Trade direction (1=long)

        Returns:
            TradeExit object
        """
        current_bar = data.iloc[current_idx]
        current_price = current_bar['Close']
        current_time = data.index[current_idx]
        atr = current_bar.get('atr', 10.0)  # Default ATR if missing

        # 1. Strategy-specific exit
        strategy_exit = self.exit_logic(
            data, self.params, entry_idx, entry_price, current_idx
        )
        if strategy_exit.exit:
            strategy_exit.exit_price = strategy_exit.exit_price or current_price
            return strategy_exit

        # 2. Stop loss
        if self.check_stop_loss(entry_price, current_price, atr, direction):
            stop_price = entry_price - (self.stop_loss_atr * atr * direction)
            return TradeExit(exit=True, exit_type='stop_loss', exit_price=stop_price)

        # 3. Take profit
        if self.check_take_profit(entry_price, current_price, atr, direction):
            target_price = entry_price + (self.take_profit_atr * atr * direction)
            return TradeExit(exit=True, exit_type='take_profit', exit_price=target_price)

        # 4. Time-based exit
        if self.check_time_exit(entry_idx, current_idx):
            return TradeExit(exit=True, exit_type='time', exit_price=current_price)

        # 5. End-of-day exit
        if self.check_eod_exit(current_time):
            return TradeExit(exit=True, exit_type='eod', exit_price=current_price)

        # No exit
        return TradeExit(exit=False, exit_type='none')

    def __repr__(self) -> str:
        """String representation of strategy."""
        param_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"


# Helper functions for common indicators

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        close: Close prices
        period: RSI period

    Returns:
        RSI values (0-100)
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Avoid division by zero
    rs = gain / loss.where(loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Resets daily (for intraday data).

    Args:
        data: DataFrame with High, Low, Close, volume columns

    Returns:
        VWAP series
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

    return vwap
