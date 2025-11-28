"""
Feature Extractor for Vectorized Strategies.

Extends the backtester to capture features at each entry point,
creating training data suitable for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add src to path for contract specs
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
try:
    from strategy.contract_specs import CONTRACT_SPECS, point_value
except ImportError:
    CONTRACT_SPECS = {}
    def point_value(symbol: str) -> float:
        return 50.0  # Default for ES

from ..strategies.base import BaseStrategy
from ..engine.backtester import Backtester, Trade, BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class TradeWithFeatures:
    """Trade record with associated features at entry time."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_gross: float  # Before commissions
    pnl_net: float    # After commissions
    pnl_return: float
    bars_held: int
    exit_type: str
    features: Dict[str, Any]


class FeatureExtractor:
    """
    Extracts features and trade outcomes from vectorized strategies.

    This class runs backtests while capturing the indicator values
    at each trade entry, creating training data for ML models.

    Output format matches train_rf_cpcv_bo.py expectations:
    - Date/Time: Entry datetime
    - Exit Date/Time: Exit datetime
    - Entry_Price: Entry price
    - Exit_Price: Exit price
    - y_return: Price return (exit-entry)/entry
    - y_binary: 1 if net PnL > 0, else 0
    - y_pnl_usd: Net PnL after commissions
    - y_pnl_gross: Gross PnL before commissions
    - [indicator columns]: Values at entry time
    """

    def __init__(
        self,
        commission_per_side: float = 1.00,
        slippage_ticks: float = 1.0
    ):
        """
        Initialize feature extractor.

        Args:
            commission_per_side: Commission per side in dollars
            slippage_ticks: Number of ticks slippage per side
        """
        self.commission_per_side = commission_per_side
        self.slippage_ticks = slippage_ticks

    def extract(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract training data from a strategy backtest.

        Args:
            strategy: Strategy instance with parameters set
            data: OHLCV dataframe
            symbol: Symbol being traded
            feature_columns: List of column names to extract as features.
                           If None, extracts all numeric columns except OHLCV.

        Returns:
            DataFrame with trade records and features in ML-ready format
        """
        # Get contract specs
        spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 0.01, "tick_value": 1.0})
        tick_size = spec["tick_size"]
        tick_value = spec["tick_value"]
        slippage_points = tick_size * self.slippage_ticks
        pv = point_value(symbol)

        # Prepare data (calculate indicators)
        data = strategy.prepare_data(data.copy())

        # Determine feature columns
        if feature_columns is None:
            # Auto-detect: all numeric columns except standard OHLCV
            exclude_cols = {'Open', 'High', 'Low', 'Close', 'volume', 'datetime', 'index'}
            feature_columns = [
                col for col in data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]

        logger.info(f"Extracting {len(feature_columns)} features: {feature_columns[:5]}...")

        # Skip warmup period
        warmup = strategy.warmup_period
        data = data.iloc[warmup:].reset_index(drop=False)

        # Generate entry signals
        entries = strategy.entry_logic(data, strategy.params)

        # Run backtest and collect trades with features
        trades_with_features: List[TradeWithFeatures] = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_time = None
        entry_features = None

        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = current_bar['datetime']
            current_price = current_bar['Close']

            # Check for exit if in position
            if in_position:
                trade_exit = strategy.get_exit(
                    data, entry_idx, entry_price, i, direction=1
                )

                if trade_exit.exit:
                    # Exit trade
                    exit_price = trade_exit.exit_price
                    exit_price -= slippage_points  # Apply slippage

                    # Calculate P&L
                    pnl_points = exit_price - entry_price
                    pnl_return = pnl_points / entry_price

                    # Gross and net PnL
                    pnl_gross = pnl_points * pv
                    commission = 2 * self.commission_per_side
                    pnl_net = pnl_gross - commission

                    # Create trade record with features
                    trade = TradeWithFeatures(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        pnl_return=pnl_return,
                        bars_held=i - entry_idx,
                        exit_type=trade_exit.exit_type,
                        features=entry_features
                    )
                    trades_with_features.append(trade)

                    # Reset position
                    in_position = False
                    entry_idx = None
                    entry_price = None
                    entry_time = None
                    entry_features = None

            # Check for new entry if not in position
            elif entries.iloc[i] and not in_position:
                # Entry signal - capture features BEFORE entry
                entry_price = current_price + slippage_points
                entry_time = current_time
                entry_idx = i

                # Capture all feature values at entry
                entry_features = {}
                for col in feature_columns:
                    val = current_bar.get(col)
                    if val is not None and not pd.isna(val):
                        entry_features[col] = val

                in_position = True

        # Convert to DataFrame
        if not trades_with_features:
            logger.warning(f"No trades extracted for {strategy.name}")
            return pd.DataFrame()

        records = []
        for trade in trades_with_features:
            record = {
                'Date/Time': trade.entry_time,
                'Exit Date/Time': trade.exit_time,
                'Date': trade.entry_time.date() if hasattr(trade.entry_time, 'date') else trade.entry_time,
                'Exit_Date': trade.exit_time.date() if hasattr(trade.exit_time, 'date') else trade.exit_time,
                'Entry_Price': trade.entry_price,
                'Exit_Price': trade.exit_price,
                'Symbol': symbol,
                'Strategy': strategy.name,
                'y_return': trade.pnl_return,
                'y_binary': 1 if trade.pnl_net > 0 else 0,
                'y_pnl_usd': trade.pnl_net,
                'y_pnl_gross': trade.pnl_gross,
                'bars_held': trade.bars_held,
                'exit_type': trade.exit_type,
            }
            # Add features
            record.update(trade.features)
            records.append(record)

        df = pd.DataFrame(records)

        logger.info(f"Extracted {len(df)} trades with {len(feature_columns)} features")

        return df

    def extract_with_cross_asset_features(
        self,
        strategy: BaseStrategy,
        primary_data: pd.DataFrame,
        symbol: str,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        lookback_periods: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        Extract training data with cross-asset features.

        This adds features from other symbols (e.g., TLT, VIX, GC)
        that may have predictive power.

        Args:
            strategy: Strategy instance with parameters set
            primary_data: OHLCV dataframe for primary symbol
            symbol: Primary symbol being traded
            cross_asset_data: Dict mapping symbol -> OHLCV dataframe
            lookback_periods: Periods for calculating returns/z-scores

        Returns:
            DataFrame with trade records, strategy features, and cross-asset features
        """
        # First extract basic features
        df = self.extract(strategy, primary_data, symbol)

        if df.empty or cross_asset_data is None:
            return df

        # Add cross-asset features
        for cross_symbol, cross_data in cross_asset_data.items():
            logger.info(f"Adding cross-asset features from {cross_symbol}")

            # Align cross-asset data with trade entry times
            cross_data = cross_data.copy()
            if 'datetime' not in cross_data.columns:
                cross_data['datetime'] = cross_data.index

            # Create features for each lookback period
            for period in lookback_periods:
                # Returns
                cross_data[f'{cross_symbol.lower()}_return_{period}'] = (
                    cross_data['Close'].pct_change(period)
                )

                # Z-scores
                rolling_mean = cross_data['Close'].rolling(period * 5).mean()
                rolling_std = cross_data['Close'].rolling(period * 5).std()
                cross_data[f'{cross_symbol.lower()}_zscore_{period}'] = (
                    (cross_data['Close'] - rolling_mean) / rolling_std
                )

            # Merge with main dataframe by nearest time
            for idx in df.index:
                entry_time = df.loc[idx, 'Date/Time']

                # Find nearest cross-asset bar
                time_diffs = abs(cross_data['datetime'] - entry_time)
                nearest_idx = time_diffs.idxmin()
                nearest_bar = cross_data.loc[nearest_idx]

                # Add cross-asset features
                for period in lookback_periods:
                    ret_col = f'{cross_symbol.lower()}_return_{period}'
                    zscore_col = f'{cross_symbol.lower()}_zscore_{period}'

                    if ret_col in cross_data.columns:
                        df.loc[idx, ret_col] = nearest_bar.get(ret_col)
                    if zscore_col in cross_data.columns:
                        df.loc[idx, zscore_col] = nearest_bar.get(zscore_col)

        return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands and related features."""
    close = data['Close']
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    # %B - where price is relative to bands (0 = lower, 1 = upper)
    pct_b = (close - lower) / (upper - lower + 0.0001)

    # Bandwidth - volatility measure
    bandwidth = (upper - lower) / sma

    return sma, upper, lower, pct_b, bandwidth


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD indicator."""
    close = data['Close']
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Calculate Stochastic oscillator."""
    high = data['High']
    low = data['Low']
    close = data['Close']

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 0.0001)
    stoch_d = stoch_k.rolling(d_period).mean()

    return stoch_k, stoch_d


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling z-score."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


def calculate_additional_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive ML features matching IbsStrategy's collect_filter_values().

    Generates 150+ features across categories:
    - Calendar features (day of week, month, DOM patterns)
    - IBS variants (current, previous, rolling)
    - RSI at multiple periods (2, 5, 14)
    - Volatility (ATR, ATR z-score, realized vol)
    - Volume (volume z-score, volume ratio, OBV)
    - Momentum (ROC, momentum z-score, distance z-score)
    - Price features (z-score, returns at various periods)
    - Pattern detection (inside bar, N consecutive bars, gaps)
    - Bollinger Bands (%B, bandwidth)
    - Moving Averages (EMA 8, 20, 50, 200 and positions)
    - MACD features
    - Stochastic features

    Args:
        data: OHLCV dataframe with datetime index or column

    Returns:
        DataFrame with 150+ additional feature columns
    """
    df = data.copy()

    # =========================================================================
    # CALENDAR FEATURES
    # =========================================================================
    if 'datetime' in df.columns or isinstance(df.index, pd.DatetimeIndex):
        dt = df['datetime'] if 'datetime' in df.columns else df.index
        if isinstance(dt, pd.DatetimeIndex):
            df['day_of_week'] = dt.dayofweek  # 0=Mon, 6=Sun
            df['hour'] = dt.hour
            df['day_of_month'] = dt.day
            df['month'] = dt.month
            df['is_month_end'] = dt.is_month_end.astype(int)
            df['is_month_start'] = dt.is_month_start.astype(int)
            df['is_quarter_end'] = dt.is_quarter_end.astype(int)
            df['week_of_year'] = dt.isocalendar().week.astype(int)
        else:
            df['day_of_week'] = dt.dt.dayofweek
            df['hour'] = dt.dt.hour
            df['day_of_month'] = dt.dt.day
            df['month'] = dt.dt.month
            df['is_month_end'] = dt.dt.is_month_end.astype(int)
            df['is_month_start'] = dt.dt.is_month_start.astype(int)
            df['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
            df['week_of_year'] = dt.dt.isocalendar().week.astype(int)

        # Derived calendar features
        df['is_beginning_of_week'] = (df['day_of_week'] <= 1).astype(int)  # Mon/Tue
        df['is_end_of_week'] = (df['day_of_week'] >= 3).astype(int)  # Thu/Fri
        df['is_even_day'] = (df['day_of_month'] % 2 == 0).astype(int)
        df['dom_first_half'] = (df['day_of_month'] <= 15).astype(int)

    # =========================================================================
    # IBS (INTERNAL BAR STRENGTH) VARIANTS
    # =========================================================================
    range_hl = df['High'] - df['Low']
    range_hl_safe = range_hl.replace(0, np.nan)

    # Current bar IBS
    df['ibs'] = (df['Close'] - df['Low']) / range_hl_safe
    df['ibs'] = df['ibs'].fillna(0.5)

    # Previous bar IBS
    df['prev_ibs'] = df['ibs'].shift(1)

    # Rolling IBS statistics
    df['ibs_sma_5'] = df['ibs'].rolling(5).mean()
    df['ibs_sma_10'] = df['ibs'].rolling(10).mean()
    df['ibs_std_10'] = df['ibs'].rolling(10).std()
    df['ibs_zscore'] = zscore(df['ibs'], 20)

    # IBS percentile (where current IBS ranks in recent history)
    df['ibs_percentile_20'] = df['ibs'].rolling(20).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    )

    # =========================================================================
    # PRICE RETURNS AT VARIOUS PERIODS
    # =========================================================================
    for period in [1, 2, 3, 5, 10, 20, 60]:
        df[f'returns_{period}'] = df['Close'].pct_change(period)

    # Previous day return (using 24 bars for ~daily on 15min data)
    df['prev_day_pct'] = df['Close'].pct_change(96)  # ~1 day on 15min
    df['prev_bar_pct'] = df['Close'].pct_change(1)

    # =========================================================================
    # RSI AT MULTIPLE PERIODS
    # =========================================================================
    for period in [2, 5, 9, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)

    # RSI z-scores and percentiles
    df['rsi_14_zscore'] = zscore(df['rsi_14'], 50)

    # =========================================================================
    # ATR AND VOLATILITY FEATURES
    # =========================================================================
    for period in [5, 10, 14, 20]:
        df[f'atr_{period}'] = calculate_atr(df, period)

    # ATR as percentage of price
    df['atr_pct'] = df['atr_14'] / df['Close']

    # ATR z-score
    df['atrz'] = zscore(df['atr_14'], 50)

    # Realized volatility
    df['volatility_5'] = df['returns_1'].rolling(5).std() * np.sqrt(252 * 24)  # Annualized
    df['volatility_10'] = df['returns_1'].rolling(10).std() * np.sqrt(252 * 24)
    df['volatility_20'] = df['returns_1'].rolling(20).std() * np.sqrt(252 * 24)
    df['volatility_60'] = df['returns_1'].rolling(60).std() * np.sqrt(252 * 24)

    # Volatility ratio (short-term vs long-term)
    df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20'].replace(0, np.nan)
    df['vol_ratio_10_60'] = df['volatility_10'] / df['volatility_60'].replace(0, np.nan)

    # =========================================================================
    # RANGE FEATURES
    # =========================================================================
    df['range_pct'] = range_hl / df['Close']
    df['range_pct_avg_5'] = df['range_pct'].rolling(5).mean()
    df['range_pct_avg_20'] = df['range_pct'].rolling(20).mean()
    df['range_expansion'] = df['range_pct'] / df['range_pct_avg_20'].replace(0, np.nan)

    # =========================================================================
    # VOLUME FEATURES (if available)
    # =========================================================================
    if 'volume' in df.columns:
        vol = df['volume'].replace(0, np.nan)

        # Volume ratios and z-scores
        df['volume_sma_20'] = vol.rolling(20).mean()
        df['volume_ratio'] = vol / df['volume_sma_20']
        df['volz'] = zscore(vol, 20)
        df['dvolz'] = zscore(vol, 50)  # Daily-ish volume z-score

        # Volume trend
        df['volume_sma_5'] = vol.rolling(5).mean()
        df['volume_trend'] = df['volume_sma_5'] / df['volume_sma_20'].replace(0, np.nan)

        # On-Balance Volume (simplified)
        df['obv_change'] = np.where(df['Close'] > df['Close'].shift(1), vol,
                                    np.where(df['Close'] < df['Close'].shift(1), -vol, 0))
        df['obv'] = df['obv_change'].cumsum()
        df['obv_zscore'] = zscore(df['obv'], 20)

    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================
    # Rate of change
    for period in [3, 5, 10, 20]:
        df[f'roc_{period}'] = (df['Close'] / df['Close'].shift(period) - 1) * 100

    # Momentum z-scores
    df['mom3_z'] = zscore(df['roc_3'], 20)
    df['mom5_z'] = zscore(df['roc_5'], 20)
    df['mom10_z'] = zscore(df['roc_10'], 20)

    # Distance from moving average (z-score)
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    df['dist_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
    df['dist_z'] = zscore(df['dist_sma_20'], 20)

    # Price z-score
    df['z_score'] = zscore(df['Close'], 20)
    df['z_score_50'] = zscore(df['Close'], 50)

    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    bb_sma, bb_upper, bb_lower, bb_pct_b, bb_bandwidth = calculate_bollinger_bands(df, 20, 2.0)
    df['bb_sma'] = bb_sma
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_pct_b'] = bb_pct_b
    df['bb_bandwidth'] = bb_bandwidth
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 0.0001)

    # Daily-ish Bollinger (using 50 period as proxy)
    _, bb_upper_d, bb_lower_d, bb_pct_b_d, bb_bw_d = calculate_bollinger_bands(df, 50, 2.0)
    df['bb_pct_b_daily'] = bb_pct_b_d
    df['bb_bandwidth_daily'] = bb_bw_d

    # =========================================================================
    # MOVING AVERAGES AND TREND
    # =========================================================================
    for period in [8, 20, 50, 100, 200]:
        df[f'ema_{period}'] = calculate_ema(df['Close'], period)

    # Price position relative to EMAs (1 = above, 0 = below)
    df['above_ema_8'] = (df['Close'] > df['ema_8']).astype(int)
    df['above_ema_20'] = (df['Close'] > df['ema_20']).astype(int)
    df['above_ema_50'] = (df['Close'] > df['ema_50']).astype(int)
    df['above_ema_200'] = (df['Close'] > df['ema_200']).astype(int)

    # EMA slope (trend direction)
    df['ema_20_slope'] = df['ema_20'].pct_change(5)
    df['ema_50_slope'] = df['ema_50'].pct_change(10)

    # EMA crossover signals
    df['ema_8_20_diff'] = (df['ema_8'] - df['ema_20']) / df['Close']
    df['ema_20_50_diff'] = (df['ema_20'] - df['ema_50']) / df['Close']

    # =========================================================================
    # MACD
    # =========================================================================
    macd_line, signal_line, histogram = calculate_macd(df)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    df['macd_zscore'] = zscore(macd_line, 20)

    # =========================================================================
    # STOCHASTIC
    # =========================================================================
    stoch_k, stoch_d = calculate_stochastic(df, 14, 3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['stoch_diff'] = stoch_k - stoch_d

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================
    # Inside bar (current bar's range inside previous bar's range)
    df['inside_bar'] = (
        (df['High'] < df['High'].shift(1)) &
        (df['Low'] > df['Low'].shift(1))
    ).astype(int)

    # Outside bar / engulfing
    df['outside_bar'] = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low'] < df['Low'].shift(1))
    ).astype(int)

    # Consecutive bearish/bullish bars
    df['bearish_bar'] = (df['Close'] < df['Open']).astype(int)
    df['bullish_bar'] = (df['Close'] > df['Open']).astype(int)

    # Count consecutive bearish bars
    df['bear_streak'] = df['bearish_bar'].groupby(
        (df['bearish_bar'] != df['bearish_bar'].shift()).cumsum()
    ).cumsum() * df['bearish_bar']

    # Count consecutive bullish bars
    df['bull_streak'] = df['bullish_bar'].groupby(
        (df['bullish_bar'] != df['bullish_bar'].shift()).cumsum()
    ).cumsum() * df['bullish_bar']

    # N-bar patterns
    df['n7_bar'] = (df['Low'] <= df['Low'].rolling(7).min()).astype(int)
    df['n10_bar_low'] = (df['Low'] <= df['Low'].rolling(10).min()).astype(int)
    df['n10_bar_high'] = (df['High'] >= df['High'].rolling(10).max()).astype(int)

    # Gap features
    df['gap'] = df['Open'] - df['Close'].shift(1)
    df['gap_pct'] = df['gap'] / df['Close'].shift(1)
    df['gap_filled'] = (
        ((df['gap'] > 0) & (df['Low'] <= df['Close'].shift(1))) |
        ((df['gap'] < 0) & (df['High'] >= df['Close'].shift(1)))
    ).astype(int)

    # =========================================================================
    # CANDLE FEATURES
    # =========================================================================
    body = abs(df['Close'] - df['Open'])
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']

    df['body_pct'] = body / range_hl_safe
    df['upper_wick_pct'] = upper_wick / range_hl_safe
    df['lower_wick_pct'] = lower_wick / range_hl_safe

    # Doji detection (small body relative to range)
    df['is_doji'] = (body / range_hl_safe < 0.1).astype(int)

    # Hammer pattern (long lower wick, small body at top)
    df['is_hammer'] = (
        (df['lower_wick_pct'] > 0.6) &
        (df['body_pct'] < 0.3) &
        (df['upper_wick_pct'] < 0.1)
    ).astype(int)

    # Shooting star (long upper wick, small body at bottom)
    df['is_shooting_star'] = (
        (df['upper_wick_pct'] > 0.6) &
        (df['body_pct'] < 0.3) &
        (df['lower_wick_pct'] < 0.1)
    ).astype(int)

    # =========================================================================
    # HIGH/LOW RELATIVE FEATURES
    # =========================================================================
    for period in [5, 10, 20, 50]:
        df[f'high_{period}'] = df['High'].rolling(period).max()
        df[f'low_{period}'] = df['Low'].rolling(period).min()
        df[f'pct_from_high_{period}'] = (df['Close'] - df[f'high_{period}']) / df[f'high_{period}']
        df[f'pct_from_low_{period}'] = (df['Close'] - df[f'low_{period}']) / df[f'low_{period}']

    # =========================================================================
    # EFFICIENCY RATIO (Trend strength)
    # =========================================================================
    for period in [10, 20]:
        direction = abs(df['Close'] - df['Close'].shift(period))
        volatility = df['Close'].diff().abs().rolling(period).sum()
        df[f'efficiency_ratio_{period}'] = direction / volatility.replace(0, np.nan)

    # =========================================================================
    # DAILY AGGREGATES (using 96 bars as proxy for daily on 15-min data)
    # =========================================================================
    daily_period = 96  # Approximate daily bars for 15-min data

    # Daily IBS (approximate)
    daily_high = df['High'].rolling(daily_period).max()
    daily_low = df['Low'].rolling(daily_period).min()
    df['daily_ibs'] = (df['Close'] - daily_low) / (daily_high - daily_low + 0.0001)

    # Previous daily IBS
    df['prev_daily_ibs'] = df['daily_ibs'].shift(daily_period)

    # Daily ATR z-score
    df['datrz'] = zscore(df['atr_14'].rolling(daily_period).mean(), 50)

    # =========================================================================
    # CLEAN UP
    # =========================================================================
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df
