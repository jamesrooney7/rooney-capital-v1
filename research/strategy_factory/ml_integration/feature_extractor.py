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


def calculate_additional_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional features that may improve ML performance.

    Args:
        data: OHLCV dataframe

    Returns:
        DataFrame with additional feature columns
    """
    df = data.copy()

    # Price-based features
    df['returns_1'] = df['Close'].pct_change(1)
    df['returns_5'] = df['Close'].pct_change(5)
    df['returns_20'] = df['Close'].pct_change(20)

    # Volatility features
    df['volatility_5'] = df['returns_1'].rolling(5).std()
    df['volatility_20'] = df['returns_1'].rolling(20).std()

    # Range features
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['range_pct_avg_5'] = df['range_pct'].rolling(5).mean()

    # Position in range
    df['pos_in_range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)

    # IBS (Internal Bar Strength)
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)

    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_zscore'] = (
            (df['volume'] - df['volume'].rolling(20).mean()) /
            df['volume'].rolling(20).std()
        )

    # Calendar features
    if 'datetime' in df.columns or isinstance(df.index, pd.DatetimeIndex):
        dt = df['datetime'] if 'datetime' in df.columns else df.index
        df['day_of_week'] = dt.dt.dayofweek
        df['hour'] = dt.dt.hour
        df['day_of_month'] = dt.dt.day
        df['is_month_end'] = dt.dt.is_month_end.astype(int)
        df['is_month_start'] = dt.dt.is_month_start.astype(int)

    return df
