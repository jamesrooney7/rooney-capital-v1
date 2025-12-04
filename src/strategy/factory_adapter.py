"""
Strategy Factory Adapter for Backtrader Live Execution.

This module bridges Strategy Factory research strategies to the Backtrader
live execution environment. It allows running any Strategy Factory strategy
(AvgHLRangeIBS, IBSStrategy, etc.) in production with ML filtering.

The adapter:
1. Accumulates OHLCV bars into a DataFrame for strategy calculations
2. Calls Strategy Factory entry_logic/exit_logic methods
3. Integrates with ML models for trade filtering
4. Places orders via Backtrader broker

Usage in config.yml:
    portfolio:
      instruments:
        - symbol: ES
          strategy: AvgHLRangeIBS
          params:
            length: 20
            range_multiplier: 2.5
            bars_below_threshold: 2
            ibs_buy_threshold: 0.2
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import backtrader as bt
import pandas as pd
import numpy as np
from collections import deque as collections_deque

# Import trades database
try:
    from utils.trades_db import TradesDB
except ImportError:
    TradesDB = None

# Import Strategy Factory base and strategies
import sys
from pathlib import Path

# Symbols to track for cross-asset features
CROSS_ASSET_SYMBOLS = [
    'ES', 'NQ', 'YM', 'RTY',  # Equity indices
    '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S',  # Currencies
    'CL', 'GC', 'SI', 'HG', 'NG', 'PL',  # Commodities
    'TLT',  # Reference
]

# Feature name mappings - maps model feature names to calculation keys
FEATURE_NAME_MAP = {
    'NQ Hourly Return': ('NQ', 'hourly_return'),
    'ES Hourly Return': ('ES', 'hourly_return'),
    'YM Hourly Return': ('YM', 'hourly_return'),
    'RTY Hourly Return': ('RTY', 'hourly_return'),
    '6J Hourly Return': ('6J', 'hourly_return'),
    '6A Hourly Return': ('6A', 'hourly_return'),
    '6M Hourly Return': ('6M', 'hourly_return'),
    '6C Hourly Return': ('6C', 'hourly_return'),
    '6E Hourly Return': ('6E', 'hourly_return'),
    '6B Hourly Return': ('6B', 'hourly_return'),
    'nq_hourly_return': ('NQ', 'hourly_return'),
    'es_hourly_return': ('ES', 'hourly_return'),
    'ym_hourly_return': ('YM', 'hourly_return'),
    'rty_hourly_return': ('RTY', 'hourly_return'),
    'ES Hourly Z Score': ('ES', 'hourly_zscore'),
    '6B Hourly Z Score': ('6B', 'hourly_zscore'),
    '6B Daily Z Score': ('6B', 'daily_zscore'),
    'nq_z_score_hour': ('NQ', 'hourly_zscore'),
    '6m_z_score_hour': ('6M', 'hourly_zscore'),
    '6n_daily_z_score': ('6N', 'daily_zscore'),
    '6a_daily_z_score': ('6A', 'daily_zscore'),
}

# Add research to path for Strategy Factory imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.strategy_factory.strategies.base import BaseStrategy, TradeExit
from research.strategy_factory.strategies.avg_hl_range_ibs import AvgHLRangeIBS
from research.strategy_factory.strategies.atr_buy_dip import ATRBuyDip
from research.strategy_factory.strategies.moving_average_envelope import MovingAverageEnvelope
from research.strategy_factory.strategies.ibs_strategy import IBSStrategy

logger = logging.getLogger(__name__)

# Registry of available Strategy Factory strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'AvgHLRangeIBS': AvgHLRangeIBS,
    'ATRBuyDip': ATRBuyDip,
    'MovingAverageEnvelope': MovingAverageEnvelope,
    'IBSStrategy': IBSStrategy,
}

# Optimal strategy mapping from ML validation results (Dec 2024)
# Only includes strategies that PASSED ML validation with positive improvement
VALIDATED_STRATEGIES: Dict[str, Dict[str, Any]] = {
    'NQ': {'strategy': 'ATRBuyDip', 'sharpe': 2.97},
    'CL': {'strategy': 'ATRBuyDip', 'sharpe': 2.15},
    'HG': {'strategy': 'MovingAverageEnvelope', 'sharpe': 2.01},
    'GC': {'strategy': 'ATRBuyDip', 'sharpe': 1.87},
    'ES': {'strategy': 'AvgHLRangeIBS', 'sharpe': 1.83},
    # Note: 6B and RTY failed ML validation (Sharpe degraded after ML filtering)
}


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """Get Strategy Factory strategy class by name."""
    if name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGY_REGISTRY[name]


class StrategyFactoryAdapter(bt.Strategy):
    """
    Backtrader strategy that wraps a Strategy Factory strategy.

    This adapter maintains a rolling window of OHLCV data, converts it to
    a DataFrame, and calls the underlying Strategy Factory strategy's
    entry_logic and exit_logic methods.

    Parameters:
        symbol: Trading symbol (e.g., 'ES')
        strategy_name: Name of Strategy Factory strategy (e.g., 'AvgHLRangeIBS')
        strategy_params: Parameters for the strategy
        ml_model: Optional ML model for trade filtering
        ml_features: Feature names for ML model
        ml_threshold: ML prediction threshold (default 0.5)
        warmup_bars: Number of bars to accumulate before trading
        order_callbacks: Callbacks for order notifications
        trade_callbacks: Callbacks for trade notifications
    """

    params = (
        ('symbol', None),
        ('strategy_name', 'AvgHLRangeIBS'),
        ('strategy_params', {}),
        ('ml_model', None),
        ('ml_features', []),
        ('ml_threshold', 0.5),
        ('warmup_bars', 252),  # 1 year of daily bars
        ('order_callbacks', []),
        ('trade_callbacks', []),
        ('portfolio_coordinator', None),
        ('feature_tracker', None),
        ('queue_manager', None),  # QueueFanout for contract symbol lookup
        ('size', 1),  # Position size (number of contracts)
        ('ml_feature_collector', None),  # Optional feature collector for ML warmup
        ('trades_db', None),  # TradesDB instance for persisting trades
    )

    def __init__(self):
        """Initialize the adapter."""
        self.symbol = self.p.symbol

        # Initialize the Strategy Factory strategy
        strategy_cls = get_strategy_class(self.p.strategy_name)
        self.factory_strategy = strategy_cls(params=self.p.strategy_params)

        logger.info(
            f"StrategyFactoryAdapter initialized for {self.symbol} "
            f"using {self.p.strategy_name} with params: {self.p.strategy_params}"
        )

        # Rolling window of bars for DataFrame construction
        self._bar_buffer: deque = deque(maxlen=max(500, self.p.warmup_bars + 50))
        self._bars_seen = 0

        # State tracking
        self._in_position = False
        self._entry_idx = None
        self._entry_price = None
        self._entry_time = None
        self._order_pending = False
        self._pending_order = None

        # ML filtering
        self._ml_last_score = None
        self._latest_ml_score = None  # Alias for compatibility with Discord callback

        # Data feeds - build map of symbol -> data feed
        # NOTE: Using 15-minute bars to match Strategy Factory training
        self.hourly_data = self.datas[0]  # Primary 15-min data (named hourly_data for compatibility)
        self.daily_data = None
        self._data_feeds: Dict[str, Any] = {}  # symbol -> 15-min data feed
        self._daily_feeds: Dict[str, Any] = {}  # symbol -> daily data feed

        # Find and map all data feeds
        # Priority: 15-min feeds > base feeds > daily feeds for feature calculation
        for d in self.datas:
            name = getattr(d, '_name', '')
            if not name:
                continue

            # Check for daily feed
            if '_day' in name.lower() or '_daily' in name.lower():
                if self.symbol.upper() in name.upper():
                    self.daily_data = d
                # Extract symbol and store daily feed
                symbol_part = name.split('_')[0].upper()
                if symbol_part:
                    self._daily_feeds[symbol_part] = d
                continue

            # Extract symbol from feed name (e.g., "ES_15min" -> "ES", "ES" -> "ES")
            symbol_part = name.split('_')[0].upper()
            if not symbol_part:
                continue

            # Prefer 15-min feeds for feature calculation (also check for _hour for backwards compat)
            is_15min = '_15min' in name.lower() or '_hour' in name.lower()
            if is_15min or symbol_part not in self._data_feeds:
                self._data_feeds[symbol_part] = d

        # Set hourly_data to the correct feed for THIS strategy's symbol
        # (kept as hourly_data for compatibility, but actually uses 15-min bars)
        symbol_upper = self.symbol.upper()
        if symbol_upper in self._data_feeds:
            self.hourly_data = self._data_feeds[symbol_upper]
            logger.info(f"{self.symbol}: Using data feed '{getattr(self.hourly_data, '_name', 'unknown')}'")
        else:
            logger.warning(f"{self.symbol}: No matching 15-min feed found, using datas[0]")

        # Cross-asset feature calculation
        # Price buffers for calculating returns and z-scores
        self._price_buffers: Dict[str, deque] = {}  # symbol -> deque of closes
        self._daily_price_buffers: Dict[str, deque] = {}  # symbol -> deque of daily closes
        zscore_window = 20  # Window for z-score calculation

        for sym in CROSS_ASSET_SYMBOLS:
            self._price_buffers[sym] = collections_deque(maxlen=100)
            self._daily_price_buffers[sym] = collections_deque(maxlen=60)

        # Feature collector for ML features
        self._ml_feature_collector = None
        if self.p.feature_tracker:
            ml_features = self.p.ml_features if self.p.ml_features else ()
            self._ml_feature_collector = self.p.feature_tracker.register_bundle(
                self.symbol, ml_features
            )

        # Track last bar time to detect new bars
        self._last_bar_time = None

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed]:
            self._order_pending = False

            if order.isbuy():
                self._in_position = True
                self._entry_price = order.executed.price
                self._entry_time = self.hourly_data.datetime.datetime(0)
                self._entry_idx = self._bars_seen
                # Diagnostic: log bar data at execution time
                bar_dt = self.hourly_data.datetime.datetime(0)
                bar_open = self.hourly_data.open[0]
                bar_close = self.hourly_data.close[0]
                logger.info(
                    f"{self.symbol}: BUY executed at {order.executed.price:.2f} | "
                    f"Bar[0]: dt={bar_dt}, O={bar_open:.2f}, C={bar_close:.2f}"
                )
            else:
                self._in_position = False
                pnl = order.executed.price - self._entry_price if self._entry_price else 0
                # Diagnostic: log bar data at execution time
                bar_dt = self.hourly_data.datetime.datetime(0)
                bar_open = self.hourly_data.open[0]
                bar_close = self.hourly_data.close[0]
                logger.info(
                    f"{self.symbol}: SELL executed at {order.executed.price:.2f}, "
                    f"P&L: {pnl:.2f} | Bar[0]: dt={bar_dt}, O={bar_open:.2f}, C={bar_close:.2f}"
                )
                self._entry_idx = None
                self._entry_price = None
                self._entry_time = None

            # Notify callbacks
            for callback in self.p.order_callbacks:
                try:
                    callback(self, order)
                except Exception as e:
                    logger.error(f"Order callback error: {e}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._order_pending = False
            logger.warning(f"{self.symbol}: Order {order.status}")

    def notify_trade(self, trade):
        """Handle trade notifications."""
        exit_snapshot = None
        if trade.isclosed:
            # Capture exit snapshot for callbacks
            exit_snapshot = {
                'exit_reason': 'strategy_exit',
                'strategy_name': self.p.strategy_name,
                'symbol': self.symbol,
                'size': trade.size,
                'price': trade.price,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'dt': self.hourly_data.datetime.datetime(0),
                'ml_score': self._ml_last_score,
            }

            logger.info(
                f"{self.symbol}: Trade closed, P&L: {trade.pnl:.2f}, "
                f"Commission: {trade.commission:.2f}"
            )

            # Notify portfolio coordinator
            if self.p.portfolio_coordinator:
                self.p.portfolio_coordinator.register_position_closed(
                    symbol=self.symbol,
                    pnl=trade.pnlcomm
                )

            # Persist trade to database
            if self.p.trades_db is not None:
                try:
                    # Calculate P&L percentage
                    entry_price = self._entry_price or trade.price
                    exit_price = trade.price
                    pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0.0

                    self.p.trades_db.insert_trade(
                        symbol=self.symbol,
                        side='long',  # Factory strategies are long-only for now
                        entry_time=self._entry_time or datetime.now(),
                        entry_price=entry_price,
                        entry_size=abs(self.p.size),
                        exit_time=self.hourly_data.datetime.datetime(0),
                        exit_price=exit_price,
                        exit_size=abs(self.p.size),
                        pnl=trade.pnlcomm,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_snapshot.get('exit_reason', 'strategy_exit'),
                        ml_score=self._ml_last_score,
                    )
                    logger.info(f"{self.symbol}: Trade persisted to database")
                except Exception as e:
                    logger.error(f"{self.symbol}: Failed to persist trade to database: {e}")

            # Notify callbacks with exit_snapshot (matching NotifyingIbsStrategy signature)
            for callback in self.p.trade_callbacks:
                try:
                    callback(self, trade, exit_snapshot)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")

    def next(self):
        """Process each bar."""
        self._bars_seen += 1

        # Accumulate bar data
        bar = {
            'datetime': self.hourly_data.datetime.datetime(0),
            'Open': self.hourly_data.open[0],
            'High': self.hourly_data.high[0],
            'Low': self.hourly_data.low[0],
            'Close': self.hourly_data.close[0],
            'volume': self.hourly_data.volume[0] if hasattr(self.hourly_data, 'volume') else 0,
        }
        self._bar_buffer.append(bar)

        # Update cross-asset features for ML filtering
        self._update_cross_asset_features()

        # Log progress periodically (every 50 bars or first 5 bars)
        if self._bars_seen <= 5 or self._bars_seen % 50 == 0:
            logger.info(
                f"{self.symbol}: Bar {self._bars_seen}/{self.p.warmup_bars} "
                f"at {bar['datetime']} C={bar['Close']:.2f}"
            )

        # Wait for warmup
        if self._bars_seen < self.p.warmup_bars:
            if self._bars_seen == 1:
                logger.info(f"{self.symbol}: Starting warmup (need {self.p.warmup_bars} bars)")
            return

        # Log when warmup completes
        if self._bars_seen == self.p.warmup_bars:
            logger.info(f"{self.symbol}: Warmup complete - now evaluating signals")
            # Log data feeds available for cross-asset features
            available_feeds = list(self._data_feeds.keys())
            logger.info(f"{self.symbol}: Data feeds available for features: {sorted(available_feeds)}")
            # Log expected ML features
            if self.p.ml_features:
                logger.info(f"{self.symbol}: ML model expects {len(self.p.ml_features)} features: {list(self.p.ml_features)[:5]}...")

        # Skip if order pending
        if self._order_pending:
            return

        # Check portfolio stop loss
        if self.p.portfolio_coordinator:
            if self.p.portfolio_coordinator.stopped_out:
                if self._in_position:
                    self._exit_position('portfolio_stop')
                return

        # Build DataFrame from buffer
        df = self._build_dataframe()
        if df is None or len(df) < self.factory_strategy.warmup_period:
            return

        # Calculate indicators
        try:
            df = self.factory_strategy.prepare_data(df)
        except Exception as e:
            logger.error(f"{self.symbol}: Error calculating indicators: {e}")
            return

        # Current bar index in DataFrame
        current_idx = len(df) - 1

        if self._in_position:
            # Check exit conditions
            self._check_exit(df, current_idx)
        else:
            # Check entry conditions
            self._check_entry(df, current_idx)

    def _build_dataframe(self) -> Optional[pd.DataFrame]:
        """Build DataFrame from bar buffer."""
        if len(self._bar_buffer) < 10:
            return None

        df = pd.DataFrame(list(self._bar_buffer))
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def _check_entry(self, df: pd.DataFrame, current_idx: int):
        """Check if entry conditions are met."""
        # Get entry signals from Strategy Factory
        try:
            entry_signals = self.factory_strategy.entry_logic(
                df, self.factory_strategy.params
            )
        except Exception as e:
            logger.error(f"{self.symbol}: Error in entry_logic: {e}")
            return

        # Check if current bar has entry signal
        has_signal = entry_signals.iloc[current_idx] if current_idx < len(entry_signals) else False

        # Log signal status periodically (every 10 bars after warmup)
        if self._bars_seen % 10 == 0:
            recent_signals = entry_signals.tail(10).sum()

            # Get ML feature status
            ml_status = self._get_ml_feature_status()

            # Debug: show buffer sizes
            bar_buf_size = len(self._bar_buffer)
            price_buf_sizes = {k: len(v) for k, v in self._price_buffers.items() if len(v) > 0}
            feeds_available = len(self._data_feeds)

            logger.info(
                f"{self.symbol}: Signal check - current={has_signal}, "
                f"last_10_bars_signals={recent_signals}, close={df['Close'].iloc[-1]:.2f}, "
                f"ml_features={ml_status}, bar_buf={bar_buf_size}, feeds={feeds_available}"
            )

            # Log detailed debug info every 100 bars
            if self._bars_seen % 100 == 0:
                logger.info(f"{self.symbol}: DEBUG - price_buffers: {price_buf_sizes}")
                logger.info(f"{self.symbol}: DEBUG - data_feeds: {sorted(self._data_feeds.keys())}")

        if not has_signal:
            return

        logger.info(f"{self.symbol}: 📊 ENTRY SIGNAL detected at {df['datetime'].iloc[-1]}")

        # Check portfolio coordinator for position limits
        if self.p.portfolio_coordinator:
            if not self.p.portfolio_coordinator.can_open_position(self.symbol):
                logger.info(f"{self.symbol}: Position blocked by portfolio coordinator (max positions reached)")
                return

        # ML filtering
        if self.p.ml_model is not None:
            ml_score = self._get_ml_score(df, current_idx)
            self._ml_last_score = ml_score
            self._latest_ml_score = ml_score  # Alias for compatibility

            if ml_score is None:
                logger.warning(f"{self.symbol}: ML score is None - features may be missing")
                return

            if ml_score < self.p.ml_threshold:
                logger.info(
                    f"{self.symbol}: ❌ Entry blocked by ML filter "
                    f"(score={ml_score:.3f} < threshold={self.p.ml_threshold})"
                )
                return

            logger.info(
                f"{self.symbol}: ✅ ML approved entry (score={ml_score:.3f} >= {self.p.ml_threshold})"
            )
        else:
            logger.info(f"{self.symbol}: No ML model - proceeding without filter")

        # Place entry order
        self._enter_position()

    def _check_exit(self, df: pd.DataFrame, current_idx: int):
        """Check if exit conditions are met."""
        if self._entry_idx is None or self._entry_price is None:
            return

        # Map entry index to DataFrame index
        # Entry was at _entry_idx bars ago from start
        bars_since_entry = self._bars_seen - self._entry_idx
        df_entry_idx = max(0, current_idx - bars_since_entry)

        # Check exit using Strategy Factory
        try:
            exit_result = self.factory_strategy.get_exit(
                data=df,
                entry_idx=df_entry_idx,
                entry_price=self._entry_price,
                current_idx=current_idx,
                direction=1  # Long only
            )
        except Exception as e:
            logger.error(f"{self.symbol}: Error in exit_logic: {e}")
            return

        if exit_result.exit:
            self._exit_position(exit_result.exit_type)

    def _enter_position(self):
        """Place entry order."""
        if self._order_pending or self._in_position:
            return

        # Note: Position slot is already reserved by can_open_position() call in _check_entry()
        # No need to call reserve_position separately

        self._order_pending = True
        # Use self.hourly_data to ensure we trade on the correct symbol's feed
        self._pending_order = self.buy(data=self.hourly_data, size=self.p.size)

        # Diagnostic: log bar data when order is placed
        bar_dt = self.hourly_data.datetime.datetime(0)
        bar_open = self.hourly_data.open[0]
        bar_close = self.hourly_data.close[0]
        logger.info(
            f"{self.symbol}: Entry order placed (size={self.p.size}) | "
            f"Bar[0]: dt={bar_dt}, O={bar_open:.2f}, C={bar_close:.2f}"
        )

    def _exit_position(self, reason: str):
        """Place exit order."""
        if self._order_pending or not self._in_position:
            return

        self._order_pending = True
        # Use self.hourly_data to ensure we close on the correct symbol's feed
        self._pending_order = self.close(data=self.hourly_data)

        # Diagnostic: log bar data when order is placed
        bar_dt = self.hourly_data.datetime.datetime(0)
        bar_open = self.hourly_data.open[0]
        bar_close = self.hourly_data.close[0]
        logger.info(
            f"{self.symbol}: Exit order placed (reason: {reason}) | "
            f"Bar[0]: dt={bar_dt}, O={bar_open:.2f}, C={bar_close:.2f}"
        )

    def _get_ml_score(self, df: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Get ML model prediction score for current entry."""
        if self.p.ml_model is None:
            return None

        try:
            # Get feature values from feature tracker or DataFrame
            features = {}
            if self.p.feature_tracker:
                # Use snapshot() method - returns dict of feature name -> value
                features = dict(self.p.feature_tracker.snapshot(self.symbol))

            # Fallback: Extract features from DataFrame if tracker is empty
            if not features:
                current_row = df.iloc[current_idx]
                for f in self.p.ml_features:
                    if f in current_row:
                        features[f] = current_row[f]

            if not features:
                logger.warning(f"{self.symbol}: No ML features available")
                return None

            # Check if we have all required features
            missing = [f for f in self.p.ml_features if f not in features or features[f] is None]
            if missing:
                logger.debug(
                    f"{self.symbol}: Missing {len(missing)} ML features: {missing[:5]}..."
                )
                return None

            # Build feature vector in correct order
            X = pd.DataFrame([{f: features.get(f) for f in self.p.ml_features}])

            # Get prediction probability
            if hasattr(self.p.ml_model, 'predict_proba'):
                proba = self.p.ml_model.predict_proba(X)[0, 1]
                return float(proba)
            else:
                pred = self.p.ml_model.predict(X)[0]
                return float(pred)

        except Exception as e:
            logger.error(f"{self.symbol}: ML scoring error: {e}")
            return None

    def _get_ml_feature_status(self) -> str:
        """Get status string showing ML feature availability.

        Returns a string like "10/15" meaning 10 of 15 required features are available.
        """
        if not self.p.ml_features:
            return "no_model"

        total_features = len(self.p.ml_features)

        if self.p.feature_tracker is None:
            return f"0/{total_features} (no tracker)"

        try:
            # Get current feature snapshot
            features = dict(self.p.feature_tracker.snapshot(self.symbol))

            # Count features that are available and not None/NaN
            available = 0
            missing = []
            for f in self.p.ml_features:
                if f in features and features[f] is not None:
                    try:
                        import math
                        if not math.isnan(features[f]):
                            available += 1
                        else:
                            missing.append(f)
                    except (TypeError, ValueError):
                        # Not a number, but has a value
                        available += 1
                else:
                    missing.append(f)

            # Log missing features periodically (first time or every 100 bars)
            if missing and (self._bars_seen == self.p.warmup_bars or self._bars_seen % 100 == 0):
                logger.warning(
                    f"{self.symbol}: Missing ML features ({len(missing)}/{total_features}): "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )

            return f"{available}/{total_features}"
        except Exception as e:
            return f"error: {e}"

    def _update_cross_asset_features(self):
        """Update cross-asset features from all available data feeds.

        This method:
        1. Collects close prices from all cross-asset symbols
        2. Calculates hourly returns
        3. Calculates z-scores
        4. Publishes features to the ML feature tracker
        """
        if self._ml_feature_collector is None:
            return

        # Update price buffers from all data feeds
        for sym in CROSS_ASSET_SYMBOLS:
            data = self._data_feeds.get(sym)
            if data is None:
                continue

            try:
                # Get current close price
                close = data.close[0]
                if close is not None and not np.isnan(close):
                    self._price_buffers[sym].append(float(close))
            except (IndexError, AttributeError):
                continue

        # Calculate and publish features
        self._calculate_and_publish_features()

    def _calculate_and_publish_features(self):
        """Calculate features from price buffers and publish to tracker.

        IMPORTANT: Features are calculated to match Strategy Factory training:
        - Using 15-minute bars
        - hourly_return = 4-bar return (4 * 15min = 60min = 1 hour)
        - hourly_z_score = 20-bar z-score (~5 hours on 15min data)
        """
        if self._ml_feature_collector is None:
            return

        # Return periods to match training: hourly=4 bars, daily=96 bars on 15-min data
        HOURLY_BARS = 4  # 4 * 15min = 60min = 1 hour
        DAILY_BARS = 96  # 96 * 15min = 1440min = 24 hours

        # Calculate hourly and daily returns/z-scores for all symbols
        for sym in CROSS_ASSET_SYMBOLS:
            buffer = self._price_buffers.get(sym)
            if buffer is None or len(buffer) < HOURLY_BARS + 1:
                continue

            prices = list(buffer)
            sym_lower = sym.lower()

            # Hourly return: 4-bar return to match training (4 * 15min = 1 hour)
            if len(prices) >= HOURLY_BARS + 1:
                hourly_return = (prices[-1] - prices[-HOURLY_BARS - 1]) / prices[-HOURLY_BARS - 1] if prices[-HOURLY_BARS - 1] != 0 else 0.0
            else:
                hourly_return = 0.0

            # Publish hourly return with various naming conventions (model may use any)
            hourly_return_names = [
                f"{sym} Hourly Return",
                f"{sym_lower}_hourly_return",
                f"{sym_lower}_daily_return",  # Some models use this for hourly too
            ]
            for fname in hourly_return_names:
                try:
                    self._ml_feature_collector.record_feature(fname, hourly_return)
                except Exception:
                    pass

            # Calculate z-score if we have enough data (20+ bars)
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                mean_price = sum(recent_prices) / len(recent_prices)
                std_price = np.std(recent_prices)

                if std_price > 0:
                    zscore = (prices[-1] - mean_price) / std_price

                    # Publish z-score with various naming conventions (model may use any)
                    zscore_names = [
                        f"{sym} Hourly Z Score",
                        f"{sym_lower}_z_score_hour",
                        f"{sym_lower}_hourly_z_score",
                        f"{sym_lower}_hourly_z_pipeline",  # Model uses this format
                        f"{sym_lower}_daily_z_score",      # Model uses this format
                        f"{sym_lower}_daily_z_pipeline",   # Model uses this format
                        f"enable{sym}ZScoreHour",
                    ]
                    for fname in zscore_names:
                        try:
                            self._ml_feature_collector.record_feature(fname, zscore)
                        except Exception:
                            pass

        # Calculate and publish strategy-specific features from DataFrame
        self._publish_strategy_features()

    def _publish_strategy_features(self):
        """Publish strategy-specific features from the bar buffer."""
        if self._ml_feature_collector is None or len(self._bar_buffer) < 20:
            return

        try:
            # Build DataFrame from buffer for indicator calculation
            df = pd.DataFrame(list(self._bar_buffer))
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)

            if len(df) < 14:
                return

            # Calculate RSI
            close = df['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            if not rsi.empty and not np.isnan(rsi.iloc[-1]):
                self._ml_feature_collector.record_feature('rsi_len14', float(rsi.iloc[-1]))
                self._ml_feature_collector.record_feature('daily_rsi', float(rsi.iloc[-1]))

            # Calculate IBS (Internal Bar Strength)
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            close_val = df['Close'].iloc[-1]
            if high != low:
                ibs = (close_val - low) / (high - low)
                self._ml_feature_collector.record_feature('pair_ibs_daily', float(ibs))

            # Calculate ATR Z-score
            if len(df) >= 20:
                tr = np.maximum(
                    df['High'] - df['Low'],
                    np.maximum(
                        abs(df['High'] - df['Close'].shift(1)),
                        abs(df['Low'] - df['Close'].shift(1))
                    )
                )
                atr = tr.rolling(window=14).mean()
                atr_mean = atr.rolling(window=20).mean()
                atr_std = atr.rolling(window=20).std()

                if atr_std.iloc[-1] > 0:
                    atrz = (atr.iloc[-1] - atr_mean.iloc[-1]) / atr_std.iloc[-1]
                    self._ml_feature_collector.record_feature('atrz_pct', float(atrz))
                    self._ml_feature_collector.record_feature('enableDATRZ', float(atrz))

            # RSI percentile (simplified - just use RSI value scaled)
            if not rsi.empty and not np.isnan(rsi.iloc[-1]):
                # Approximate percentile based on RSI value
                rsi_percentile = rsi.iloc[-1] / 100.0
                self._ml_feature_collector.record_feature('rsi_len_2_percentile', float(rsi_percentile))

            # bars_held - always 0 at entry evaluation
            self._ml_feature_collector.record_feature('bars_held', 0.0)

            # SMA 50
            if len(df) >= 50:
                sma_50 = close.rolling(window=50).mean().iloc[-1]
                if not np.isnan(sma_50):
                    self._ml_feature_collector.record_feature('sma_50', float(sma_50))

            # EMA 20 and 50 difference
            if len(df) >= 50:
                ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
                ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
                if not np.isnan(ema_20) and not np.isnan(ema_50):
                    ema_diff = ema_20 - ema_50
                    self._ml_feature_collector.record_feature('ema_20_50_diff', float(ema_diff))

            # Momentum 3-period z-score
            if len(df) >= 23:  # Need 20 + 3 for z-score of 3-period momentum
                mom3 = close.diff(3)
                mom3_mean = mom3.rolling(window=20).mean()
                mom3_std = mom3.rolling(window=20).std()
                if mom3_std.iloc[-1] > 0:
                    mom3_z = (mom3.iloc[-1] - mom3_mean.iloc[-1]) / mom3_std.iloc[-1]
                    self._ml_feature_collector.record_feature('mom3_z', float(mom3_z))

            # Volume z-score (dvolz)
            if 'volume' in df.columns and len(df) >= 20:
                vol = df['volume']
                vol_mean = vol.rolling(window=20).mean()
                vol_std = vol.rolling(window=20).std()
                if vol_std.iloc[-1] > 0:
                    dvolz = (vol.iloc[-1] - vol_mean.iloc[-1]) / vol_std.iloc[-1]
                    self._ml_feature_collector.record_feature('dvolz', float(dvolz))
                    self._ml_feature_collector.record_feature('volz', float(dvolz))

            # IBS (Internal Bar Strength) features
            high = df['High']
            low = df['Low']
            ibs_series = (close - low) / (high - low).replace(0, np.nan)
            ibs_val = ibs_series.iloc[-1]
            if not np.isnan(ibs_val):
                self._ml_feature_collector.record_feature('ibs', float(ibs_val))
                # IBS z-score
                if len(ibs_series) >= 20:
                    ibs_mean = ibs_series.rolling(window=20).mean()
                    ibs_std_val = ibs_series.rolling(window=20).std()
                    if ibs_std_val.iloc[-1] > 0:
                        ibs_z = (ibs_val - ibs_mean.iloc[-1]) / ibs_std_val.iloc[-1]
                        self._ml_feature_collector.record_feature('ibs_zscore', float(ibs_z))
                # IBS SMA 5
                if len(ibs_series) >= 5:
                    ibs_sma = ibs_series.rolling(window=5).mean().iloc[-1]
                    if not np.isnan(ibs_sma):
                        self._ml_feature_collector.record_feature('ibs_sma_5', float(ibs_sma))
                # IBS std 10
                if len(ibs_series) >= 10:
                    ibs_std_10 = ibs_series.rolling(window=10).std().iloc[-1]
                    if not np.isnan(ibs_std_10):
                        self._ml_feature_collector.record_feature('ibs_std_10', float(ibs_std_10))
                # Previous daily IBS
                if len(ibs_series) >= 2:
                    prev_ibs = ibs_series.iloc[-2]
                    if not np.isnan(prev_ibs):
                        self._ml_feature_collector.record_feature('prev_daily_ibs', float(prev_ibs))

            # OBV (On Balance Volume)
            if 'volume' in df.columns and len(df) >= 2:
                obv = (np.sign(close.diff()) * df['volume']).fillna(0).cumsum()
                obv_val = obv.iloc[-1]
                self._ml_feature_collector.record_feature('obv', float(obv_val))
                # OBV z-score
                if len(obv) >= 20:
                    obv_mean = obv.rolling(window=20).mean()
                    obv_std = obv.rolling(window=20).std()
                    if obv_std.iloc[-1] > 0:
                        obv_z = (obv_val - obv_mean.iloc[-1]) / obv_std.iloc[-1]
                        self._ml_feature_collector.record_feature('obv_zscore', float(obv_z))

            # ATR z-score (datrz/atrz)
            if len(df) >= 20:
                tr = np.maximum(
                    high - low,
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
                )
                atr = tr.rolling(window=14).mean()
                atr_mean = atr.rolling(window=20).mean()
                atr_std = atr.rolling(window=20).std()
                if atr_std.iloc[-1] > 0:
                    atrz_val = (atr.iloc[-1] - atr_mean.iloc[-1]) / atr_std.iloc[-1]
                    self._ml_feature_collector.record_feature('atrz', float(atrz_val))
                    self._ml_feature_collector.record_feature('datrz', float(atrz_val))

            # Previous day percentage change
            if len(close) >= 3:
                prev_day_pct = (close.iloc[-2] - close.iloc[-3]) / close.iloc[-3] if close.iloc[-3] != 0 else 0.0
                self._ml_feature_collector.record_feature('prev_day_pct', float(prev_day_pct))

            # RSI 14 and RSI 21
            for period in [14, 21]:
                if len(df) >= period + 1:
                    rsi_delta = close.diff()
                    rsi_gain = rsi_delta.where(rsi_delta > 0, 0).rolling(window=period).mean()
                    rsi_loss = (-rsi_delta.where(rsi_delta < 0, 0)).rolling(window=period).mean()
                    rs = rsi_gain / rsi_loss.replace(0, np.nan)
                    rsi_val = 100 - (100 / (1 + rs))
                    if not np.isnan(rsi_val.iloc[-1]):
                        self._ml_feature_collector.record_feature(f'rsi_{period}', float(rsi_val.iloc[-1]))

            # MACD
            if len(df) >= 26:
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - signal_line
                self._ml_feature_collector.record_feature('macd', float(macd_line.iloc[-1]))
                self._ml_feature_collector.record_feature('macd_histogram', float(macd_hist.iloc[-1]))

            # Stochastic
            if len(df) >= 14:
                low_14 = low.rolling(window=14).min()
                high_14 = high.rolling(window=14).max()
                stoch_k = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
                stoch_d = stoch_k.rolling(window=3).mean()
                if not np.isnan(stoch_k.iloc[-1]):
                    self._ml_feature_collector.record_feature('stoch_k', float(stoch_k.iloc[-1]))
                if not np.isnan(stoch_d.iloc[-1]):
                    self._ml_feature_collector.record_feature('stoch_d', float(stoch_d.iloc[-1]))
                    stoch_diff = stoch_k.iloc[-1] - stoch_d.iloc[-1]
                    self._ml_feature_collector.record_feature('stoch_diff', float(stoch_diff))

            # Volatility (5 and 20 period)
            for period in [5, 20]:
                if len(close) >= period:
                    vol_val = close.pct_change().rolling(window=period).std().iloc[-1]
                    if not np.isnan(vol_val):
                        self._ml_feature_collector.record_feature(f'volatility_{period}', float(vol_val))

            # Range features
            range_val = high.iloc[-1] - low.iloc[-1]
            range_pct = range_val / close.iloc[-1] if close.iloc[-1] != 0 else 0.0
            self._ml_feature_collector.record_feature('range_pct', float(range_pct))
            if len(df) >= 5:
                range_pct_series = (high - low) / close.replace(0, np.nan)
                range_pct_avg_5 = range_pct_series.rolling(window=5).mean().iloc[-1]
                if not np.isnan(range_pct_avg_5):
                    self._ml_feature_collector.record_feature('range_pct_avg_5', float(range_pct_avg_5))
            if len(df) >= 20:
                avg_range = (high - low).rolling(window=20).mean().iloc[-1]
                if avg_range > 0:
                    range_expansion = range_val / avg_range
                    self._ml_feature_collector.record_feature('range_expansion', float(range_expansion))

            # EMA 50
            if len(df) >= 50:
                ema_50_val = close.ewm(span=50, adjust=False).mean().iloc[-1]
                self._ml_feature_collector.record_feature('ema_50', float(ema_50_val))
                self._ml_feature_collector.record_feature('ema', float(ema_50_val))

            # Percent from high/low
            if len(df) >= 50:
                high_50 = high.rolling(window=50).max().iloc[-1]
                low_50 = low.rolling(window=50).min().iloc[-1]
                pct_from_high_50 = (close.iloc[-1] - high_50) / high_50 if high_50 != 0 else 0.0
                pct_from_low_50 = (close.iloc[-1] - low_50) / low_50 if low_50 != 0 else 0.0
                self._ml_feature_collector.record_feature('pct_from_high_50', float(pct_from_high_50))
                self._ml_feature_collector.record_feature('pct_from_low_50', float(pct_from_low_50))
            if len(df) >= 20:
                low_20 = low.rolling(window=20).min().iloc[-1]
                pct_from_low_20 = (close.iloc[-1] - low_20) / low_20 if low_20 != 0 else 0.0
                self._ml_feature_collector.record_feature('pct_from_low_20', float(pct_from_low_20))

            # Volume features
            if 'volume' in df.columns:
                vol = df['volume']
                if len(vol) >= 5:
                    volume_sma_5 = vol.rolling(window=5).mean().iloc[-1]
                    self._ml_feature_collector.record_feature('volume_sma_5', float(volume_sma_5))
                if len(vol) >= 20:
                    vol_sma_5 = vol.rolling(window=5).mean().iloc[-1]
                    vol_sma_20 = vol.rolling(window=20).mean().iloc[-1]
                    if vol_sma_20 > 0:
                        vol_ratio = vol_sma_5 / vol_sma_20
                        self._ml_feature_collector.record_feature('vol_ratio_5_20', float(vol_ratio))
                        self._ml_feature_collector.record_feature('volume_ratio', float(vol_ratio))

            # Upper wick percentage
            if 'Open' in df.columns:
                upper_wick = high.iloc[-1] - max(close.iloc[-1], df['Open'].iloc[-1])
                upper_wick_pct = upper_wick / range_val if range_val > 0 else 0.0
                self._ml_feature_collector.record_feature('upper_wick_pct', float(upper_wick_pct))

            # Returns over different periods
            if len(close) >= 3:
                returns_2 = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3] if close.iloc[-3] != 0 else 0.0
                self._ml_feature_collector.record_feature('returns_2', float(returns_2))
            if len(close) >= 61:
                returns_60 = (close.iloc[-1] - close.iloc[-61]) / close.iloc[-61] if close.iloc[-61] != 0 else 0.0
                self._ml_feature_collector.record_feature('returns_60', float(returns_60))

            # Distance z-score (dist_z) - distance from SMA as z-score
            if len(df) >= 20:
                sma_20 = close.rolling(window=20).mean()
                dist = close - sma_20
                dist_std = dist.rolling(window=20).std()
                if dist_std.iloc[-1] > 0:
                    dist_z = dist.iloc[-1] / dist_std.iloc[-1]
                    self._ml_feature_collector.record_feature('dist_z', float(dist_z))

            # Resistance/Support (simple: recent high/low)
            if len(df) >= 20:
                resistance = high.rolling(window=20).max().iloc[-1]
                support = low.rolling(window=20).min().iloc[-1]
                self._ml_feature_collector.record_feature('resistance', float(resistance))
                self._ml_feature_collector.record_feature('support', float(support))

        except Exception as e:
            logger.warning(f"{self.symbol}: Error calculating strategy features: {e}", exc_info=True)


class NotifyingFactoryAdapter(StrategyFactoryAdapter):
    """
    StrategyFactoryAdapter with notification callbacks for TradersPost/Discord.

    This is the production wrapper used by LiveWorker.
    The base class handles all trade/order notifications and callback invocation.
    """

    def __init__(self):
        super().__init__()

    # notify_order and notify_trade are handled by the base class
    # which properly captures exit snapshots and invokes callbacks
