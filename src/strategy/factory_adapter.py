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

        # Data feeds - build map of symbol -> data feed
        self.hourly_data = self.datas[0]  # Primary hourly data
        self.daily_data = None
        self._data_feeds: Dict[str, Any] = {}  # symbol -> hourly data feed
        self._daily_feeds: Dict[str, Any] = {}  # symbol -> daily data feed

        # Find and map all data feeds
        # Priority: hourly feeds > base feeds > daily feeds for feature calculation
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

            # Extract symbol from feed name (e.g., "ES_hour" -> "ES", "ES" -> "ES")
            symbol_part = name.split('_')[0].upper()
            if not symbol_part:
                continue

            # Prefer hourly feeds for feature calculation
            is_hourly = '_hour' in name.lower()
            if is_hourly or symbol_part not in self._data_feeds:
                self._data_feeds[symbol_part] = d

        # Set hourly_data to the correct feed for THIS strategy's symbol
        symbol_upper = self.symbol.upper()
        if symbol_upper in self._data_feeds:
            self.hourly_data = self._data_feeds[symbol_upper]
            logger.info(f"{self.symbol}: Using data feed '{getattr(self.hourly_data, '_name', 'unknown')}'")
        else:
            logger.warning(f"{self.symbol}: No matching hourly feed found, using datas[0]")

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
                logger.info(
                    f"{self.symbol}: BUY executed at {order.executed.price:.2f}"
                )
            else:
                self._in_position = False
                pnl = order.executed.price - self._entry_price if self._entry_price else 0
                logger.info(
                    f"{self.symbol}: SELL executed at {order.executed.price:.2f}, "
                    f"P&L: {pnl:.2f}"
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
            }

            logger.info(
                f"{self.symbol}: Trade closed, P&L: {trade.pnl:.2f}, "
                f"Commission: {trade.commission:.2f}"
            )

            # Notify portfolio coordinator
            if self.p.portfolio_coordinator:
                self.p.portfolio_coordinator.record_trade_exit(
                    symbol=self.symbol,
                    pnl=trade.pnlcomm
                )

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
            logger.info(
                f"{self.symbol}: Signal check - current={has_signal}, "
                f"last_10_bars_signals={recent_signals}, close={df['Close'].iloc[-1]:.2f}"
            )

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

        # Reserve position with portfolio coordinator
        if self.p.portfolio_coordinator:
            self.p.portfolio_coordinator.reserve_position(self.symbol)

        self._order_pending = True
        self._pending_order = self.buy(size=self.p.size)

        logger.info(f"{self.symbol}: Entry order placed (size={self.p.size})")

    def _exit_position(self, reason: str):
        """Place exit order."""
        if self._order_pending or not self._in_position:
            return

        self._order_pending = True
        self._pending_order = self.close()

        logger.info(f"{self.symbol}: Exit order placed (reason: {reason})")

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
        """Calculate features from price buffers and publish to tracker."""
        if self._ml_feature_collector is None:
            return

        # Calculate hourly returns for all symbols
        for sym in CROSS_ASSET_SYMBOLS:
            buffer = self._price_buffers.get(sym)
            if buffer is None or len(buffer) < 2:
                continue

            prices = list(buffer)

            # Hourly return: (current - previous) / previous
            hourly_return = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0

            # Publish hourly return with various naming conventions
            feature_names = [
                f"{sym} Hourly Return",
                f"{sym.lower()}_hourly_return",
            ]
            for fname in feature_names:
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

                    # Publish z-score with various naming conventions
                    zscore_names = [
                        f"{sym} Hourly Z Score",
                        f"{sym.lower()}_z_score_hour",
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

        except Exception as e:
            logger.debug(f"{self.symbol}: Error calculating strategy features: {e}")


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
