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

# Import Strategy Factory base and strategies
import sys
from pathlib import Path

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

        # Data feeds
        self.hourly_data = self.datas[0]  # Primary hourly data
        self.daily_data = None

        # Find daily data feed if available
        for d in self.datas:
            name = getattr(d, '_name', '')
            if '_daily' in name.lower() or 'daily' in name.lower():
                self.daily_data = d
                break

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
        if trade.isclosed:
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

            # Notify callbacks
            for callback in self.p.trade_callbacks:
                try:
                    callback(self, trade)
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

        # Wait for warmup
        if self._bars_seen < self.p.warmup_bars:
            return

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
        if not entry_signals.iloc[current_idx]:
            return

        # Check portfolio coordinator for position limits
        if self.p.portfolio_coordinator:
            if not self.p.portfolio_coordinator.can_open_position(self.symbol):
                logger.debug(f"{self.symbol}: Position blocked by portfolio coordinator")
                return

        # ML filtering
        if self.p.ml_model is not None:
            ml_score = self._get_ml_score(df, current_idx)
            self._ml_last_score = ml_score

            if ml_score is None or ml_score < self.p.ml_threshold:
                logger.debug(
                    f"{self.symbol}: Entry blocked by ML filter "
                    f"(score={ml_score:.3f}, threshold={self.p.ml_threshold})"
                )
                return

            logger.info(
                f"{self.symbol}: ML approved entry (score={ml_score:.3f})"
            )

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
        self._pending_order = self.buy()

        logger.info(f"{self.symbol}: Entry order placed")

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
            if self.p.feature_tracker:
                features = self.p.feature_tracker.get_features(
                    self.symbol, self.p.ml_features
                )
            else:
                # Extract features from DataFrame
                current_row = df.iloc[current_idx]
                features = {}
                for f in self.p.ml_features:
                    if f in current_row:
                        features[f] = current_row[f]

            if not features:
                logger.warning(f"{self.symbol}: No ML features available")
                return None

            # Build feature vector in correct order
            X = pd.DataFrame([features])[self.p.ml_features]

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


class NotifyingFactoryAdapter(StrategyFactoryAdapter):
    """
    StrategyFactoryAdapter with notification callbacks for TradersPost/Discord.

    This is the production wrapper used by LiveWorker.
    """

    params = StrategyFactoryAdapter.params + (
        ('queue_manager', None),  # QueueFanout for contract symbol lookup
    )

    def __init__(self):
        super().__init__()
        self._exit_snapshot = None

    def notify_order(self, order):
        """Enhanced order notification with TradersPost support."""
        super().notify_order(order)

        # Additional processing for TradersPost can be added here
        # The order_callbacks handle the actual webhook posting

    def notify_trade(self, trade):
        """Enhanced trade notification with snapshot."""
        # Capture exit snapshot before parent processes
        if trade.isclosed:
            self._exit_snapshot = {
                'exit_reason': 'strategy_exit',
                'ibs_value': None,  # Could be populated from strategy
                'size': trade.size,
                'price': trade.price,
                'dt': self.hourly_data.datetime.datetime(0),
            }

        super().notify_trade(trade)
