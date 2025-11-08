"""
Base strategy interface for multi-alpha architecture.

All strategies must inherit from BaseStrategy and implement the abstract methods.
This ensures consistent behavior across all strategies while allowing
strategy-specific logic to be pluggable.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import backtrader as bt

logger = logging.getLogger(__name__)


class BaseStrategy(bt.Strategy, ABC):
    """
    Abstract base class for all trading strategies.

    Responsibilities:
    - Define common interface for signal generation
    - Handle position tracking and lifecycle
    - Integrate with portfolio coordinator
    - Integrate with ML models for signal filtering
    - Provide order/trade notification hooks
    - Track performance metrics

    Subclasses must implement:
    - should_enter_long(symbol)
    - should_enter_short(symbol)
    - should_exit(symbol)
    """

    # Common parameters shared across all strategies
    params = dict(
        # Strategy identification
        strategy_name="",  # Name of the strategy (e.g., "ibs", "breakout")
        symbol="ES",  # Primary trading symbol
        size=1,  # Position size (number of contracts)

        # Portfolio coordination
        portfolio_coordinator=None,  # Shared position/risk manager

        # ML integration
        ml_model=None,  # Trained ML model for signal filtering
        ml_features=None,  # List of feature names the model expects
        ml_threshold=None,  # Probability threshold for ML veto
        ml_feature_collector=None,  # Optional feature collection for retraining

        # Position management
        max_bars_in_trade=100,  # Max time in trade (bars)

        # Callbacks
        on_order_callback=None,  # Called when order is placed/executed
        on_trade_callback=None,  # Called when trade completes
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize base strategy.

        Sets up common infrastructure:
        - Position tracking
        - Feed references
        - ML model integration
        - Performance tracking
        """
        super().__init__(*args, **kwargs)

        # Strategy identification
        self.strategy_name = self.p.strategy_name or self.__class__.__name__
        self.symbol = self.p.symbol

        # Position tracking
        self.position_size = 0  # Current position size
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.bars_in_trade = 0

        # Order tracking
        self.pending_order = None
        self.pending_exit = False

        # ML model
        self.ml_model = self.p.ml_model
        self.ml_features = self.p.ml_features or []
        self.ml_threshold = self.p.ml_threshold or 0.5
        self.ml_feature_collector = self.p.ml_feature_collector

        # Portfolio coordinator
        self.portfolio_coordinator = self.p.portfolio_coordinator

        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

        # Callbacks
        self.on_order_callback = self.p.on_order_callback
        self.on_trade_callback = self.p.on_trade_callback

        # Data feeds (will be populated in setup_feeds if overridden)
        self.data_feeds: Dict[str, Any] = {}

        logger.info(
            "Initialized %s strategy for %s (size=%d)",
            self.strategy_name,
            self.symbol,
            self.p.size
        )

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def should_enter_long(self) -> bool:
        """
        Check if conditions are met for a LONG entry.

        Returns:
            True if should enter long, False otherwise
        """
        pass

    @abstractmethod
    def should_enter_short(self) -> bool:
        """
        Check if conditions are met for a SHORT entry.

        Returns:
            True if should enter short, False otherwise
        """
        pass

    @abstractmethod
    def should_exit(self) -> bool:
        """
        Check if current position should be exited.

        Returns:
            True if should exit, False otherwise
        """
        pass

    # -------------------------------------------------------------------------
    # Optional Methods - Can be overridden by subclasses
    # -------------------------------------------------------------------------

    def calculate_position_size(self) -> int:
        """
        Calculate position size for next trade.

        Default: Use configured size parameter.
        Override for dynamic sizing based on volatility, risk, etc.

        Returns:
            Number of contracts to trade
        """
        return self.p.size

    def get_features_snapshot(self) -> Dict[str, Any]:
        """
        Get current feature values for ML model or logging.

        Default: Returns empty dict.
        Override to collect strategy-specific features.

        Returns:
            Dict of feature_name -> value
        """
        return {}

    # -------------------------------------------------------------------------
    # Core Strategy Logic (called by Backtrader)
    # -------------------------------------------------------------------------

    def next(self):
        """
        Main strategy logic called on each bar.

        Flow:
        1. Update position state
        2. Check exit conditions if in position
        3. Check entry conditions if flat
        4. Apply ML veto if configured
        5. Execute orders
        """
        # Update bars in trade counter
        if self.position:
            self.bars_in_trade += 1
        else:
            self.bars_in_trade = 0

        # Skip if we have pending orders
        if self.pending_order:
            return

        # Check for exit if in position
        if self.position:
            if self._check_exit_conditions():
                self._exit_position()
                return

        # Check for entry if flat
        if not self.position:
            self._check_entry_conditions()

    def _check_exit_conditions(self) -> bool:
        """
        Check all exit conditions.

        Returns:
            True if should exit
        """
        # Time-based exit
        if self.bars_in_trade >= self.p.max_bars_in_trade:
            logger.info(
                "%s: Time-based exit triggered (bars_in_trade=%d >= max=%d)",
                self.symbol,
                self.bars_in_trade,
                self.p.max_bars_in_trade
            )
            return True

        # Strategy-specific exit logic
        if self.should_exit():
            logger.info("%s: Strategy exit signal triggered", self.symbol)
            return True

        return False

    def _check_entry_conditions(self):
        """Check entry conditions and place orders if signals present."""
        # Check long signal
        if self.should_enter_long():
            if self._ml_veto_check('long'):
                return  # Vetoed by ML

            if not self._portfolio_check():
                return  # Blocked by portfolio coordinator

            # Enter long
            size = self.calculate_position_size()
            self._enter_position(size, 'long')

        # Check short signal
        elif self.should_enter_short():
            if self._ml_veto_check('short'):
                return  # Vetoed by ML

            if not self._portfolio_check():
                return  # Blocked by portfolio coordinator

            # Enter short
            size = self.calculate_position_size()
            self._enter_position(-size, 'short')

    def _ml_veto_check(self, direction: str) -> bool:
        """
        Apply ML model veto to trading signal.

        Args:
            direction: 'long' or 'short'

        Returns:
            True if trade is vetoed, False if allowed
        """
        if not self.ml_model or not self.ml_features:
            return False  # No ML model, allow trade

        try:
            # Get feature snapshot
            features = self.get_features_snapshot()

            # Extract required features in correct order
            feature_values = []
            for feat_name in self.ml_features:
                if feat_name not in features:
                    logger.warning(
                        "%s: Missing ML feature '%s', allowing trade",
                        self.symbol,
                        feat_name
                    )
                    return False
                feature_values.append(features[feat_name])

            # Reshape for model prediction
            import numpy as np
            X = np.array([feature_values])

            # Get prediction probability
            proba = self.ml_model.predict_proba(X)[0][1]  # P(win)

            # Check threshold
            if proba < self.ml_threshold:
                logger.info(
                    "%s: ML veto applied (proba=%.3f < threshold=%.3f)",
                    self.symbol,
                    proba,
                    self.ml_threshold
                )
                return True  # Vetoed

            logger.info(
                "%s: ML filter passed (proba=%.3f >= threshold=%.3f)",
                self.symbol,
                proba,
                self.ml_threshold
            )
            return False  # Allowed

        except Exception as e:
            logger.error(
                "%s: ML veto check failed: %s (allowing trade)",
                self.symbol,
                e
            )
            return False  # On error, allow trade

    def _portfolio_check(self) -> bool:
        """
        Check with portfolio coordinator if we can open a position.

        Returns:
            True if allowed, False if blocked
        """
        if not self.portfolio_coordinator:
            return True  # No coordinator, allow

        can_open, reason = self.portfolio_coordinator.can_open_position(self.symbol)

        if not can_open:
            logger.info(
                "%s: Portfolio coordinator blocked trade: %s",
                self.symbol,
                reason
            )

        return can_open

    def _enter_position(self, size: int, direction: str):
        """
        Enter a position.

        Args:
            size: Position size (positive for long, negative for short)
            direction: 'long' or 'short' (for logging)
        """
        logger.info(
            "%s: Entering %s position (size=%d) @ %.2f",
            self.symbol,
            direction,
            abs(size),
            self.data.close[0]
        )

        # Place order
        if size > 0:
            self.pending_order = self.buy(size=size)
        else:
            self.pending_order = self.sell(size=abs(size))

        # Record entry signal snapshot for ML training
        if self.ml_feature_collector:
            features = self.get_features_snapshot()
            self.ml_feature_collector.record_entry(
                symbol=self.symbol,
                timestamp=self.data.datetime.datetime(),
                direction=direction,
                features=features
            )

    def _exit_position(self):
        """Exit current position."""
        logger.info(
            "%s: Exiting position (size=%d) @ %.2f",
            self.symbol,
            self.position.size,
            self.data.close[0]
        )

        # Close position
        self.pending_order = self.close()
        self.pending_exit = True

    # -------------------------------------------------------------------------
    # Backtrader Notification Handlers
    # -------------------------------------------------------------------------

    def notify_order(self, order):
        """
        Called when order status changes.

        Args:
            order: Backtrader order object
        """
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order submitted/accepted, waiting

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(
                    "%s: BUY executed @ %.2f (size=%d)",
                    self.symbol,
                    order.executed.price,
                    order.executed.size
                )
                self.entry_price = order.executed.price
                self.entry_time = self.data.datetime.datetime()

                # Register with portfolio coordinator
                if self.portfolio_coordinator:
                    self.portfolio_coordinator.register_position_opened(
                        symbol=self.symbol,
                        size=order.executed.size,
                        entry_price=order.executed.price
                    )

            elif order.issell():
                logger.info(
                    "%s: SELL executed @ %.2f (size=%d)",
                    self.symbol,
                    order.executed.price,
                    order.executed.size
                )

            # Clear pending order
            self.pending_order = None

            # Call callback if registered
            if self.on_order_callback:
                self.on_order_callback(self.symbol, order)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(
                "%s: Order %s (status=%s)",
                self.symbol,
                "canceled" if order.status == order.Canceled else "rejected",
                order.status
            )
            self.pending_order = None

    def notify_trade(self, trade):
        """
        Called when a trade is closed.

        Args:
            trade: Backtrader trade object
        """
        if not trade.isclosed:
            return

        # Calculate P&L
        pnl = trade.pnl
        pnl_pct = (pnl / (abs(trade.value) or 1)) * 100

        # Update statistics
        self.trade_count += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.win_count += 1
            outcome = "WIN"
        else:
            self.loss_count += 1
            outcome = "LOSS"

        logger.info(
            "%s: Trade closed - %s | P&L: $%.2f (%.2f%%) | Total: $%.2f | W/L: %d/%d",
            self.symbol,
            outcome,
            pnl,
            pnl_pct,
            self.total_pnl,
            self.win_count,
            self.loss_count
        )

        # Register with portfolio coordinator
        if self.portfolio_coordinator:
            self.portfolio_coordinator.register_position_closed(
                symbol=self.symbol,
                pnl=pnl,
                exit_time=self.data.datetime.datetime()
            )

        # Record exit for ML training
        if self.ml_feature_collector:
            features = self.get_features_snapshot()
            self.ml_feature_collector.record_exit(
                symbol=self.symbol,
                timestamp=self.data.datetime.datetime(),
                pnl=pnl,
                outcome='win' if pnl > 0 else 'loss',
                features=features
            )

        # Reset position tracking
        self.entry_price = None
        self.entry_time = None
        self.bars_in_trade = 0
        self.pending_exit = False

        # Call callback if registered
        if self.on_trade_callback:
            self.on_trade_callback(self.symbol, trade)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.

        Returns:
            Dict with performance metrics
        """
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0

        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_position': self.position.size if self.position else 0,
            'bars_in_trade': self.bars_in_trade,
        }

    def __repr__(self):
        return f"<{self.strategy_name} symbol={self.symbol} trades={self.trade_count}>"
