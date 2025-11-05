"""Portfolio-level position and risk management coordinator.

This module provides centralized coordination for portfolio-wide constraints:
- Max positions limit across all symbols
- Daily portfolio stop loss with emergency exits
- Position tracking and P&L aggregation

Used by LiveWorker to enforce portfolio-level rules that individual
IbsStrategy instances cannot enforce alone.
"""

import logging
import threading
from datetime import date, datetime
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str
    size: float
    entry_time: datetime
    entry_price: Optional[float] = None


class PortfolioCoordinator:
    """Coordinates portfolio-wide position limits and risk management.

    Thread-safe coordinator that tracks positions across multiple strategy
    instances and enforces portfolio-level constraints.

    Attributes:
        max_positions: Maximum number of concurrent positions across all symbols
        daily_stop_loss: Daily loss limit in dollars (positive number)
        open_positions: Currently open positions {symbol: PositionInfo}
        daily_pnl: Cumulative P&L for current trading day
        stopped_out: Whether daily stop loss has been triggered
        current_day: Current trading day for P&L tracking
    """

    def __init__(
        self,
        max_positions: int,
        daily_stop_loss: float = 2500.0,
        emergency_exit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        daily_summary_callback: Optional[Callable[[], None]] = None
    ):
        """Initialize portfolio coordinator.

        Args:
            max_positions: Maximum concurrent positions (e.g., 4)
            daily_stop_loss: Daily loss limit in dollars (e.g., 2500.0)
            emergency_exit_callback: Optional callback when stop loss hit.
                Signature: callback(reason: str, context: dict)
            daily_summary_callback: Optional callback when day rolls over.
                Called before resetting daily state for new day.
        """
        self.max_positions = max_positions
        self.daily_stop_loss = abs(daily_stop_loss)  # Ensure positive
        self.emergency_exit_callback = emergency_exit_callback
        self.daily_summary_callback = daily_summary_callback

        # Thread-safe state
        self._lock = threading.RLock()

        # Position tracking
        self.open_positions: Dict[str, PositionInfo] = {}
        self.pending_positions: Dict[str, datetime] = {}  # Reserved but not yet filled

        # Daily P&L tracking
        self.daily_pnl: float = 0.0
        self.stopped_out: bool = False
        self.current_day: Optional[date] = None

        # Statistics
        self.total_entries_requested: int = 0
        self.total_entries_allowed: int = 0
        self.total_entries_blocked: int = 0
        self.total_exits: int = 0
        self.stop_loss_triggers: int = 0

        logger.info(
            f"PortfolioCoordinator initialized: max_positions={max_positions}, "
            f"daily_stop_loss=${daily_stop_loss:,.0f}"
        )

    def reset_daily_state(self, trading_day: date) -> None:
        """Reset daily P&L and stop-out status for a new trading day.

        Args:
            trading_day: The new trading day
        """
        with self._lock:
            self.current_day = trading_day
            self.daily_pnl = 0.0
            self.stopped_out = False
            logger.info(f"Daily state reset for {trading_day}")

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """Check if a symbol can open a new position and reserve slot atomically.

        Args:
            symbol: Symbol requesting to open position

        Returns:
            (allowed, reason) tuple:
                - allowed: True if position can be opened (slot reserved)
                - reason: Explanation string
        """
        with self._lock:
            self.total_entries_requested += 1

            # Check if already have position or pending order in this symbol
            if symbol in self.open_positions:
                return (False, f"{symbol} already has open position")

            if symbol in self.pending_positions:
                return (False, f"{symbol} already has pending order")

            # Check if stopped out for the day
            if self.stopped_out:
                self.total_entries_blocked += 1
                return (False, "Portfolio stopped out for the day")

            # Check max positions (count both open and pending)
            total_positions = len(self.open_positions) + len(self.pending_positions)
            if total_positions >= self.max_positions:
                self.total_entries_blocked += 1
                open_symbols = ', '.join(self.open_positions.keys())
                pending_symbols = ', '.join(self.pending_positions.keys())
                symbols_str = f"Open: {open_symbols}" if open_symbols else ""
                if pending_symbols:
                    symbols_str += f" | Pending: {pending_symbols}" if symbols_str else f"Pending: {pending_symbols}"
                return (
                    False,
                    f"Max positions ({self.max_positions}) reached. {symbols_str}"
                )

            # All checks passed - RESERVE the slot immediately
            self.pending_positions[symbol] = datetime.now()
            self.total_entries_allowed += 1
            logger.debug(
                f"Position slot reserved: {symbol} | "
                f"Total: {len(self.open_positions)} open + {len(self.pending_positions)} pending = {total_positions + 1}/{self.max_positions}"
            )
            return (True, "Entry allowed")

    def register_position_opened(
        self,
        symbol: str,
        size: float,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None
    ) -> None:
        """Register that a position has been opened (convert pending to open).

        Args:
            symbol: Symbol of opened position
            size: Position size (number of contracts)
            entry_price: Entry price (optional)
            entry_time: Entry timestamp (defaults to now)
        """
        with self._lock:
            # Remove from pending (if it was reserved)
            self.pending_positions.pop(symbol, None)

            if entry_time is None:
                entry_time = datetime.now()

            position = PositionInfo(
                symbol=symbol,
                size=size,
                entry_time=entry_time,
                entry_price=entry_price
            )

            self.open_positions[symbol] = position

            logger.info(
                f"Position opened: {symbol} | Size: {size} | "
                f"Open positions: {len(self.open_positions)}/{self.max_positions}"
            )

    def register_position_closed(
        self,
        symbol: str,
        pnl: float,
        exit_time: Optional[datetime] = None
    ) -> None:
        """Register that a position has been closed and check for stop loss.

        Args:
            symbol: Symbol of closed position
            pnl: Profit/loss in dollars (can be negative)
            exit_time: Exit timestamp (defaults to now)
        """
        with self._lock:
            if exit_time is None:
                exit_time = datetime.now()

            # Check if new trading day
            today = exit_time.date()
            if self.current_day is not None and today != self.current_day:
                # New day detected - trigger daily summary for previous day
                logger.info(f"Day rollover detected: {self.current_day} → {today}")
                if self.daily_summary_callback:
                    try:
                        self.daily_summary_callback()
                    except Exception:
                        logger.exception("Daily summary callback failed")
                self.reset_daily_state(today)
            elif self.current_day is None:
                # First day - just initialize
                self.reset_daily_state(today)

            # Remove from open positions
            position = self.open_positions.pop(symbol, None)
            self.total_exits += 1

            # Update daily P&L
            self.daily_pnl += pnl

            logger.info(
                f"Position closed: {symbol} | P&L: ${pnl:,.2f} | "
                f"Daily P&L: ${self.daily_pnl:,.2f} | "
                f"Open positions: {len(self.open_positions)}/{self.max_positions}"
            )

            # Check for daily stop loss
            if self.daily_pnl <= -self.daily_stop_loss:
                self._trigger_stop_loss(exit_time)

    def release_pending_position(self, symbol: str) -> None:
        """Release a pending position slot if order fails or is canceled.

        Args:
            symbol: Symbol to release from pending
        """
        with self._lock:
            if symbol in self.pending_positions:
                del self.pending_positions[symbol]
                logger.debug(f"Released pending position slot for {symbol}")

    def _trigger_stop_loss(self, trigger_time: datetime) -> None:
        """Trigger portfolio-wide stop loss (called with lock held).

        Args:
            trigger_time: When stop loss was triggered
        """
        if self.stopped_out:
            return  # Already stopped out

        self.stopped_out = True
        self.stop_loss_triggers += 1

        logger.critical(
            f"🚨 PORTFOLIO STOP LOSS TRIGGERED 🚨 | "
            f"Daily P&L: ${self.daily_pnl:,.2f} | "
            f"Limit: ${self.daily_stop_loss:,.2f} | "
            f"Open positions: {len(self.open_positions)}"
        )

        # Call emergency exit callback if configured
        if self.emergency_exit_callback:
            context = {
                'daily_pnl': self.daily_pnl,
                'stop_loss_limit': self.daily_stop_loss,
                'open_positions': list(self.open_positions.keys()),
                'trigger_time': trigger_time.isoformat(),
                'trading_day': self.current_day.isoformat() if self.current_day else None
            }

            try:
                self.emergency_exit_callback("daily_stop_loss", context)
            except Exception as e:
                logger.exception(f"Emergency exit callback failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current portfolio coordinator status.

        Returns:
            Dictionary with current state and statistics
        """
        with self._lock:
            return {
                'max_positions': self.max_positions,
                'daily_stop_loss': self.daily_stop_loss,
                'open_positions_count': len(self.open_positions),
                'open_positions': list(self.open_positions.keys()),
                'daily_pnl': self.daily_pnl,
                'stopped_out': self.stopped_out,
                'current_day': self.current_day.isoformat() if self.current_day else None,
                'stats': {
                    'total_entries_requested': self.total_entries_requested,
                    'total_entries_allowed': self.total_entries_allowed,
                    'total_entries_blocked': self.total_entries_blocked,
                    'total_exits': self.total_exits,
                    'stop_loss_triggers': self.stop_loss_triggers,
                    'block_rate': (
                        self.total_entries_blocked / self.total_entries_requested
                        if self.total_entries_requested > 0 else 0.0
                    )
                }
            }

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"PortfolioCoordinator(max_positions={self.max_positions}, "
                f"open={len(self.open_positions)}, daily_pnl=${self.daily_pnl:.2f}, "
                f"stopped_out={self.stopped_out})"
            )
