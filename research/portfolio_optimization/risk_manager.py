#!/usr/bin/env python3
"""
Portfolio Risk Manager

Real-time risk management for live trading:
- Monitors daily P&L across all strategies
- Triggers emergency shutdown if daily loss limit hit
- Resets at start of each trading day
- Logs all risk events

Usage:
    from research.portfolio_optimization.risk_manager import PortfolioRiskManager

    # Initialize
    risk_mgr = PortfolioRiskManager(
        daily_loss_limit=3000,
        max_drawdown_alert=6000
    )

    # In your trading loop
    if risk_mgr.is_shutdown:
        # Don't take new trades
        continue

    # After each trade
    risk_mgr.update_pnl(trade_pnl)

    # At start of each day
    risk_mgr.reset_daily()

Author: Rooney Capital
Date: 2025-01-22
"""

import logging
from datetime import datetime, time
from typing import Optional, Callable, List
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Real-time portfolio risk management with daily loss limits.

    Features:
    - Daily P&L tracking across all strategies
    - Automatic shutdown when daily loss limit hit
    - Max drawdown alerting
    - Risk event logging
    - Emergency position closing callbacks
    """

    def __init__(
        self,
        daily_loss_limit: float = 3000.0,
        max_drawdown_alert: float = 6000.0,
        log_dir: Optional[Path] = None,
        shutdown_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize risk manager.

        Args:
            daily_loss_limit: Maximum daily loss in dollars (triggers shutdown)
            max_drawdown_alert: Max drawdown threshold for alerting (doesn't stop trading)
            log_dir: Directory for risk event logs
            shutdown_callbacks: List of functions to call on shutdown (e.g., close_all_positions)
        """
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_alert = max_drawdown_alert
        self.shutdown_callbacks = shutdown_callbacks or []

        # State tracking
        self.daily_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.is_shutdown = False
        self.last_reset_date = None
        self.trade_count_today = 0

        # Event logging
        self.log_dir = Path(log_dir) if log_dir else Path('logs/risk_management')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.risk_events = []

        logger.info("=" * 80)
        logger.info("PORTFOLIO RISK MANAGER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Daily Loss Limit: ${self.daily_loss_limit:,.0f}")
        logger.info(f"Max Drawdown Alert: ${self.max_drawdown_alert:,.0f}")
        logger.info(f"Log Directory: {self.log_dir}")
        logger.info("=" * 80)

    def update_pnl(self, trade_pnl: float, strategy_id: str = None):
        """
        Update P&L tracking after a trade.

        Args:
            trade_pnl: Profit/loss from the trade in dollars
            strategy_id: Optional strategy identifier for logging
        """
        if self.is_shutdown:
            logger.warning(f"⚠️  Risk manager is shutdown - ignoring trade P&L update")
            return

        # Update daily P&L
        self.daily_pnl += trade_pnl
        self.cumulative_pnl += trade_pnl
        self.trade_count_today += 1

        # Update drawdown tracking
        if self.cumulative_pnl > self.peak_equity:
            self.peak_equity = self.cumulative_pnl

        self.current_drawdown = self.peak_equity - self.cumulative_pnl

        # Log trade
        trade_info = (
            f"Trade #{self.trade_count_today}: "
            f"P&L=${trade_pnl:+,.0f}, "
            f"Daily=${self.daily_pnl:+,.0f}, "
            f"Cumulative=${self.cumulative_pnl:+,.0f}, "
            f"DD=${self.current_drawdown:,.0f}"
        )
        if strategy_id:
            trade_info = f"[{strategy_id}] {trade_info}"

        logger.info(trade_info)

        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            self._trigger_shutdown(reason='daily_loss_limit')

        # Check drawdown alert (warning only, doesn't shutdown)
        if self.current_drawdown >= self.max_drawdown_alert:
            self._log_risk_event(
                event_type='drawdown_alert',
                message=f"Max drawdown alert: ${self.current_drawdown:,.0f} >= ${self.max_drawdown_alert:,.0f}",
                severity='warning'
            )

    def _trigger_shutdown(self, reason: str):
        """
        Trigger emergency shutdown.

        Args:
            reason: Reason for shutdown ('daily_loss_limit', 'manual', etc.)
        """
        if self.is_shutdown:
            return  # Already shutdown

        self.is_shutdown = True

        logger.critical("")
        logger.critical("=" * 80)
        logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")
        logger.critical("=" * 80)
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Daily P&L: ${self.daily_pnl:+,.0f}")
        logger.critical(f"Cumulative P&L: ${self.cumulative_pnl:+,.0f}")
        logger.critical(f"Current Drawdown: ${self.current_drawdown:,.0f}")
        logger.critical(f"Time: {datetime.now()}")
        logger.critical("=" * 80)

        # Log shutdown event
        self._log_risk_event(
            event_type='shutdown',
            message=f"Emergency shutdown: {reason}",
            severity='critical',
            data={
                'daily_pnl': self.daily_pnl,
                'cumulative_pnl': self.cumulative_pnl,
                'current_drawdown': self.current_drawdown,
                'trade_count': self.trade_count_today
            }
        )

        # Execute shutdown callbacks (close positions, cancel orders, etc.)
        logger.critical("Executing shutdown callbacks...")
        for i, callback in enumerate(self.shutdown_callbacks, 1):
            try:
                logger.critical(f"  Callback {i}/{len(self.shutdown_callbacks)}: {callback.__name__}")
                callback()
            except Exception as e:
                logger.error(f"  Error in shutdown callback: {e}")

        logger.critical("Shutdown complete. No new trades will be taken.")
        logger.critical("")

    def reset_daily(self, force: bool = False):
        """
        Reset daily tracking at start of new trading day.

        Args:
            force: Force reset even if already reset today
        """
        today = datetime.now().date()

        if not force and self.last_reset_date == today:
            logger.debug("Already reset today")
            return

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"DAILY RESET - {today}")
        logger.info("=" * 80)
        logger.info(f"Previous Day P&L: ${self.daily_pnl:+,.0f}")
        logger.info(f"Cumulative P&L: ${self.cumulative_pnl:+,.0f}")
        logger.info(f"Current Drawdown: ${self.current_drawdown:,.0f}")
        logger.info("=" * 80)

        # Reset daily counters
        self.daily_pnl = 0.0
        self.is_shutdown = False
        self.trade_count_today = 0
        self.last_reset_date = today

        logger.info("Daily counters reset. Ready for new trading day.")
        logger.info("")

    def manual_shutdown(self):
        """Manually trigger shutdown (e.g., end of trading day, emergency)."""
        self._trigger_shutdown(reason='manual')

    def get_status(self) -> dict:
        """
        Get current risk manager status.

        Returns:
            Dictionary with current state
        """
        return {
            'is_shutdown': self.is_shutdown,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'trade_count_today': self.trade_count_today,
            'daily_loss_limit': self.daily_loss_limit,
            'max_drawdown_alert': self.max_drawdown_alert,
            'remaining_daily_loss_budget': self.daily_loss_limit + self.daily_pnl  # How much more we can lose
        }

    def _log_risk_event(
        self,
        event_type: str,
        message: str,
        severity: str = 'info',
        data: dict = None
    ):
        """
        Log a risk event.

        Args:
            event_type: Type of event ('shutdown', 'drawdown_alert', etc.)
            message: Event description
            severity: 'info', 'warning', 'critical'
            data: Additional event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'message': message,
            'data': data or {}
        }

        self.risk_events.append(event)

        # Write to daily log file
        log_file = self.log_dir / f"risk_events_{datetime.now().date()}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def print_daily_summary(self):
        """Print end-of-day summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("DAILY RISK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Date: {datetime.now().date()}")
        logger.info(f"Daily P&L: ${self.daily_pnl:+,.0f}")
        logger.info(f"Trade Count: {self.trade_count_today}")
        logger.info(f"Cumulative P&L: ${self.cumulative_pnl:+,.0f}")
        logger.info(f"Peak Equity: ${self.peak_equity:+,.0f}")
        logger.info(f"Current Drawdown: ${self.current_drawdown:,.0f}")
        logger.info(f"Shutdown Events: {sum(1 for e in self.risk_events if e['event_type'] == 'shutdown')}")
        logger.info("=" * 80)
        logger.info("")


# Example usage
if __name__ == '__main__':
    # Demonstration of risk manager
    print("Portfolio Risk Manager - Demo")
    print("=" * 80)

    def close_all_positions():
        """Example callback - close all positions."""
        print("🔴 Closing all open positions...")
        print("🔴 Cancelling all pending orders...")

    def send_alert():
        """Example callback - send alert."""
        print("📧 Sending shutdown alert to trading desk...")

    # Initialize risk manager
    risk_mgr = PortfolioRiskManager(
        daily_loss_limit=3000,
        max_drawdown_alert=6000,
        shutdown_callbacks=[close_all_positions, send_alert]
    )

    # Simulate trading day
    print("\n📊 Simulating trades...\n")

    # Some winning trades
    risk_mgr.update_pnl(250, 'ES_21')
    risk_mgr.update_pnl(180, 'NQ_37')
    risk_mgr.update_pnl(-120, 'GC_42')

    # Simulate big losses
    print("\n⚠️  Simulating large losses...\n")
    risk_mgr.update_pnl(-1500, 'ES_21')
    risk_mgr.update_pnl(-800, 'NQ_37')

    # This should trigger shutdown (total daily loss > $3000)
    risk_mgr.update_pnl(-1200, 'CL_45')

    # Try to update after shutdown (should be blocked)
    print("\n🚫 Attempting trade after shutdown...\n")
    risk_mgr.update_pnl(500, 'ES_21')

    # Print summary
    risk_mgr.print_daily_summary()

    # Check status
    print("\nCurrent Status:")
    print(json.dumps(risk_mgr.get_status(), indent=2))
