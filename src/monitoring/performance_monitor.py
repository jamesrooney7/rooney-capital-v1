"""Production Performance Monitoring System.

This module implements automated monitoring, alerting, and retraining triggers
for production trading strategies to detect regime changes and degradation early.

Key Features:
- Multi-metric monitoring (win rate, Sharpe ratios, transaction costs)
- Four-level alert system (NORMAL, WARNING, CRITICAL, SHUTDOWN)
- Automated retraining triggers with cooldown protection
- Safety shutdown triggers for catastrophic failures

Usage:
    from src.monitoring.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(symbol="ES")
    monitor.update(recent_trades, current_date)

    if monitor.alert_level == "CRITICAL":
        trigger_retraining()
    elif monitor.alert_level == "SHUTDOWN":
        disable_strategy()
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for production monitoring."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SHUTDOWN = "SHUTDOWN"


class PerformanceMonitor:
    """Production performance monitoring with automated alerts and retraining triggers.

    Tracks multiple metrics on rolling windows and triggers alerts when thresholds
    are breached. Prevents over-retraining with cooldown periods.

    Attributes:
        symbol: Trading symbol being monitored
        alert_level: Current alert level (NORMAL/WARNING/CRITICAL/SHUTDOWN)
        last_retrain_date: Date of last retraining (None if never)
        consecutive_failures: Count of consecutive retraining failures
        days_below_threshold: Days Sharpe has been below threshold
        alert_history: Recent alert messages (last 7 days)
    """

    def __init__(
        self,
        symbol: str,
        # Win rate thresholds
        win_rate_alert_threshold: float = 0.55,
        win_rate_retrain_threshold: float = 0.52,
        win_rate_window: int = 100,  # trades
        # Sharpe ratio thresholds
        sharpe_60d_alert_threshold: float = 1.0,
        sharpe_60d_retrain_threshold: float = 0.5,
        sharpe_90d_shutdown_threshold: float = 0.3,
        # Transaction cost thresholds
        txn_cost_alert_bp: float = 1.0,  # basis points
        txn_cost_window: int = 20,  # trades
        # Retraining control
        retrain_cooldown_days: int = 63,  # quarterly baseline
        consecutive_failure_shutdown: int = 3,
        sharpe_below_threshold_days: int = 45,  # for retraining trigger
        shutdown_below_threshold_days: int = 30,  # for shutdown
        # Initial state
        last_retrain_date: Optional[datetime] = None,
    ):
        """Initialize performance monitor.

        Args:
            symbol: Trading symbol (e.g., "ES", "NQ")
            win_rate_alert_threshold: Win rate alert threshold (default: 0.55)
            win_rate_retrain_threshold: Win rate retraining threshold (default: 0.52)
            win_rate_window: Number of recent trades for win rate (default: 100)
            sharpe_60d_alert_threshold: 60-day Sharpe alert threshold (default: 1.0)
            sharpe_60d_retrain_threshold: 60-day Sharpe retrain threshold (default: 0.5)
            sharpe_90d_shutdown_threshold: 90-day Sharpe shutdown threshold (default: 0.3)
            txn_cost_alert_bp: Transaction cost alert in basis points (default: 1.0)
            txn_cost_window: Number of trades for txn cost average (default: 20)
            retrain_cooldown_days: Minimum days between retrainings (default: 63)
            consecutive_failure_shutdown: Failures before shutdown (default: 3)
            sharpe_below_threshold_days: Days below threshold to trigger retrain (default: 45)
            shutdown_below_threshold_days: Days below threshold to shutdown (default: 30)
            last_retrain_date: Date of last retraining (None if never)
        """
        self.symbol = symbol

        # Thresholds
        self.win_rate_alert_threshold = win_rate_alert_threshold
        self.win_rate_retrain_threshold = win_rate_retrain_threshold
        self.win_rate_window = win_rate_window

        self.sharpe_60d_alert_threshold = sharpe_60d_alert_threshold
        self.sharpe_60d_retrain_threshold = sharpe_60d_retrain_threshold
        self.sharpe_90d_shutdown_threshold = sharpe_90d_shutdown_threshold

        self.txn_cost_alert_bp = txn_cost_alert_bp
        self.txn_cost_window = txn_cost_window

        self.retrain_cooldown_days = retrain_cooldown_days
        self.consecutive_failure_shutdown = consecutive_failure_shutdown
        self.sharpe_below_threshold_days = sharpe_below_threshold_days
        self.shutdown_below_threshold_days = shutdown_below_threshold_days

        # State
        self.last_retrain_date = last_retrain_date
        self.consecutive_failures = 0
        self.days_below_sharpe_threshold = 0
        self.alert_level = AlertLevel.NORMAL
        self.alert_history = deque(maxlen=7)  # Last 7 days
        self.manual_shutdown = False

        # Metrics cache
        self._metrics = {}

    def update(
        self,
        trades: List[Dict[str, Any]],
        daily_returns: pd.Series,
        current_date: datetime,
    ) -> Tuple[AlertLevel, List[str]]:
        """Update monitoring state with recent performance data.

        Args:
            trades: List of recent trade dictionaries with fields:
                - pnl: Trade P&L (float)
                - commission: Commission paid (float)
                - entry_time: Entry timestamp (datetime or str)
                - exit_time: Exit timestamp (datetime or str)
            daily_returns: Series of daily returns (index=date, values=returns)
                Must include at least 90 days for full monitoring
            current_date: Current date (for cooldown calculations)

        Returns:
            Tuple of (alert_level, list_of_alert_messages)
        """
        if not trades:
            logger.warning(f"{self.symbol}: No trades provided for monitoring")
            return AlertLevel.NORMAL, []

        logger.info(f"\n{'='*70}")
        logger.info(f"PERFORMANCE MONITORING - {self.symbol} - {current_date.date()}")
        logger.info(f"{'='*70}")

        alerts = []

        # Calculate metrics
        self._calculate_metrics(trades, daily_returns, current_date)

        # Check for manual shutdown
        if self.manual_shutdown:
            self.alert_level = AlertLevel.SHUTDOWN
            msg = "🛑 MANUAL SHUTDOWN OVERRIDE ACTIVE"
            alerts.append(msg)
            logger.critical(msg)
            return self.alert_level, alerts

        # Check thresholds and generate alerts
        alerts.extend(self._check_win_rate())
        alerts.extend(self._check_sharpe_ratios(current_date))
        alerts.extend(self._check_transaction_costs())
        alerts.extend(self._check_retraining_status(current_date))
        alerts.extend(self._check_shutdown_conditions())

        # Determine overall alert level
        self._determine_alert_level(alerts)

        # Store alerts in history
        if alerts:
            self.alert_history.append({
                "date": current_date,
                "alert_level": self.alert_level.value,
                "alerts": alerts,
            })

        # Log summary
        logger.info(f"\n{'='*70}")
        logger.info(f"ALERT LEVEL: {self.alert_level.value}")
        if alerts:
            logger.info(f"ALERTS ({len(alerts)}):")
            for alert in alerts:
                logger.info(f"  {alert}")
        else:
            logger.info("✅ All metrics within acceptable bounds")
        logger.info(f"{'='*70}\n")

        return self.alert_level, alerts

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        daily_returns: pd.Series,
        current_date: datetime,
    ) -> None:
        """Calculate all monitoring metrics."""
        # Win rate (last N trades)
        recent_trades = trades[-self.win_rate_window:]
        wins = sum(1 for t in recent_trades if t["pnl"] > 0)
        self._metrics["win_rate"] = wins / len(recent_trades) if recent_trades else 0.0
        self._metrics["n_trades_window"] = len(recent_trades)

        # Sharpe ratios (60-day and 90-day)
        if len(daily_returns) >= 60:
            returns_60d = daily_returns.tail(60)
            self._metrics["sharpe_60d"] = self._calculate_sharpe(returns_60d)
        else:
            self._metrics["sharpe_60d"] = None

        if len(daily_returns) >= 90:
            returns_90d = daily_returns.tail(90)
            self._metrics["sharpe_90d"] = self._calculate_sharpe(returns_90d)
        else:
            self._metrics["sharpe_90d"] = None

        # Transaction costs (last N trades, in basis points)
        recent_for_costs = trades[-self.txn_cost_window:]
        if recent_for_costs:
            # Assume each trade record has 'commission' and 'entry_price' fields
            # If not available, skip this check
            try:
                total_commission = sum(t.get("commission", 0) for t in recent_for_costs)
                total_notional = sum(
                    abs(t.get("pnl", 0) + t.get("commission", 0))
                    for t in recent_for_costs
                )
                if total_notional > 0:
                    txn_cost_pct = total_commission / total_notional
                    self._metrics["txn_cost_bp"] = txn_cost_pct * 10000  # Convert to bp
                else:
                    self._metrics["txn_cost_bp"] = 0.0
            except (KeyError, TypeError):
                self._metrics["txn_cost_bp"] = None  # Data not available
        else:
            self._metrics["txn_cost_bp"] = None

        # Days since last retrain
        if self.last_retrain_date:
            self._metrics["days_since_retrain"] = (current_date - self.last_retrain_date).days
        else:
            self._metrics["days_since_retrain"] = None

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio from returns."""
        if len(returns) == 0:
            return 0.0
        mean_return = returns.mean()
        std_return = returns.std(ddof=1)
        if std_return == 0:
            return 0.0
        return (mean_return / std_return) * np.sqrt(252)

    def _check_win_rate(self) -> List[str]:
        """Check win rate thresholds."""
        alerts = []
        win_rate = self._metrics.get("win_rate")
        n_trades = self._metrics.get("n_trades_window")

        if win_rate is None or n_trades < self.win_rate_window:
            return alerts

        if win_rate < self.win_rate_retrain_threshold:
            alerts.append(
                f"🔴 CRITICAL: Win rate {win_rate*100:.1f}% < {self.win_rate_retrain_threshold*100:.1f}% "
                f"(last {n_trades} trades) - RETRAINING RECOMMENDED"
            )
        elif win_rate < self.win_rate_alert_threshold:
            alerts.append(
                f"⚠️  WARNING: Win rate {win_rate*100:.1f}% < {self.win_rate_alert_threshold*100:.1f}% "
                f"(last {n_trades} trades)"
            )

        return alerts

    def _check_sharpe_ratios(self, current_date: datetime) -> List[str]:
        """Check Sharpe ratio thresholds."""
        alerts = []

        # 60-day Sharpe
        sharpe_60d = self._metrics.get("sharpe_60d")
        if sharpe_60d is not None:
            if sharpe_60d < self.sharpe_60d_retrain_threshold:
                self.days_below_sharpe_threshold += 1
                alerts.append(
                    f"🔴 CRITICAL: 60-day Sharpe {sharpe_60d:.3f} < {self.sharpe_60d_retrain_threshold:.3f} "
                    f"({self.days_below_sharpe_threshold} consecutive days)"
                )

                if self.days_below_sharpe_threshold >= self.sharpe_below_threshold_days:
                    alerts.append(
                        f"🔥 RETRAINING TRIGGER: Sharpe below threshold for {self.days_below_sharpe_threshold} days "
                        f"(>= {self.sharpe_below_threshold_days} day threshold)"
                    )
            elif sharpe_60d < self.sharpe_60d_alert_threshold:
                self.days_below_sharpe_threshold = 0  # Reset counter
                alerts.append(
                    f"⚠️  WARNING: 60-day Sharpe {sharpe_60d:.3f} < {self.sharpe_60d_alert_threshold:.3f}"
                )
            else:
                self.days_below_sharpe_threshold = 0  # Reset counter

        # 90-day Sharpe (shutdown check)
        sharpe_90d = self._metrics.get("sharpe_90d")
        if sharpe_90d is not None:
            if sharpe_90d < self.sharpe_90d_shutdown_threshold:
                alerts.append(
                    f"🛑 SHUTDOWN RISK: 90-day Sharpe {sharpe_90d:.3f} < {self.sharpe_90d_shutdown_threshold:.3f}"
                )

        return alerts

    def _check_transaction_costs(self) -> List[str]:
        """Check transaction cost thresholds."""
        alerts = []
        txn_cost_bp = self._metrics.get("txn_cost_bp")

        if txn_cost_bp is not None and txn_cost_bp > self.txn_cost_alert_bp:
            alerts.append(
                f"⚠️  WARNING: Transaction costs {txn_cost_bp:.2f}bp > {self.txn_cost_alert_bp:.2f}bp "
                f"(last {self.txn_cost_window} trades) - Check slippage/spreads"
            )

        return alerts

    def _check_retraining_status(self, current_date: datetime) -> List[str]:
        """Check retraining eligibility and cooldown."""
        alerts = []

        days_since_retrain = self._metrics.get("days_since_retrain")

        if days_since_retrain is not None:
            if days_since_retrain < self.retrain_cooldown_days:
                days_remaining = self.retrain_cooldown_days - days_since_retrain
                alerts.append(
                    f"ℹ️  INFO: Retraining cooldown active ({days_remaining} days remaining)"
                )
            else:
                alerts.append(
                    f"✅ INFO: Retraining eligible (last retrain {days_since_retrain} days ago)"
                )

        if self.consecutive_failures > 0:
            alerts.append(
                f"⚠️  WARNING: {self.consecutive_failures} consecutive retraining failure(s)"
            )

        return alerts

    def _check_shutdown_conditions(self) -> List[str]:
        """Check conditions that should trigger strategy shutdown."""
        alerts = []

        # Condition 1: 90-day Sharpe below threshold for too long
        sharpe_90d = self._metrics.get("sharpe_90d")
        if sharpe_90d is not None and sharpe_90d < self.sharpe_90d_shutdown_threshold:
            if self.days_below_sharpe_threshold >= self.shutdown_below_threshold_days:
                alerts.append(
                    f"🛑 SHUTDOWN TRIGGER: 90-day Sharpe below {self.sharpe_90d_shutdown_threshold:.3f} "
                    f"for {self.days_below_sharpe_threshold} days (>= {self.shutdown_below_threshold_days})"
                )

        # Condition 2: Too many consecutive retraining failures
        if self.consecutive_failures >= self.consecutive_failure_shutdown:
            alerts.append(
                f"🛑 SHUTDOWN TRIGGER: {self.consecutive_failures} consecutive retraining failures "
                f"(>= {self.consecutive_failure_shutdown})"
            )

        return alerts

    def _determine_alert_level(self, alerts: List[str]) -> None:
        """Determine overall alert level from alert messages."""
        if any("🛑 SHUTDOWN" in a for a in alerts):
            self.alert_level = AlertLevel.SHUTDOWN
        elif any("🔴 CRITICAL" in a or "🔥 RETRAINING TRIGGER" in a for a in alerts):
            self.alert_level = AlertLevel.CRITICAL
        elif any("⚠️  WARNING" in a for a in alerts):
            self.alert_level = AlertLevel.WARNING
        else:
            self.alert_level = AlertLevel.NORMAL

    def record_retrain_success(self, retrain_date: datetime) -> None:
        """Record successful retraining event.

        Args:
            retrain_date: Date of successful retraining
        """
        self.last_retrain_date = retrain_date
        self.consecutive_failures = 0
        self.days_below_sharpe_threshold = 0
        logger.info(f"✅ Retraining success recorded for {self.symbol} on {retrain_date.date()}")

    def record_retrain_failure(self) -> None:
        """Record failed retraining event."""
        self.consecutive_failures += 1
        logger.warning(
            f"❌ Retraining failure recorded for {self.symbol} "
            f"(consecutive failures: {self.consecutive_failures})"
        )

    def enable_manual_shutdown(self) -> None:
        """Manually shutdown strategy monitoring."""
        self.manual_shutdown = True
        logger.critical(f"🛑 Manual shutdown enabled for {self.symbol}")

    def disable_manual_shutdown(self) -> None:
        """Re-enable strategy after manual shutdown."""
        self.manual_shutdown = False
        logger.info(f"✅ Manual shutdown disabled for {self.symbol}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary.

        Returns:
            Dictionary with current monitoring metrics
        """
        return {
            "symbol": self.symbol,
            "alert_level": self.alert_level.value,
            "metrics": self._metrics.copy(),
            "state": {
                "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                "consecutive_failures": self.consecutive_failures,
                "days_below_sharpe_threshold": self.days_below_sharpe_threshold,
                "manual_shutdown": self.manual_shutdown,
            },
            "thresholds": {
                "win_rate_alert": self.win_rate_alert_threshold,
                "win_rate_retrain": self.win_rate_retrain_threshold,
                "sharpe_60d_alert": self.sharpe_60d_alert_threshold,
                "sharpe_60d_retrain": self.sharpe_60d_retrain_threshold,
                "sharpe_90d_shutdown": self.sharpe_90d_shutdown_threshold,
                "txn_cost_alert_bp": self.txn_cost_alert_bp,
                "retrain_cooldown_days": self.retrain_cooldown_days,
            },
        }

    def should_retrain(self, current_date: datetime) -> bool:
        """Check if retraining should be triggered.

        Args:
            current_date: Current date

        Returns:
            True if retraining should be triggered
        """
        # Check cooldown period
        if self.last_retrain_date:
            days_since = (current_date - self.last_retrain_date).days
            if days_since < self.retrain_cooldown_days:
                return False

        # Check if CRITICAL alert level
        return self.alert_level == AlertLevel.CRITICAL

    def should_shutdown(self) -> bool:
        """Check if strategy should be shut down.

        Returns:
            True if strategy should be disabled
        """
        return self.alert_level == AlertLevel.SHUTDOWN or self.manual_shutdown


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Create monitor
    monitor = PerformanceMonitor(symbol="ES")

    # Simulate 100 trades with declining performance
    trades = []
    for i in range(100):
        # Simulate declining win rate (starts 60%, ends 45%)
        win_prob = 0.60 - (i / 100) * 0.15
        is_win = np.random.random() < win_prob
        pnl = 100.0 if is_win else -80.0
        commission = 2.5

        trades.append({
            "pnl": pnl,
            "commission": commission,
            "entry_time": datetime.now() - timedelta(days=100-i),
            "exit_time": datetime.now() - timedelta(days=100-i) + timedelta(hours=6),
        })

    # Simulate daily returns (90 days, declining Sharpe)
    dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
    returns = []
    for i in range(90):
        # Declining Sharpe: good at start, poor at end
        mean_return = 0.002 - (i / 90) * 0.003  # 0.2% -> -0.1%
        returns.append(np.random.normal(mean_return, 0.01))

    daily_returns = pd.Series(returns, index=dates)

    # Update monitor
    alert_level, alerts = monitor.update(trades, daily_returns, datetime.now())

    print(f"\n{'='*70}")
    print("EXAMPLE MONITORING OUTPUT")
    print(f"{'='*70}")
    print(f"Final Alert Level: {alert_level.value}")
    print(f"Should Retrain: {monitor.should_retrain(datetime.now())}")
    print(f"Should Shutdown: {monitor.should_shutdown()}")
    print(f"\nMetrics Summary:")
    summary = monitor.get_metrics_summary()
    for key, value in summary["metrics"].items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    print(f"{'='*70}\n")
