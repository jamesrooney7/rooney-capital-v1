"""Production monitoring system for trading strategies.

This package provides automated performance monitoring, alerting, and retraining
triggers to detect regime changes and performance degradation early.

Main components:
- PerformanceMonitor: Core monitoring class with multi-metric tracking
- AlertLevel: Enum for alert severity (NORMAL/WARNING/CRITICAL/SHUTDOWN)

Key Features:
- Win rate tracking (30-day rolling)
- Sharpe ratio tracking (60-day and 90-day)
- Transaction cost monitoring
- Automated retraining triggers with cooldown
- Safety shutdown triggers
"""

from .performance_monitor import AlertLevel, PerformanceMonitor

__all__ = ["PerformanceMonitor", "AlertLevel"]
