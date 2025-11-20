"""
Database management for Strategy Factory.

SQLite database for storing:
- Backtest results (Phase 1)
- Multi-symbol validation (Phase 2)
- ML integration results (Phase 3)
- Meta-learning insights
- Execution run tracking
"""

from .manager import DatabaseManager

__all__ = ['DatabaseManager']
