"""SQLite database for tracking completed trades.

This module provides a simple interface for persisting trade data
and calculating performance metrics.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("/opt/pine/runtime/trades.db")


class TradesDB:
    """SQLite database manager for trade tracking."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_size REAL NOT NULL,
                    exit_time TEXT NOT NULL,
                    exit_price REAL NOT NULL,
                    exit_size REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL,
                    exit_reason TEXT,
                    ml_score REAL,
                    ibs_entry REAL,
                    ibs_exit REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, entry_time, exit_time)
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time
                ON trades(entry_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_exit_time
                ON trades(exit_time)
            """)
            conn.commit()

    def insert_trade(
        self,
        symbol: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        entry_size: float,
        exit_time: datetime,
        exit_price: float,
        exit_size: float,
        pnl: float,
        pnl_percent: Optional[float] = None,
        exit_reason: Optional[str] = None,
        ml_score: Optional[float] = None,
        ibs_entry: Optional[float] = None,
        ibs_exit: Optional[float] = None,
    ) -> int:
        """Insert a completed trade into the database.

        Args:
            symbol: Trading symbol (e.g., "ES", "NQ")
            side: "long" or "short"
            entry_time: Entry timestamp
            entry_price: Entry execution price
            entry_size: Position size (contracts)
            exit_time: Exit timestamp
            exit_price: Exit execution price
            exit_size: Exit size (should match entry_size)
            pnl: Realized profit/loss in dollars
            pnl_percent: P&L as percentage of entry value
            exit_reason: Why trade was closed (e.g., "IBS exit", "stop loss")
            ml_score: ML filter score at entry
            ibs_entry: IBS value at entry
            ibs_exit: IBS value at exit

        Returns:
            Trade ID (row id)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO trades (
                    symbol, side, entry_time, entry_price, entry_size,
                    exit_time, exit_price, exit_size, pnl, pnl_percent,
                    exit_reason, ml_score, ibs_entry, ibs_exit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    entry_time.isoformat(),
                    entry_price,
                    entry_size,
                    exit_time.isoformat(),
                    exit_price,
                    exit_size,
                    pnl,
                    pnl_percent,
                    exit_reason,
                    ml_score,
                    ibs_entry,
                    ibs_exit,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_trades(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get all trades, ordered by exit time descending.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        query = """
            SELECT * FROM trades
            ORDER BY exit_time DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_by_symbol(
        self, symbol: str, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Get trades for a specific symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        query = """
            SELECT * FROM trades
            WHERE symbol = ?
            ORDER BY exit_time DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (symbol,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_since(
        self, since: datetime, symbol: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get trades since a specific time.

        Args:
            since: Get trades with exit_time >= this datetime
            symbol: Optional symbol filter

        Returns:
            List of trade dictionaries
        """
        if symbol:
            query = """
                SELECT * FROM trades
                WHERE exit_time >= ? AND symbol = ?
                ORDER BY exit_time DESC
            """
            params = (since.isoformat(), symbol)
        else:
            query = """
                SELECT * FROM trades
                WHERE exit_time >= ?
                ORDER BY exit_time DESC
            """
            params = (since.isoformat(),)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_between(
        self,
        start: datetime,
        end: datetime,
        symbol: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get trades between two datetimes.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            symbol: Optional symbol filter

        Returns:
            List of trade dictionaries
        """
        if symbol:
            query = """
                SELECT * FROM trades
                WHERE exit_time >= ? AND exit_time <= ? AND symbol = ?
                ORDER BY exit_time ASC
            """
            params = (start.isoformat(), end.isoformat(), symbol)
        else:
            query = """
                SELECT * FROM trades
                WHERE exit_time >= ? AND exit_time <= ?
                ORDER BY exit_time ASC
            """
            params = (start.isoformat(), end.isoformat())

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_daily_pnl(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> dict[str, float]:
        """Get daily P&L aggregated by date.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping date string (YYYY-MM-DD) to total P&L
        """
        query = """
            SELECT DATE(exit_time) as date, SUM(pnl) as daily_pnl
            FROM trades
        """
        params = []

        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("DATE(exit_time) >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("DATE(exit_time) <= ?")
                params.append(end_date.isoformat())
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY DATE(exit_time) ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_summary_stats(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Get summary statistics for all trades or a specific symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dictionary with summary statistics
        """
        if symbol:
            query = "SELECT * FROM trades WHERE symbol = ?"
            params = (symbol,)
        else:
            query = "SELECT * FROM trades"
            params = ()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            trades = [dict(row) for row in cursor.fetchall()]

        if not trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]

        total_wins = sum(wins) if wins else 0.0
        total_losses = sum(losses) if losses else 0.0

        return {
            "total_trades": len(trades),
            "total_pnl": sum(t["pnl"] for t in trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "avg_win": total_wins / len(wins) if wins else 0.0,
            "avg_loss": total_losses / len(losses) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": max(losses) if losses else 0.0,
        }

    def close(self):
        """Close database connection (if needed for cleanup)."""
        # sqlite3 connections are closed automatically in context managers
        pass
