"""
Database manager for Strategy Factory.

Handles all database operations:
- Storing backtest results
- Querying filtered results
- Tracking execution runs
- Meta-learning data
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from ..engine.backtester import BacktestResults

logger = logging.getLogger(__name__)

# Database path (default)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "results" / "strategy_factory.db"


class DatabaseManager:
    """
    Manages SQLite database for Strategy Factory results.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file (creates if doesn't exist)
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Database initialized at {self.db_path}")

    def _init_database(self):
        """Create database schema if it doesn't exist."""
        schema_path = Path(__file__).parent / "schema.sql"

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        conn = sqlite3.connect(self.db_path)
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()

    def create_run(
        self,
        phase: int,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str,
        workers: int
    ) -> str:
        """
        Create a new execution run.

        Args:
            phase: Phase number (1, 2, or 3)
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            workers: Number of workers

        Returns:
            run_id: UUID for this run
        """
        run_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO execution_runs (
                run_id, phase, symbols, start_date, end_date, timeframe, workers,
                strategies_tested, total_backtests, strategies_passed,
                runtime_seconds, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 'running')
        """, (
            run_id, phase, json.dumps(symbols), start_date, end_date,
            timeframe, workers
        ))

        conn.commit()
        conn.close()

        logger.info(f"Created execution run: {run_id} (Phase {phase})")

        return run_id

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update run status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE execution_runs
            SET status = ?, error_message = ?
            WHERE run_id = ?
        """, (status, error_message, run_id))

        conn.commit()
        conn.close()

    def save_backtest_result(
        self,
        run_id: str,
        result: BacktestResults
    ):
        """
        Save a single backtest result.

        Args:
            run_id: Execution run ID
            result: BacktestResults object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert result to dict
        data = result.to_dict()

        cursor.execute("""
            INSERT OR REPLACE INTO backtest_results (
                run_id, strategy_id, strategy_name, symbol, params,
                start_date, end_date, total_bars,
                total_trades, winning_trades, losing_trades, win_rate,
                total_pnl, total_pnl_pct, avg_pnl_per_trade,
                avg_win, avg_loss, largest_win, largest_loss,
                sharpe_ratio, max_drawdown, max_drawdown_pct, profit_factor,
                avg_bars_held, max_bars_held, min_bars_held, exit_counts
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            run_id,
            data['strategy_id'],
            data['strategy_name'],
            data['symbol'],
            data['params'],
            data['start_date'],
            data['end_date'],
            data['total_bars'],
            data['total_trades'],
            data['winning_trades'],
            data['losing_trades'],
            data['win_rate'],
            data['total_pnl'],
            data['total_pnl_pct'],
            data['avg_pnl_per_trade'],
            data['avg_win'],
            data['avg_loss'],
            data['largest_win'],
            data['largest_loss'],
            data['sharpe_ratio'],
            data['max_drawdown'],
            data['max_drawdown_pct'],
            data['profit_factor'],
            data['avg_bars_held'],
            data['max_bars_held'],
            data['min_bars_held'],
            data['exit_counts']
        ))

        conn.commit()
        conn.close()

    def save_backtest_results_batch(
        self,
        run_id: str,
        results: List[BacktestResults]
    ):
        """
        Save multiple backtest results in batch.

        Args:
            run_id: Execution run ID
            results: List of BacktestResults
        """
        saved_count = 0
        failed_count = 0

        for result in results:
            try:
                self.save_backtest_result(run_id, result)
                saved_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(
                    f"Failed to save result for {result.strategy_name} "
                    f"(params: {result.params}): {e}"
                )

        if failed_count > 0:
            logger.warning(
                f"Saved {saved_count}/{len(results)} results "
                f"({failed_count} failed)"
            )
        else:
            logger.info(f"Saved {len(results)} backtest results to run {run_id}")

    def get_results(
        self,
        run_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_trades: Optional[int] = None,
        passed_filters: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Query backtest results with filters.

        Args:
            run_id: Filter by run ID
            strategy_name: Filter by strategy name
            min_sharpe: Minimum Sharpe ratio
            min_trades: Minimum trade count
            passed_filters: Filter by passed_all_filters flag

        Returns:
            List of result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if min_sharpe is not None:
            query += " AND sharpe_ratio >= ?"
            params.append(min_sharpe)

        if min_trades is not None:
            query += " AND total_trades >= ?"
            params.append(min_trades)

        if passed_filters is not None:
            query += " AND passed_all_filters = ?"
            params.append(1 if passed_filters else 0)

        query += " ORDER BY sharpe_ratio DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]

        conn.close()

        return results

    def get_top_strategies(
        self,
        run_id: str,
        limit: int = 10,
        min_sharpe: float = 0.2,
        min_trades: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Get top strategies from a run.

        Args:
            run_id: Execution run ID
            limit: Maximum number of results
            min_sharpe: Minimum Sharpe threshold
            min_trades: Minimum trade count threshold

        Returns:
            List of top strategies
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM backtest_results
            WHERE run_id = ?
              AND sharpe_ratio >= ?
              AND total_trades >= ?
            ORDER BY sharpe_ratio DESC
            LIMIT ?
        """, (run_id, min_sharpe, min_trades, limit))

        rows = cursor.fetchall()
        results = [dict(row) for row in rows]

        conn.close()

        return results

    def mark_filtered_strategies(
        self,
        run_id: str,
        passed_ids: List[int],
        filter_name: str
    ):
        """
        Mark strategies that passed a specific filter.

        Args:
            run_id: Execution run ID
            passed_ids: List of result IDs that passed
            filter_name: Name of filter (gate1, walkforward, regime, stability, statistical)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        column_map = {
            'gate1': 'passed_gate1',
            'walkforward': 'passed_walkforward',
            'regime': 'passed_regime',
            'stability': 'passed_stability',
            'statistical': 'passed_statistical'
        }

        column = column_map.get(filter_name, filter_name)

        # Mark passed
        if passed_ids:
            placeholders = ','.join('?' * len(passed_ids))
            cursor.execute(f"""
                UPDATE backtest_results
                SET {column} = 1
                WHERE run_id = ? AND id IN ({placeholders})
            """, [run_id] + passed_ids)

        conn.commit()
        conn.close()

        logger.info(f"Marked {len(passed_ids)} strategies as passed {filter_name}")

    def finalize_run(
        self,
        run_id: str,
        strategies_tested: int,
        total_backtests: int,
        strategies_passed: int,
        runtime_seconds: float,
        report_path: Optional[str] = None,
        charts_path: Optional[str] = None
    ):
        """
        Finalize execution run with summary stats.

        Args:
            run_id: Execution run ID
            strategies_tested: Number of strategies tested
            total_backtests: Total backtest count
            strategies_passed: Number that passed all filters
            runtime_seconds: Total runtime
            report_path: Path to generated report
            charts_path: Path to generated charts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE execution_runs
            SET status = 'completed',
                strategies_tested = ?,
                total_backtests = ?,
                strategies_passed = ?,
                runtime_seconds = ?,
                report_path = ?,
                charts_path = ?
            WHERE run_id = ?
        """, (
            strategies_tested, total_backtests, strategies_passed,
            runtime_seconds, report_path, charts_path, run_id
        ))

        conn.commit()
        conn.close()

        logger.info(f"Finalized run {run_id}: {strategies_passed}/{strategies_tested} passed")


if __name__ == "__main__":
    """
    Test database manager.
    """
    import tempfile
    from ..engine.backtester import BacktestResults

    print("Testing Database Manager")
    print("=" * 80)
    print()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)

    print(f"Test database: {db_path}")
    print()

    # Initialize manager
    db = DatabaseManager(db_path)

    # Create run
    run_id = db.create_run(
        phase=1,
        symbols=['ES'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        timeframe='15min',
        workers=16
    )
    print(f"Created run: {run_id}")
    print()

    # Query results (empty)
    results = db.get_results(run_id=run_id)
    print(f"Results in run: {len(results)}")

    # Cleanup
    db_path.unlink()
    print("Test complete")
