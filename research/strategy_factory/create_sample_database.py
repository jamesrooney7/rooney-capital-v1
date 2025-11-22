#!/usr/bin/env python3
"""
Create Sample Strategy Factory Database

Generates a realistic SQLite database with Phase 1 backtest results for testing
the extract_winners.py script.

Creates:
- Execution runs for multiple instruments
- Backtest results for multiple strategies
- Realistic performance metrics (Sharpe, win rate, trades, etc.)

Author: Rooney Capital
Date: 2025-01-22
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import numpy as np

np.random.seed(42)


def create_database_schema(db_path: Path):
    """Create the Strategy Factory database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execution runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_runs (
            run_id TEXT PRIMARY KEY,
            phase INTEGER,
            status TEXT,
            symbols TEXT,
            strategies_passed INTEGER,
            total_backtests INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Backtest results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            strategy_id INTEGER,
            strategy_name TEXT,
            params TEXT,
            sharpe_ratio REAL,
            total_trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown_pct REAL,
            total_pnl_pct REAL,
            avg_bars_held REAL,
            FOREIGN KEY (run_id) REFERENCES execution_runs(run_id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Created database schema: {db_path}")


def generate_strategy_results(strategy_id: int, strategy_name: str, base_sharpe: float):
    """Generate realistic backtest results for a strategy."""
    # Add some noise to the base performance
    sharpe = max(0.1, np.random.normal(base_sharpe, 0.3))

    # Correlated metrics
    win_rate = min(0.85, max(0.25, 0.45 + sharpe * 0.08 + np.random.normal(0, 0.05)))
    total_trades = int(np.random.normal(800, 200))
    profit_factor = max(0.8, 1.0 + sharpe * 0.2 + np.random.normal(0, 0.15))
    max_dd_pct = max(5, abs(np.random.normal(15, 5)))
    total_pnl_pct = sharpe * 10 + np.random.normal(0, 5)
    avg_bars_held = np.random.uniform(1, 5)

    return {
        'strategy_id': strategy_id,
        'strategy_name': strategy_name,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd_pct,
        'total_pnl_pct': total_pnl_pct,
        'avg_bars_held': avg_bars_held
    }


def populate_sample_data(db_path: Path):
    """Populate database with sample backtest results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define instruments
    instruments = ['ES', 'NQ', 'YM', 'RTY', 'GC', 'SI', 'HG', 'CL', 'NG',
                   '6A', '6B', '6E', '6J', '6N', '6S']

    # Strategy definitions (ID, name, base_sharpe)
    strategies = [
        (21, 'RSI2_MeanReversion', 2.5),
        (37, 'Double7s', 2.0),
        (40, 'BuyOn5BarLow', 1.8),
        (41, 'ThreeBarLow', 1.9),
        (42, 'GapDownReversal', 2.2),
        (43, 'TurnOfMonth', 1.5),
        (44, 'BBIBSReversal', 2.1),
        (45, 'IBSStrategy', 2.3),
        (1, 'BollingerBands', 1.7),
        (2, 'KeltnerChannelBreakout', 1.9),
        (17, 'MACross', 1.6),
        (21, 'RSI2_MeanReversion', 2.4),  # Repeat for variety
    ]

    print("\nPopulating database with sample backtest results...")
    print("=" * 80)

    total_results = 0

    for symbol in instruments:
        # Create execution run for this instrument
        run_id = f"test_run_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cursor.execute("""
            INSERT INTO execution_runs (run_id, phase, status, symbols, strategies_passed, total_backtests)
            VALUES (?, 1, 'completed', ?, ?, ?)
        """, (run_id, json.dumps([symbol]), 0, len(strategies)))

        print(f"\n{symbol}:")
        winners_count = 0

        # Generate results for each strategy
        for strategy_id, strategy_name, base_sharpe in strategies:
            # Vary performance by instrument
            instrument_factor = np.random.uniform(0.8, 1.2)
            adjusted_sharpe = base_sharpe * instrument_factor

            result = generate_strategy_results(strategy_id, strategy_name, adjusted_sharpe)

            # Only add to DB if meets minimum criteria (simulate Phase 1 filtering)
            if result['total_trades'] >= 500 and result['win_rate'] >= 0.30:
                params = {
                    'stop_loss_atr': 1.0,
                    'take_profit_atr': 2.0,
                    'max_bars_held': 5
                }

                cursor.execute("""
                    INSERT INTO backtest_results
                    (run_id, strategy_id, strategy_name, params, sharpe_ratio,
                     total_trades, win_rate, profit_factor, max_drawdown_pct,
                     total_pnl_pct, avg_bars_held)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, result['strategy_id'], result['strategy_name'],
                    json.dumps(params), result['sharpe_ratio'], result['total_trades'],
                    result['win_rate'], result['profit_factor'], result['max_drawdown_pct'],
                    result['total_pnl_pct'], result['avg_bars_held']
                ))

                winners_count += 1
                total_results += 1

                print(f"  ✅ {strategy_name} (ID {strategy_id}): "
                      f"Sharpe={result['sharpe_ratio']:.2f}, "
                      f"WinRate={result['win_rate']:.1%}, "
                      f"Trades={result['total_trades']}")

        # Update strategies_passed count
        cursor.execute("""
            UPDATE execution_runs SET strategies_passed = ? WHERE run_id = ?
        """, (winners_count, run_id))

    conn.commit()
    conn.close()

    print("\n" + "=" * 80)
    print(f"✅ Populated database with {total_results} backtest results")
    print(f"   Instruments: {len(instruments)}")
    print(f"   Strategies per instrument: ~{total_results // len(instruments)}")
    print(f"   Database: {db_path}")


def main():
    print("=" * 80)
    print("CREATING SAMPLE STRATEGY FACTORY DATABASE")
    print("=" * 80)

    # Create output directory
    output_dir = Path('test_data/strategy_factory/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / 'strategy_factory.db'

    # Remove existing database
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing database: {db_path}")

    # Create schema
    create_database_schema(db_path)

    # Populate with sample data
    populate_sample_data(db_path)

    print("\n" + "=" * 80)
    print("DATABASE CREATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Test extract_winners.py with this database:")
    print(f"  python research/strategy_factory/extract_winners.py \\")
    print(f"    --db-path {db_path} \\")
    print(f"    --top-n 10 \\")
    print(f"    --output test_data/winners_manifest.json")
    print()


if __name__ == '__main__':
    main()
