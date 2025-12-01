#!/usr/bin/env python3
"""
Parse ML pipeline results and store in SQLite database for easy analysis.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def parse_executive_summary(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parse an executive summary file and extract key metrics."""
    try:
        content = filepath.read_text()

        # Extract symbol name
        symbol_match = re.search(r'Symbol:\s*(\S+)', content)
        symbol = symbol_match.group(1) if symbol_match else filepath.stem.replace('_ml_meta_labeling_executive_summary', '')

        # Parse symbol into instrument and strategy
        parts = symbol.split('_', 1)
        instrument = parts[0] if len(parts) > 0 else symbol
        strategy = parts[1] if len(parts) > 1 else ''

        # Extract selected features count
        features_match = re.search(r'Selected Features:\s*(\d+)', content)
        n_features = int(features_match.group(1)) if features_match else None

        # Extract walk-forward stats
        mean_sharpe_match = re.search(r'Mean Test Sharpe:\s*([-\d.]+)\s*±\s*([-\d.]+)', content)
        wf_mean_sharpe = float(mean_sharpe_match.group(1)) if mean_sharpe_match else None
        wf_std_sharpe = float(mean_sharpe_match.group(2)) if mean_sharpe_match else None

        mean_auc_match = re.search(r'Mean Test AUC:\s*([-\d.]+)\s*±\s*([-\d.]+)', content)
        wf_mean_auc = float(mean_auc_match.group(1)) if mean_auc_match else None

        pos_windows_match = re.search(r'Positive Windows:\s*(\d+)/(\d+)', content)
        positive_windows = int(pos_windows_match.group(1)) if pos_windows_match else None
        total_windows = int(pos_windows_match.group(2)) if pos_windows_match else None

        # Extract held-out test metrics
        test_auc_match = re.search(r'Test AUC:\s*([-\d.]+)', content)
        test_auc = float(test_auc_match.group(1)) if test_auc_match else None

        # Unfiltered metrics
        unfiltered_section = re.search(r'Unfiltered.*?Total Trades:\s*(\d+).*?Win Rate:\s*([-\d.]+)%.*?Sharpe Ratio:\s*([-\d.]+).*?Profit Factor:\s*([-\d.]+)', content, re.DOTALL)
        if unfiltered_section:
            trades_unfiltered = int(unfiltered_section.group(1))
            winrate_unfiltered = float(unfiltered_section.group(2))
            sharpe_unfiltered = float(unfiltered_section.group(3))
            pf_unfiltered = float(unfiltered_section.group(4))
        else:
            trades_unfiltered = winrate_unfiltered = sharpe_unfiltered = pf_unfiltered = None

        # Filtered metrics
        filtered_section = re.search(r'Filtered.*?Total Trades:\s*(\d+).*?Filter Rate:\s*([-\d.]+)%.*?Win Rate:\s*([-\d.]+)%.*?Sharpe Ratio:\s*([-\d.]+).*?Profit Factor:\s*([-\d.]+)', content, re.DOTALL)
        if filtered_section:
            trades_filtered = int(filtered_section.group(1))
            filter_rate = float(filtered_section.group(2))
            winrate_filtered = float(filtered_section.group(3))
            sharpe_filtered = float(filtered_section.group(4))
            pf_filtered = float(filtered_section.group(5))
        else:
            trades_filtered = filter_rate = winrate_filtered = sharpe_filtered = pf_filtered = None

        # Calculate improvements
        sharpe_improvement = None
        sharpe_improved = None
        if sharpe_filtered is not None and sharpe_unfiltered is not None:
            sharpe_improvement = sharpe_filtered - sharpe_unfiltered
            sharpe_improved = sharpe_filtered > sharpe_unfiltered

        pf_improvement = None
        if pf_filtered is not None and pf_unfiltered is not None:
            pf_improvement = pf_filtered - pf_unfiltered

        winrate_improvement = None
        if winrate_filtered is not None and winrate_unfiltered is not None:
            winrate_improvement = winrate_filtered - winrate_unfiltered

        return {
            'symbol': symbol,
            'instrument': instrument,
            'strategy': strategy,
            'n_features': n_features,
            'wf_mean_sharpe': wf_mean_sharpe,
            'wf_std_sharpe': wf_std_sharpe,
            'wf_mean_auc': wf_mean_auc,
            'positive_windows': positive_windows,
            'total_windows': total_windows,
            'test_auc': test_auc,
            'trades_unfiltered': trades_unfiltered,
            'trades_filtered': trades_filtered,
            'filter_rate': filter_rate,
            'winrate_unfiltered': winrate_unfiltered,
            'winrate_filtered': winrate_filtered,
            'winrate_improvement': winrate_improvement,
            'sharpe_unfiltered': sharpe_unfiltered,
            'sharpe_filtered': sharpe_filtered,
            'sharpe_improvement': sharpe_improvement,
            'sharpe_improved': sharpe_improved,
            'pf_unfiltered': pf_unfiltered,
            'pf_filtered': pf_filtered,
            'pf_improvement': pf_improvement,
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def create_database(db_path: str):
    """Create SQLite database with results table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE,
            instrument TEXT,
            strategy TEXT,
            n_features INTEGER,

            -- Walk-forward validation metrics
            wf_mean_sharpe REAL,
            wf_std_sharpe REAL,
            wf_mean_auc REAL,
            positive_windows INTEGER,
            total_windows INTEGER,

            -- Held-out test metrics
            test_auc REAL,

            -- Unfiltered (baseline) metrics
            trades_unfiltered INTEGER,
            winrate_unfiltered REAL,
            sharpe_unfiltered REAL,
            pf_unfiltered REAL,

            -- Filtered (ML) metrics
            trades_filtered INTEGER,
            filter_rate REAL,
            winrate_filtered REAL,
            sharpe_filtered REAL,
            pf_filtered REAL,

            -- Improvements
            winrate_improvement REAL,
            sharpe_improvement REAL,
            sharpe_improved BOOLEAN,
            pf_improvement REAL,

            -- Status
            passed BOOLEAN,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    return conn


def insert_result(conn: sqlite3.Connection, data: Dict[str, Any]):
    """Insert or replace a result in the database."""
    cursor = conn.cursor()

    # Determine if it "passed" (Sharpe improved)
    passed = data.get('sharpe_improved', False)

    cursor.execute('''
        INSERT OR REPLACE INTO ml_results (
            symbol, instrument, strategy, n_features,
            wf_mean_sharpe, wf_std_sharpe, wf_mean_auc, positive_windows, total_windows,
            test_auc,
            trades_unfiltered, winrate_unfiltered, sharpe_unfiltered, pf_unfiltered,
            trades_filtered, filter_rate, winrate_filtered, sharpe_filtered, pf_filtered,
            winrate_improvement, sharpe_improvement, sharpe_improved, pf_improvement,
            passed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['symbol'], data['instrument'], data['strategy'], data['n_features'],
        data['wf_mean_sharpe'], data['wf_std_sharpe'], data['wf_mean_auc'],
        data['positive_windows'], data['total_windows'],
        data['test_auc'],
        data['trades_unfiltered'], data['winrate_unfiltered'], data['sharpe_unfiltered'], data['pf_unfiltered'],
        data['trades_filtered'], data['filter_rate'], data['winrate_filtered'], data['sharpe_filtered'], data['pf_filtered'],
        data['winrate_improvement'], data['sharpe_improvement'], data['sharpe_improved'], data['pf_improvement'],
        passed
    ))
    conn.commit()


def print_summary(conn: sqlite3.Connection):
    """Print a summary of results."""
    cursor = conn.cursor()

    print("\n" + "=" * 100)
    print("ML PIPELINE RESULTS SUMMARY")
    print("=" * 100)

    # Overall stats
    cursor.execute("SELECT COUNT(*) FROM ml_results")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ml_results WHERE passed = 1")
    passed = cursor.fetchone()[0]

    print(f"\nTotal strategies: {total}")
    print(f"Sharpe improved:  {passed} ({100*passed/total:.1f}%)")
    print(f"Sharpe worse:     {total - passed} ({100*(total-passed)/total:.1f}%)")

    # Top performers (by Sharpe improvement)
    print("\n" + "-" * 100)
    print("TOP 10 BY SHARPE IMPROVEMENT")
    print("-" * 100)
    print(f"{'Symbol':<35} {'Unfiltered':>12} {'Filtered':>12} {'Improvement':>12} {'Filter%':>10}")
    print("-" * 100)

    cursor.execute('''
        SELECT symbol, sharpe_unfiltered, sharpe_filtered, sharpe_improvement, filter_rate
        FROM ml_results
        ORDER BY sharpe_improvement DESC
        LIMIT 10
    ''')
    for row in cursor.fetchall():
        symbol, unf, filt, imp, frate = row
        print(f"{symbol:<35} {unf:>12.3f} {filt:>12.3f} {imp:>+12.3f} {frate:>9.1f}%")

    # Worst performers
    print("\n" + "-" * 100)
    print("BOTTOM 5 BY SHARPE IMPROVEMENT (ML hurt performance)")
    print("-" * 100)
    print(f"{'Symbol':<35} {'Unfiltered':>12} {'Filtered':>12} {'Improvement':>12} {'Filter%':>10}")
    print("-" * 100)

    cursor.execute('''
        SELECT symbol, sharpe_unfiltered, sharpe_filtered, sharpe_improvement, filter_rate
        FROM ml_results
        ORDER BY sharpe_improvement ASC
        LIMIT 5
    ''')
    for row in cursor.fetchall():
        symbol, unf, filt, imp, frate = row
        print(f"{symbol:<35} {unf:>12.3f} {filt:>12.3f} {imp:>+12.3f} {frate:>9.1f}%")

    # By instrument
    print("\n" + "-" * 100)
    print("BY INSTRUMENT")
    print("-" * 100)

    cursor.execute('''
        SELECT instrument,
               COUNT(*) as total,
               SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
               AVG(sharpe_improvement) as avg_improvement
        FROM ml_results
        GROUP BY instrument
        ORDER BY avg_improvement DESC
    ''')
    print(f"{'Instrument':<10} {'Total':>8} {'Passed':>8} {'Avg Improvement':>16}")
    for row in cursor.fetchall():
        inst, total, passed, avg_imp = row
        print(f"{inst:<10} {total:>8} {passed:>8} {avg_imp:>+16.3f}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Parse ML results into SQLite database')
    parser.add_argument('--results-dir', type=str,
                        default='research/strategy_factory/outputs',
                        help='Directory containing executive summary files')
    parser.add_argument('--db-path', type=str,
                        default='research/strategy_factory/outputs/ml_results.db',
                        help='Path to SQLite database')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent

    results_dir = project_root / args.results_dir
    db_path = project_root / args.db_path

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    # Create database
    conn = create_database(str(db_path))

    # Parse all executive summaries
    summary_files = list(results_dir.glob('*_executive_summary.txt'))

    if not summary_files:
        print(f"No executive summary files found in {results_dir}")
        return 1

    print(f"Found {len(summary_files)} executive summary files")

    parsed = 0
    for filepath in summary_files:
        data = parse_executive_summary(filepath)
        if data:
            insert_result(conn, data)
            parsed += 1
            if not args.quiet:
                status = "IMPROVED" if data['sharpe_improved'] else "WORSE"
                print(f"  {data['symbol']}: {status} ({data['sharpe_improvement']:+.3f})")

    print(f"\nParsed {parsed}/{len(summary_files)} files into {db_path}")

    # Print summary
    print_summary(conn)

    conn.close()

    print(f"\nDatabase saved to: {db_path}")
    print("\nQuery examples:")
    print(f"  sqlite3 {db_path} 'SELECT * FROM ml_results WHERE passed = 1'")
    print(f"  sqlite3 {db_path} 'SELECT symbol, sharpe_improvement FROM ml_results ORDER BY sharpe_improvement DESC'")

    return 0


if __name__ == '__main__':
    exit(main())
