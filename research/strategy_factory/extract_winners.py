#!/usr/bin/env python3
"""
Extract Top Winners from Strategy Factory Database

After Phase 1 completes for all instruments, this script:
1. Queries the database for each instrument's run
2. Extracts top N strategies by Sharpe ratio (after all filters)
3. Outputs a manifest for ML pipeline automation

Usage:
    python research/strategy_factory/extract_winners.py \
        --top-n 10 \
        --output winners_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import sqlite3
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from research.strategy_factory.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_top_winners(
    db_path: Path,
    top_n: int = 10,
    instruments: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract top N winners for each instrument.

    Args:
        db_path: Path to strategy_factory.db
        top_n: Number of top strategies per instrument
        instruments: List of instrument symbols (None = all)

    Returns:
        List of winner dictionaries with strategy info
    """
    if instruments is None:
        instruments = ["ES", "NQ", "YM", "RTY", "GC", "SI", "HG",
                      "CL", "NG", "6A", "6B", "6E", "6J", "6N", "6S"]

    winners = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for symbol in instruments:
        logger.info(f"Extracting top {top_n} winners for {symbol}...")

        # Find the most recent completed Phase 1 run for this symbol
        cursor = conn.execute("""
            SELECT run_id, strategies_passed, total_backtests
            FROM execution_runs
            WHERE phase = 1
              AND status = 'completed'
              AND symbols LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (f'%"{symbol}"%',))

        run = cursor.fetchone()

        if not run:
            logger.warning(f"No completed Phase 1 run found for {symbol}")
            continue

        run_id = run['run_id']
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Strategies passed: {run['strategies_passed']}")
        logger.info(f"  Total backtests: {run['total_backtests']}")

        # Get top N strategies by Sharpe ratio
        # NOTE: We want strategies that passed ALL filters (final winners)
        # For now, we'll take top N by Sharpe from all results
        # TODO: Add filter tracking to know which passed all filters
        cursor = conn.execute("""
            SELECT
                strategy_id,
                strategy_name,
                params,
                sharpe_ratio,
                total_trades,
                win_rate,
                profit_factor,
                max_drawdown_pct,
                total_pnl_pct,
                avg_bars_held
            FROM backtest_results
            WHERE run_id = ?
              AND sharpe_ratio > 0
              AND total_trades >= 500
            ORDER BY sharpe_ratio DESC
            LIMIT ?
        """, (run_id, top_n))

        symbol_winners = cursor.fetchall()

        logger.info(f"  Found {len(symbol_winners)} winners")

        for i, winner in enumerate(symbol_winners, 1):
            winner_dict = {
                'rank': i,
                'symbol': symbol,
                'run_id': run_id,
                'strategy_id': winner['strategy_id'],
                'strategy_name': winner['strategy_name'],
                'params': json.loads(winner['params']),
                'sharpe_ratio': winner['sharpe_ratio'],
                'total_trades': winner['total_trades'],
                'win_rate': winner['win_rate'],
                'profit_factor': winner['profit_factor'],
                'max_drawdown_pct': winner['max_drawdown_pct'],
                'total_pnl_pct': winner['total_pnl_pct'],
                'avg_bars_held': winner['avg_bars_held']
            }
            winners.append(winner_dict)

            logger.info(
                f"    #{i}: {winner['strategy_name']} "
                f"(Sharpe={winner['sharpe_ratio']:.3f}, "
                f"Trades={winner['total_trades']})"
            )

        logger.info("")

    conn.close()

    return winners


def generate_ml_manifest(
    winners: List[Dict[str, Any]],
    output_path: Path
):
    """
    Generate manifest file for ML pipeline automation.

    The manifest includes all info needed to:
    1. Port strategy to Backtrader
    2. Extract training data
    3. Run ML optimization
    4. Validate on 2022-2024

    Args:
        winners: List of winner dictionaries
        output_path: Path to write manifest JSON
    """
    manifest = {
        'version': '1.0',
        'created_at': pd.Timestamp.now().isoformat(),
        'total_winners': len(winners),
        'winners_by_instrument': {},
        'winners': winners
    }

    # Group by instrument for summary
    for winner in winners:
        symbol = winner['symbol']
        if symbol not in manifest['winners_by_instrument']:
            manifest['winners_by_instrument'][symbol] = []
        manifest['winners_by_instrument'][symbol].append({
            'strategy_id': winner['strategy_id'],
            'strategy_name': winner['strategy_name'],
            'sharpe_ratio': winner['sharpe_ratio']
        })

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✅ Manifest written to {output_path}")
    logger.info(f"Total winners: {len(winners)}")
    logger.info(f"Instruments: {len(manifest['winners_by_instrument'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract top winning strategies from Phase 1 results'
    )
    parser.add_argument(
        '--top-n', type=int, default=10,
        help='Number of top strategies per instrument (default: 10)'
    )
    parser.add_argument(
        '--output', type=str, default='ml_pipeline/winners_manifest.json',
        help='Output path for manifest file'
    )
    parser.add_argument(
        '--db-path', type=str,
        default='research/strategy_factory/results/strategy_factory.db',
        help='Path to strategy factory database'
    )
    parser.add_argument(
        '--instruments', nargs='+',
        help='Specific instruments to process (default: all 15)'
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    output_path = Path(args.output)

    logger.info("=" * 80)
    logger.info("EXTRACTING TOP WINNERS FROM STRATEGY FACTORY")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Top N per instrument: {args.top_n}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Extract winners
    winners = extract_top_winners(
        db_path=db_path,
        top_n=args.top_n,
        instruments=args.instruments
    )

    if not winners:
        logger.error("No winners found!")
        return 1

    # Generate manifest
    import pandas as pd  # Import here to avoid dependency in main flow
    generate_ml_manifest(winners, output_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("1. Review winners_manifest.json")
    logger.info("2. Port strategies to Backtrader format")
    logger.info("3. Run ML pipeline automation:")
    logger.info("   python research/strategy_factory/run_ml_pipeline.py \\")
    logger.info("       --manifest ml_pipeline/winners_manifest.json")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
