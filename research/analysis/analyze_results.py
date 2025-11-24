#!/usr/bin/env python3
"""Analyze Phase 1 backtest results from database."""

import sqlite3
import pandas as pd
import sys

def analyze_results(db_path='/opt/pine/rooney-capital-v1/research/strategy_factory/output/phase1_results.db'):
    """Analyze all backtest results."""

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print(f"Database path: {db_path}")
        return

    print("=" * 100)
    print("PHASE 1 BACKTEST RESULTS - ALL STRATEGIES")
    print("=" * 100)
    print()

    # Get total counts
    total_query = "SELECT COUNT(*) as total FROM backtests;"
    total_df = pd.read_sql_query(total_query, conn)
    total_backtests = total_df['total'][0]

    print(f"Total Backtests: {total_backtests}")
    print()

    # Summary by strategy
    print("=" * 100)
    print("SUMMARY BY STRATEGY")
    print("=" * 100)

    summary_query = """
    SELECT
        strategy_name,
        COUNT(*) as combos,
        ROUND(AVG(total_trades), 0) as avg_trades,
        ROUND(MIN(total_trades), 0) as min_trades,
        ROUND(MAX(total_trades), 0) as max_trades,
        ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
        ROUND(MAX(sharpe_ratio), 3) as best_sharpe,
        ROUND(AVG(profit_factor), 3) as avg_pf,
        ROUND(MAX(profit_factor), 3) as best_pf
    FROM backtests
    GROUP BY strategy_name
    ORDER BY max_trades DESC;
    """

    df_summary = pd.read_sql_query(summary_query, conn)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    print(df_summary.to_string(index=False))
    print()

    # Check filter thresholds
    print("=" * 100)
    print("GATE 1 FILTER ANALYSIS")
    print("=" * 100)

    filter_query = """
    SELECT
        SUM(CASE WHEN total_trades >= 5000 AND sharpe_ratio >= 0.0 AND profit_factor >= 1.0 AND win_rate >= 0.35 THEN 1 ELSE 0 END) as pass_5000,
        SUM(CASE WHEN total_trades >= 3000 AND sharpe_ratio >= 0.0 AND profit_factor >= 1.0 AND win_rate >= 0.35 THEN 1 ELSE 0 END) as pass_3000,
        SUM(CASE WHEN total_trades >= 2000 AND sharpe_ratio >= 0.0 AND profit_factor >= 1.0 AND win_rate >= 0.35 THEN 1 ELSE 0 END) as pass_2000,
        SUM(CASE WHEN total_trades >= 1000 AND sharpe_ratio >= 0.0 AND profit_factor >= 1.0 AND win_rate >= 0.35 THEN 1 ELSE 0 END) as pass_1000,
        COUNT(*) as total
    FROM backtests;
    """

    df_filters = pd.read_sql_query(filter_query, conn)
    print(f"Current Gate 1 Filter: trades≥1000, sharpe≥0.0, pf≥1.0, winrate≥35%")
    print(f"  Passing with ≥1000 trades: {df_filters['pass_1000'][0]} / {df_filters['total'][0]}")
    print(f"  Would pass with ≥2000 trades: {df_filters['pass_2000'][0]} / {df_filters['total'][0]}")
    print(f"  Would pass with ≥3000 trades: {df_filters['pass_3000'][0]} / {df_filters['total'][0]}")
    print(f"  Would pass with ≥5000 trades: {df_filters['pass_5000'][0]} / {df_filters['total'][0]}")
    print()

    # Top 20 performers
    print("=" * 100)
    print("TOP 20 PERFORMERS (by Sharpe Ratio)")
    print("=" * 100)

    top_query = """
    SELECT
        strategy_name,
        params,
        total_trades,
        ROUND(sharpe_ratio, 3) as sharpe,
        ROUND(profit_factor, 3) as pf,
        ROUND(total_return_pct, 2) as return_pct,
        ROUND(max_drawdown_pct, 2) as dd_pct,
        ROUND(win_rate, 3) as winrate
    FROM backtests
    ORDER BY sharpe_ratio DESC
    LIMIT 20;
    """

    df_top = pd.read_sql_query(top_query, conn)
    pd.set_option('display.max_colwidth', 50)
    print(df_top.to_string(index=False))
    print()

    # Strategies that would pass 1000 trade threshold
    print("=" * 100)
    print("STRATEGIES PASSING GATE 1 (trades≥1000, sharpe≥0, pf≥1.0, winrate≥35%)")
    print("=" * 100)

    pass_query = """
    SELECT
        strategy_name,
        params,
        total_trades,
        ROUND(sharpe_ratio, 3) as sharpe,
        ROUND(profit_factor, 3) as pf,
        ROUND(win_rate, 3) as winrate
    FROM backtests
    WHERE total_trades >= 1000
      AND sharpe_ratio >= 0.0
      AND profit_factor >= 1.0
      AND win_rate >= 0.35
    ORDER BY sharpe_ratio DESC;
    """

    df_pass = pd.read_sql_query(pass_query, conn)

    if len(df_pass) > 0:
        print(df_pass.to_string(index=False))
    else:
        print("No strategies passed Gate 1 with current thresholds.")

    conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        analyze_results()
