#!/usr/bin/env python3
"""Reset Dashboard for New Portfolio Configuration

This script backs up old trades and resets the dashboard to start fresh.
"""

import sqlite3
import shutil
import csv
from pathlib import Path
from datetime import datetime
import sys

# Paths
TRADES_DB = Path("/opt/pine/runtime/trades.db")
BACKUP_DIR = Path("/opt/pine/runtime/backups")

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_colored(text, color):
    """Print colored text to terminal."""
    print(f"{color}{text}{NC}")

def main():
    print("=" * 80)
    print("🔄 DASHBOARD RESET - New Configuration B (16 instruments, max 2 positions)")
    print("=" * 80)
    print()

    # 1. Check if database exists
    if not TRADES_DB.exists():
        print_colored("ℹ️  No existing trades database found", YELLOW)
        print("   Database will be created automatically when system runs")
        return 0

    print("📊 Current Database Status:")
    print(f"   Location: {TRADES_DB}")

    # Count existing trades
    conn = sqlite3.connect(TRADES_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    trade_count = cursor.fetchone()[0]
    print(f"   Existing trades: {trade_count}")
    print()

    if trade_count == 0:
        print_colored("✓ Database is already empty - nothing to reset", GREEN)
        conn.close()
        return 0

    # 2. Confirm reset
    print_colored("⚠️  WARNING: This will backup and clear all existing trade data", YELLOW)
    response = input("Continue with reset? (yes/no) ")
    if response.lower() != 'yes':
        print("Reset cancelled")
        conn.close()
        return 1

    # 3. Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print_colored(f"✓ Backup directory ready: {BACKUP_DIR}", GREEN)

    # 4. Generate timestamp for backup files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"trades_backup_{timestamp}.db"
    csv_file = BACKUP_DIR / f"trades_export_{timestamp}.csv"

    # 5. Backup database file
    shutil.copy2(TRADES_DB, backup_file)
    print_colored(f"✓ Backed up existing database to:", GREEN)
    print(f"   {backup_file}")

    # 6. Export trades to CSV
    cursor.execute("SELECT * FROM trades ORDER BY exit_time DESC")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

    print_colored(f"✓ Exported trades to CSV:", GREEN)
    print(f"   {csv_file}")

    # 7. Show summary of backed up data
    print()
    print("📈 Backed up data summary:")
    cursor.execute("""
        SELECT
            symbol,
            COUNT(*) as trade_count,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl
        FROM trades
        GROUP BY symbol
        ORDER BY total_pnl DESC
    """)

    print(f"{'Symbol':<10} {'Trades':<10} {'Total P&L':<15} {'Avg P&L':<15}")
    print("-" * 50)
    for row in cursor.fetchall():
        symbol, count, total_pnl, avg_pnl = row
        print(f"{symbol:<10} {count:<10} ${total_pnl:<14.2f} ${avg_pnl:<14.2f}")

    print()
    cursor.execute("SELECT ROUND(SUM(pnl), 2) as total_pnl FROM trades")
    total_pnl = cursor.fetchone()[0]
    print(f"Total P&L from backed up trades: ${total_pnl:.2f}")

    # 8. Clear the database
    print()
    print("🗑️  Clearing database...")
    cursor.execute("DELETE FROM trades")
    conn.commit()

    # Vacuum to reclaim disk space
    cursor.execute("VACUUM")
    conn.commit()

    print_colored("✓ Database cleared - starting fresh", GREEN)

    # 9. Verify database is empty
    cursor.execute("SELECT COUNT(*) FROM trades")
    new_count = cursor.fetchone()[0]
    print_colored(f"✓ Verified: Database now has {new_count} trades", GREEN)

    conn.close()

    # 10. Check for Streamlit
    print()
    print("🔍 To start the dashboard, run:")
    print("   cd /opt/pine/rooney-capital-v1/dashboard")
    print("   streamlit run app.py --server.port 8501 --server.address 0.0.0.0")

    print()
    print("=" * 80)
    print_colored("✅ DASHBOARD RESET COMPLETE", GREEN)
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • Backed up {trade_count} trades to: {backup_file}")
    print(f"  • Exported to CSV: {csv_file}")
    print("  • Database cleared and ready for Configuration B")
    print("  • Configuration: 16 instruments, max 2 positions")
    print()
    print(f"Backups are stored in: {BACKUP_DIR}")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
