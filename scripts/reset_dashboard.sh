#!/bin/bash
# Reset Dashboard for New Portfolio Configuration
# This script backs up old trades and resets the dashboard to start fresh

set -e

PROJECT_ROOT="/opt/pine/rooney-capital-v1"
TRADES_DB="/opt/pine/runtime/trades.db"
BACKUP_DIR="/opt/pine/runtime/backups"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "🔄 DASHBOARD RESET - New Configuration B (16 instruments, max 2 positions)"
echo "================================================================================"
echo ""

# 1. Check if database exists
if [ ! -f "$TRADES_DB" ]; then
    echo -e "${YELLOW}ℹ️  No existing trades database found${NC}"
    echo "   Database will be created automatically when system runs"
    exit 0
fi

echo "📊 Current Database Status:"
echo "   Location: $TRADES_DB"

# Count existing trades using sqlite3
TRADE_COUNT=$(sqlite3 "$TRADES_DB" "SELECT COUNT(*) FROM trades;" 2>/dev/null || echo "0")
echo "   Existing trades: $TRADE_COUNT"
echo ""

if [ "$TRADE_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Database is already empty - nothing to reset"
    exit 0
fi

# 2. Confirm reset
echo -e "${YELLOW}⚠️  WARNING: This will backup and clear all existing trade data${NC}"
read -p "Continue with reset? (yes/no) " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Reset cancelled"
    exit 1
fi

# 3. Create backup directory
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓${NC} Backup directory ready: $BACKUP_DIR"

# 4. Backup existing database
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/trades_backup_$TIMESTAMP.db"
cp "$TRADES_DB" "$BACKUP_FILE"
echo -e "${GREEN}✓${NC} Backed up existing database to:"
echo "   $BACKUP_FILE"

# 5. Export trades to CSV for easy reference
CSV_FILE="$BACKUP_DIR/trades_export_$TIMESTAMP.csv"
sqlite3 -header -csv "$TRADES_DB" "SELECT * FROM trades ORDER BY exit_time DESC;" > "$CSV_FILE"
echo -e "${GREEN}✓${NC} Exported trades to CSV:"
echo "   $CSV_FILE"

# 6. Show summary of backed up data
echo ""
echo "📈 Backed up data summary:"
sqlite3 "$TRADES_DB" "
SELECT
    symbol,
    COUNT(*) as trade_count,
    ROUND(SUM(pnl), 2) as total_pnl,
    ROUND(AVG(pnl), 2) as avg_pnl
FROM trades
GROUP BY symbol
ORDER BY total_pnl DESC;
" -column -header

echo ""
echo "Total P&L from backed up trades:"
sqlite3 "$TRADES_DB" "SELECT ROUND(SUM(pnl), 2) as total_pnl FROM trades;" -column

# 7. Clear the database (delete all trades)
echo ""
echo "🗑️  Clearing database..."
sqlite3 "$TRADES_DB" "DELETE FROM trades;"
sqlite3 "$TRADES_DB" "VACUUM;"  # Reclaim disk space
echo -e "${GREEN}✓${NC} Database cleared - starting fresh"

# 8. Verify database is empty
NEW_COUNT=$(sqlite3 "$TRADES_DB" "SELECT COUNT(*) FROM trades;")
echo -e "${GREEN}✓${NC} Verified: Database now has $NEW_COUNT trades"

# 9. Find and restart Streamlit dashboard
echo ""
echo "🔍 Checking for running Streamlit dashboard..."
STREAMLIT_PID=$(ps aux | grep "streamlit run" | grep -v grep | awk '{print $2}' || echo "")

if [ -n "$STREAMLIT_PID" ]; then
    echo "   Found Streamlit running (PID: $STREAMLIT_PID)"
    echo "   Stopping dashboard..."
    kill $STREAMLIT_PID 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}✓${NC} Dashboard stopped"

    # Check if running in supervisor/systemd
    if command -v supervisorctl &> /dev/null; then
        echo "   Restarting via supervisor..."
        supervisorctl restart dashboard 2>/dev/null || echo "   (no supervisor service found)"
    elif systemctl list-units --type=service | grep -q streamlit; then
        echo "   Restarting via systemd..."
        sudo systemctl restart streamlit-dashboard 2>/dev/null || echo "   (no systemd service found)"
    else
        echo "   To restart dashboard manually, run:"
        echo "   cd $PROJECT_ROOT/dashboard && streamlit run app.py --server.port 8501 &"
    fi
else
    echo "   No running Streamlit dashboard found"
    echo "   Start dashboard with:"
    echo "   cd $PROJECT_ROOT/dashboard && streamlit run app.py --server.port 8501 &"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ DASHBOARD RESET COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  • Backed up $TRADE_COUNT trades to: $BACKUP_FILE"
echo "  • Exported to CSV: $CSV_FILE"
echo "  • Database cleared and ready for Configuration B"
echo "  • Configuration: 16 instruments, max 2 positions"
echo ""
echo "Backups are stored in: $BACKUP_DIR"
echo ""
