#!/bin/bash
# Real-time Portfolio Monitoring Dashboard
# Run this in a separate terminal while the system is running

PROJECT_ROOT="/opt/pine/rooney-capital-v1"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get most recent log file
LOG_FILE=$(ls -t logs/portfolio_system_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    LOG_FILE=$(ls -t logs/worker_*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ]; then
    echo -e "${RED}❌ No log files found${NC}"
    echo "Start the system first with: ./launch_portfolio_system.sh"
    exit 1
fi

echo "================================================================================"
echo "📊 PORTFOLIO SYSTEM MONITOR"
echo "================================================================================"
echo "Monitoring: $LOG_FILE"
echo ""
echo "Press Ctrl+C to exit"
echo "================================================================================"
echo ""

# Function to show stats
show_stats() {
    clear
    echo "================================================================================"
    echo -e "${BLUE}📊 PORTFOLIO SYSTEM DASHBOARD${NC}"
    echo "================================================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log: $(basename $LOG_FILE)"
    echo ""

    # Check if system is running
    if pgrep -f "runner.main" > /dev/null; then
        echo -e "${GREEN}✓ System is RUNNING${NC}"
    else
        echo -e "${RED}✗ System is NOT RUNNING${NC}"
    fi
    echo ""

    echo "================================================================================"
    echo -e "${BLUE}PORTFOLIO COORDINATOR STATUS${NC}"
    echo "================================================================================"

    # Portfolio coordinator initialization
    if grep -q "PortfolioCoordinator" "$LOG_FILE"; then
        echo -e "${GREEN}✓${NC} Portfolio Coordinator initialized"
        grep "max_positions\|daily_stop_loss" "$LOG_FILE" | tail -3 | sed 's/^/  /'
    else
        echo -e "${YELLOW}⏳${NC} Waiting for Portfolio Coordinator initialization..."
    fi
    echo ""

    # Current positions
    echo "Current Open Positions:"
    POSITIONS=$(grep -i "registered with portfolio\|position opened" "$LOG_FILE" | tail -5)
    if [ -z "$POSITIONS" ]; then
        echo "  No positions opened yet"
    else
        echo "$POSITIONS" | sed 's/^/  /'
    fi
    echo ""

    # Position blocks
    BLOCKS=$(grep -i "ENTRY BLOCKED BY PORTFOLIO" "$LOG_FILE" | wc -l)
    echo -e "Entry Blocks (max positions): ${YELLOW}$BLOCKS${NC}"
    if [ "$BLOCKS" -gt 0 ]; then
        grep -i "ENTRY BLOCKED BY PORTFOLIO" "$LOG_FILE" | tail -3 | sed 's/^/  /'
    fi
    echo ""

    echo "================================================================================"
    echo -e "${BLUE}TRADING ACTIVITY${NC}"
    echo "================================================================================"

    # Count orders
    BUY_ORDERS=$(grep -i "PLACING BUY ORDER" "$LOG_FILE" | wc -l)
    SELL_ORDERS=$(grep -i "PLACING SELL ORDER" "$LOG_FILE" | wc -l)
    TOTAL_ORDERS=$((BUY_ORDERS + SELL_ORDERS))

    echo "Orders Placed: $TOTAL_ORDERS (Buy: $BUY_ORDERS, Sell: $SELL_ORDERS)"

    # Recent orders
    if [ "$TOTAL_ORDERS" -gt 0 ]; then
        echo ""
        echo "Recent Orders:"
        grep -E "PLACING (BUY|SELL) ORDER" "$LOG_FILE" | tail -5 | sed 's/^/  /'
    fi
    echo ""

    echo "================================================================================"
    echo -e "${BLUE}RISK MANAGEMENT${NC}"
    echo "================================================================================"

    # Stop loss events
    STOP_LOSS_COUNT=$(grep -i "STOP LOSS TRIGGERED\|stopped out" "$LOG_FILE" | wc -l)
    if [ "$STOP_LOSS_COUNT" -gt 0 ]; then
        echo -e "${RED}⚠️  Daily Stop Loss Hit: $STOP_LOSS_COUNT times${NC}"
        grep -i "STOP LOSS" "$LOG_FILE" | tail -3 | sed 's/^/  /'
    else
        echo -e "${GREEN}✓${NC} No stop loss events"
    fi
    echo ""

    # Daily P&L tracking
    echo "Daily P&L Updates:"
    grep -i "daily_pnl\|Daily P&L" "$LOG_FILE" | tail -3 | sed 's/^/  /'
    echo ""

    echo "================================================================================"
    echo -e "${BLUE}SYSTEM HEALTH${NC}"
    echo "================================================================================"

    # Errors
    ERROR_COUNT=$(grep -i "ERROR\|CRITICAL" "$LOG_FILE" | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${RED}⚠️  Errors: $ERROR_COUNT${NC}"
        grep -i "ERROR\|CRITICAL" "$LOG_FILE" | tail -3 | sed 's/^/  /'
    else
        echo -e "${GREEN}✓${NC} No errors"
    fi
    echo ""

    # Warnings
    WARN_COUNT=$(grep -i "WARNING" "$LOG_FILE" | wc -l)
    if [ "$WARN_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warnings: $WARN_COUNT${NC}"
    else
        echo -e "${GREEN}✓${NC} No warnings"
    fi
    echo ""

    echo "================================================================================"
    echo -e "${BLUE}RECENT ACTIVITY (Last 10 lines)${NC}"
    echo "================================================================================"
    tail -10 "$LOG_FILE" | sed 's/^/  /'
    echo ""
    echo "================================================================================"
}

# Main monitoring loop
while true; do
    show_stats
    sleep 5
done
