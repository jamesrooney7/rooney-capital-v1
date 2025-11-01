#!/bin/bash
# Quick Launch Script for Portfolio Optimization System
# This launches the system with real-time monitoring

set -e

PROJECT_ROOT="/opt/pine/rooney-capital-v1"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "🚀 LAUNCHING PORTFOLIO OPTIMIZATION SYSTEM"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Pre-flight checks
echo "📋 Running pre-flight checks..."
echo ""

# 1. Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment exists"

# 2. Check portfolio config
if [ ! -f "config/portfolio_optimization.json" ]; then
    echo -e "${RED}❌ Portfolio config not found: config/portfolio_optimization.json${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Portfolio config exists"

# 3. Check runtime config
if [ ! -f "config.yml" ]; then
    echo -e "${RED}❌ Runtime config not found: config.yml${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Runtime config exists"

# 4. Check .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Environment file not found: .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Environment file exists"

# 5. Check required environment variables
source .env
if [ -z "$DATABENTO_API_KEY" ]; then
    echo -e "${RED}❌ DATABENTO_API_KEY not set in .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} DATABENTO_API_KEY configured"

if [ -z "$TRADERSPOST_WEBHOOK_URL" ]; then
    echo -e "${RED}❌ TRADERSPOST_WEBHOOK_URL not set in .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} TRADERSPOST_WEBHOOK_URL configured"

# 6. Check killswitch status
if [ "$POLICY_KILLSWITCH" = "true" ]; then
    echo -e "${YELLOW}⚠️  KILLSWITCH IS ENABLED - System will not place orders${NC}"
    echo -e "${YELLOW}   Set POLICY_KILLSWITCH=false in .env to enable trading${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Killswitch disabled (trading enabled)"
fi

# 7. Show portfolio configuration
echo ""
echo "📊 Portfolio Configuration:"
cat config/portfolio_optimization.json | grep -A 4 "portfolio_constraints" | tail -4
echo ""

# 8. Create logs directory
mkdir -p logs
echo -e "${GREEN}✓${NC} Logs directory ready"

echo ""
echo "================================================================================"
echo "✅ All pre-flight checks passed!"
echo "================================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Set environment
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export PINE_RUNTIME_CONFIG="$PROJECT_ROOT/config.yml"

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Create log file with timestamp
LOG_FILE="logs/portfolio_system_$(date +%Y%m%d_%H%M%S).log"

echo "🎯 Starting system..."
echo "📝 Logging to: $LOG_FILE"
echo ""
echo "================================================================================"
echo "🔴 SYSTEM STARTING - Watch for these key events:"
echo "================================================================================"
echo "1. Portfolio Coordinator initialization (max_positions=4)"
echo "2. Symbol filtering to 10 optimized symbols"
echo "3. ML models loading for all symbols"
echo "4. Databento connection established"
echo "5. Historical data loading"
echo "6. Live data stream starting"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================================"
echo ""

# Start the system with logging
python -m src.runner.main 2>&1 | tee "$LOG_FILE"
