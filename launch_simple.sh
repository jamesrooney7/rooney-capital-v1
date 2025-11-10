#!/bin/bash
# Simple Launch Script - No Portfolio Optimization Required
# Just runs the basic multi-alpha trading system

set -e

PROJECT_ROOT="/opt/pine/rooney-capital-v1"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "🚀 LAUNCHING MULTI-ALPHA TRADING SYSTEM"
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

# 2. Check runtime config
if [ ! -f "config.yml" ]; then
    echo -e "${RED}❌ Runtime config not found: config.yml${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Runtime config exists"

# 3. Check .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Environment file not found: .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Environment file exists"

# 4. Check required environment variables
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

# 5. Check killswitch status
if [ "$POLICY_KILLSWITCH" = "true" ]; then
    echo -e "${YELLOW}⚠️  KILLSWITCH IS ENABLED - System will not place orders${NC}"
else
    echo -e "${GREEN}✓${NC} Killswitch disabled (trading enabled)"
fi

# 6. Create logs directory
mkdir -p logs
echo -e "${GREEN}✓${NC} Logs directory ready"

# 7. Check DEBUG logging enabled
if grep -q "level=logging.DEBUG" src/runner/main.py; then
    echo -e "${GREEN}✓${NC} DEBUG logging enabled (IBS and ML filters will be logged)"
else
    echo -e "${YELLOW}⚠️${NC} INFO logging only (change to DEBUG in src/runner/main.py for detailed logs)"
fi

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
LOG_FILE="logs/system_$(date +%Y%m%d_%H%M%S).log"

echo "🎯 Starting system..."
echo "📝 Logging to: $LOG_FILE"
echo ""
echo "================================================================================"
echo "🔴 WATCH FOR THESE KEY EVENTS:"
echo "================================================================================"
echo "1. ML models loading for all symbols"
echo "2. Databento connection established"
echo "3. Historical data loading"
echo "4. Live data stream starting"
echo "5. IBS calculations (hourly): 'ES | Bar XXXX | IBS: 0.XXX'"
echo "6. ML filter scores (hourly): '🤖 ES ML HOURLY | Score: 0.XXX | Passed: True/False'"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================================"
echo ""

# Start the system with logging
python -m src.runner.main 2>&1 | tee "$LOG_FILE"
