#!/bin/bash
###############################################################################
# Full Strategy Optimization Workflow
#
# This script orchestrates the complete base strategy parameter optimization:
# 1. Walk-forward optimization (5 windows)
# 2. Parameter stability analysis
# 3. Held-out validation (2021-2024)
# 4. Final approval decision
#
# Usage:
#   ./research/run_full_strategy_optimization.sh ES
#   ./research/run_full_strategy_optimization.sh ES --data-dir /path/to/data
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if symbol provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Symbol required${NC}"
    echo "Usage: $0 SYMBOL [--data-dir DIR]"
    echo "Example: $0 ES"
    exit 1
fi

SYMBOL=$1
shift  # Remove symbol from arguments

# Parse optional arguments
DATA_DIR="data/resampled"
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  BASE STRATEGY PARAMETER OPTIMIZATION                      ║${NC}"
echo -e "${BLUE}║                              Symbol: ${SYMBOL}                                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

START_TIME=$(date +%s)

###############################################################################
# STEP 1: Walk-Forward Optimization
###############################################################################

echo -e "${GREEN}[1/4] Running Walk-Forward Optimization...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python research/optimize_base_strategy_params.py \
    --symbol "$SYMBOL" \
    --data-dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Walk-forward optimization failed!${NC}"
    echo -e "${YELLOW}  This usually means insufficient trade volume.${NC}"
    echo -e "${YELLOW}  Consider adjusting parameter ranges to be more permissive.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Walk-forward optimization complete${NC}"
echo ""

###############################################################################
# STEP 2: Parameter Stability Analysis
###############################################################################

echo -e "${GREEN}[2/4] Running Parameter Stability Analysis...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python research/analyze_strategy_stability.py \
    --symbol "$SYMBOL"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Stability analysis failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Stability analysis complete${NC}"
echo ""

###############################################################################
# STEP 3: Held-Out Validation
###############################################################################

echo -e "${GREEN}[3/4] Running Held-Out Validation (2021-2024)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python research/validate_strategy_heldout.py \
    --symbol "$SYMBOL" \
    --data-dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Held-out validation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Held-out validation complete${NC}"
echo ""

###############################################################################
# STEP 4: Final Approval Decision
###############################################################################

echo -e "${GREEN}[4/4] Making Final Approval Decision...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python research/finalize_strategy_decision.py \
    --symbol "$SYMBOL"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Final decision failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Final decision complete${NC}"
echo ""

###############################################################################
# COMPLETION SUMMARY
###############################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         OPTIMIZATION COMPLETE!                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Symbol:${NC} $SYMBOL"
echo -e "${GREEN}Duration:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "${GREEN}Results:${NC} optimization_results/$SYMBOL/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  ${BLUE}1.${NC} Review final decision: ${GREEN}optimization_results/$SYMBOL/reports/final_approval_decision.json${NC}"
echo -e "  ${BLUE}2.${NC} Check config file: ${GREEN}config/strategy_params.json${NC}"
echo -e "  ${BLUE}3.${NC} If approved, run data extraction:"
echo -e "     ${GREEN}python research/extract_training_data.py --symbol $SYMBOL${NC}"
echo -e "  ${BLUE}4.${NC} Train ML model:"
echo -e "     ${GREEN}python research/train_rf_three_way_split.py --symbol $SYMBOL${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
