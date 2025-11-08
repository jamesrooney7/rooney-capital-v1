#!/bin/bash
# Start a Strategy Worker process
#
# Usage: ./scripts/start_strategy_worker.sh <strategy_name> [config_file]
#
# Examples:
#   ./scripts/start_strategy_worker.sh ibs
#   ./scripts/start_strategy_worker.sh breakout config.multi_alpha.yml

set -e

if [ -z "$1" ]; then
    echo "ERROR: Strategy name required"
    echo "Usage: $0 <strategy_name> [config_file]"
    echo ""
    echo "Example: $0 ibs"
    exit 1
fi

STRATEGY_NAME="$1"
CONFIG_FILE="${2:-config.multi_alpha.yml}"

echo "Starting Strategy Worker: $STRATEGY_NAME"
echo "Config: $CONFIG_FILE"

# Ensure required environment variables are set
if [ -z "$DATABENTO_API_KEY" ]; then
    echo "ERROR: DATABENTO_API_KEY environment variable not set"
    exit 1
fi

# Check for strategy-specific webhook
WEBHOOK_VAR="TRADERSPOST_${STRATEGY_NAME^^}_WEBHOOK"
if [ -z "${!WEBHOOK_VAR}" ]; then
    echo "WARNING: $WEBHOOK_VAR environment variable not set"
    echo "Orders will not be sent to TradersPost"
fi

# Start strategy worker
python -m src.runner.strategy_worker \
    --strategy "$STRATEGY_NAME" \
    --config "$CONFIG_FILE" \
    --log-level INFO
