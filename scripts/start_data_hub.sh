#!/bin/bash
# Start the Data Hub process
#
# Usage: ./scripts/start_data_hub.sh [config_file]

set -e

CONFIG_FILE="${1:-config.multi_alpha.yml}"

echo "Starting Data Hub..."
echo "Config: $CONFIG_FILE"

# Ensure required environment variables are set
if [ -z "$DATABENTO_API_KEY" ]; then
    echo "ERROR: DATABENTO_API_KEY environment variable not set"
    exit 1
fi

# Start data hub
python -m src.data_hub.data_hub_main \
    --config "$CONFIG_FILE" \
    --log-level INFO
