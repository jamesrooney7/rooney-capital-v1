#!/bin/bash
# Start all components: data hub + all enabled strategy workers
#
# Usage: ./scripts/start_all.sh [config_file]
#
# This script is for testing/development. In production, use supervisor/systemd.

set -e

CONFIG_FILE="${1:-config.multi_alpha.yml}"

echo "==================================="
echo " Starting Multi-Alpha Trading System"
echo "==================================="
echo "Config: $CONFIG_FILE"
echo ""

# Start data hub in background
echo "Starting Data Hub..."
./scripts/start_data_hub.sh "$CONFIG_FILE" &
DATA_HUB_PID=$!
echo "Data Hub PID: $DATA_HUB_PID"

# Wait a bit for data hub to initialize
sleep 3

# Start strategy workers
# NOTE: This reads from config to determine which strategies are enabled
# For now, we'll start ibs if it exists

echo ""
echo "Starting IBS Strategy Worker..."
./scripts/start_strategy_worker.sh ibs "$CONFIG_FILE" &
IBS_WORKER_PID=$!
echo "IBS Worker PID: $IBS_WORKER_PID"

# Add more strategies here as they're implemented
# echo ""
# echo "Starting Breakout Strategy Worker..."
# ./scripts/start_strategy_worker.sh breakout "$CONFIG_FILE" &
# BREAKOUT_WORKER_PID=$!
# echo "Breakout Worker PID: $BREAKOUT_WORKER_PID"

echo ""
echo "==================================="
echo " All processes started!"
echo "==================================="
echo ""
echo "Data Hub PID: $DATA_HUB_PID"
echo "IBS Worker PID: $IBS_WORKER_PID"
echo ""
echo "To stop all processes:"
echo "  kill $DATA_HUB_PID $IBS_WORKER_PID"
echo ""
echo "Logs are in /var/log/rooney/"
echo "Heartbeats are in /var/run/pine/"
echo ""

# Wait for all background processes
wait
