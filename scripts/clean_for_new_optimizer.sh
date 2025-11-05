#!/bin/bash
# Clean old data before deploying new optimizer configuration
# This script backs up and removes old trade data and state files

set -e

echo "=== Cleaning old optimizer data ==="

# Create backup directory with timestamp
BACKUP_DIR="/opt/pine/backups/pre-optimizer-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Created backup directory: $BACKUP_DIR"

# 1. Backup and clean trade database
if [ -f "/opt/pine/runtime/trades.db" ]; then
    echo "Backing up trades.db..."
    cp /opt/pine/runtime/trades.db "$BACKUP_DIR/trades.db"
    echo "Removing old trades.db..."
    rm /opt/pine/runtime/trades.db
    echo "✓ Trade database cleaned"
else
    echo "No trades.db found (already clean)"
fi

# 2. Clean heartbeat file
if [ -f "/var/run/pine/worker_heartbeat.json" ]; then
    echo "Backing up heartbeat file..."
    cp /var/run/pine/worker_heartbeat.json "$BACKUP_DIR/worker_heartbeat.json"
    echo "Removing old heartbeat file..."
    rm /var/run/pine/worker_heartbeat.json
    echo "✓ Heartbeat file cleaned"
else
    echo "No heartbeat file found (already clean)"
fi

# 3. Clean any old position state files (if they exist)
if [ -d "/opt/pine/runtime" ]; then
    if [ -n "$(find /opt/pine/runtime -name '*.state' -o -name '*.pkl' 2>/dev/null)" ]; then
        echo "Backing up state files..."
        find /opt/pine/runtime -name '*.state' -o -name '*.pkl' -exec cp {} "$BACKUP_DIR/" \;
        echo "Removing old state files..."
        find /opt/pine/runtime -name '*.state' -o -name '*.pkl' -delete
        echo "✓ State files cleaned"
    fi
fi

echo ""
echo "=== Cleanup Complete ==="
echo "Backup location: $BACKUP_DIR"
echo ""
echo "Summary of cleaned items:"
ls -lh "$BACKUP_DIR/"
echo ""
echo "You can now deploy the new configuration with:"
echo "  sudo systemctl restart pine-runner.service"
