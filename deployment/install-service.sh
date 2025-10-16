#!/bin/bash
# Installation script for rooney-trading systemd service

set -e

echo "Installing Rooney Capital Trading Service..."

# Create log directory
sudo mkdir -p /var/log/rooney
sudo chown linuxuser:linuxuser /var/log/rooney
echo "✓ Created log directory"

# Copy service file
sudo cp deployment/rooney-trading.service /etc/systemd/system/
echo "✓ Installed service file"

# Reload systemd
sudo systemctl daemon-reload
echo "✓ Reloaded systemd"

# Enable service to start on boot
sudo systemctl enable rooney-trading
echo "✓ Enabled service"

echo ""
echo "Installation complete!"
echo ""
echo "Management commands:"
echo "  Start:   sudo systemctl start rooney-trading"
echo "  Stop:    sudo systemctl stop rooney-trading"
echo "  Restart: sudo systemctl restart rooney-trading"
echo "  Status:  sudo systemctl status rooney-trading"
echo "  Logs:    tail -f /var/log/rooney/rooney-trading.log"
