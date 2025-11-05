#!/bin/bash
# Install Streamlit dashboard as a systemd service

set -e

echo "=== Installing Rooney Dashboard Service ==="

# Get the current user
CURRENT_USER=$(whoami)

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run this script as root. Run as your normal user and it will use sudo when needed."
    exit 1
fi

# Path to service file template
SERVICE_TEMPLATE="deploy/systemd/rooney-dashboard.service"
SERVICE_NAME="rooney-dashboard.service"
SERVICE_DEST="/etc/systemd/system/$SERVICE_NAME"

if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo "❌ Service template not found: $SERVICE_TEMPLATE"
    exit 1
fi

# Create temporary service file with user substituted
TMP_SERVICE=$(mktemp)
sed "s/__USER__/$CURRENT_USER/g" "$SERVICE_TEMPLATE" > "$TMP_SERVICE"

echo "📋 Installing service file to $SERVICE_DEST"
sudo cp "$TMP_SERVICE" "$SERVICE_DEST"
rm "$TMP_SERVICE"

# Set proper permissions
sudo chmod 644 "$SERVICE_DEST"

echo "🔄 Reloading systemd daemon"
sudo systemctl daemon-reload

echo "✅ Service installed successfully!"
echo ""
echo "To manage the dashboard service:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart: sudo systemctl restart $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable $SERVICE_NAME"
echo ""
echo "Dashboard will be accessible at: http://YOUR_SERVER_IP:8501"
