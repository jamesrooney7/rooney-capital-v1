# Pine Trading Dashboard

Real-time monitoring dashboard for the Pine algorithmic trading system. Built with Streamlit for quick deployment and easy customization.

## Features

- **System Status**: Service uptime, heartbeat monitoring
- **ML Filter Statistics**: Pass rate, blocked trades, signal quality
- **Open Positions**: Current holdings with entry prices and times
- **Recent Trades**: Last 10 entry/exit executions with details
- **TradersPost Status**: Webhook delivery monitoring
- **Error Tracking**: Recent system errors and warnings
- **Auto-refresh**: Updates every 10 seconds

## Requirements

- Python 3.10+
- Access to system logs (requires running as user with journalctl permissions)
- Access to `/var/run/pine/worker_heartbeat.json`

## Installation

### 1. Install dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

Or install in a virtual environment:

```bash
cd dashboard
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify permissions

The dashboard needs to read:
- Systemd service status: `systemctl status pine-runner.service`
- Journalctl logs: `journalctl -u pine-runner.service`
- Heartbeat file: `/var/run/pine/worker_heartbeat.json`

If running as a non-root user, ensure your user is in the `systemd-journal` group:

```bash
sudo usermod -a -G systemd-journal $USER
```

Then log out and back in for the change to take effect.

## Usage

### Running locally

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Running on a server

For remote access, bind to all interfaces:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then access via `http://<server-ip>:8501`.

### Configuration options

Use the sidebar to adjust:
- **Log lines to fetch**: Number of recent log lines to parse (500-5000)
- **Time window**: Only show trades from last N minutes (30-720)

### Auto-refresh

The dashboard automatically refreshes every 10 seconds to show real-time updates.

## Dashboard Sections

### 1. System Status

- Service running status (green = running, red = down)
- Uptime duration
- Heartbeat status (active/unknown)
- Last heartbeat timestamp

### 2. Service Details

- **TradersPost Webhook**: Configuration status, last post time, total posts
- **Databento Feed**: Connection status, last data update

### 3. ML Filter Statistics

- **Total Signals**: Number of IBS signals detected
- **Passed**: Signals that passed ML threshold
- **Blocked**: Signals rejected by ML filter
- **Pass Rate**: Percentage of signals passing (color-coded: green ≥70%, yellow ≥40%, red <40%)

### 4. Open Positions

Shows currently held positions with:
- Symbol
- Size (quantity)
- Entry price
- Entry timestamp

### 5. Recent Trades

Last 10 trades (entries and exits) with:
- Symbol and action (BUY/SELL)
- Execution price and size
- Timestamp
- IBS value at entry/exit
- ML score (for entries)
- Exit reason (for exits)

### 6. Blocked Trades

Recent trades blocked by ML filter, showing:
- Symbol
- Timestamp
- ML score vs threshold

### 7. Recent Errors

Last 10 error messages from logs for troubleshooting.

## File Structure

```
dashboard/
├── app.py              # Main Streamlit application
├── utils.py            # Log parsing and data utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### Dashboard shows "Service Status: Down"

Check if the service is running:
```bash
sudo systemctl status pine-runner.service
```

If stopped, start it:
```bash
sudo systemctl start pine-runner.service
```

### No trades showing

1. Check the time window setting in the sidebar (increase from 60 to 360 minutes)
2. Verify logs contain trade data:
```bash
journalctl -u pine-runner.service -n 100 | grep -E "ENTRY|EXIT|FILLED"
```

### "Permission denied" errors

Ensure your user can read journalctl logs:
```bash
sudo usermod -a -G systemd-journal $USER
# Log out and back in
```

### Heartbeat shows "Unknown"

Check if the heartbeat file exists and is recent:
```bash
cat /var/run/pine/worker_heartbeat.json
```

If file doesn't exist, the service may not be writing heartbeats correctly.

## Future Enhancements

Potential features for Phase 2:

- **Charts**: IBS over time, ML score distribution, P&L curve
- **Alerts**: Email/SMS notifications for errors or key events
- **Database**: Persist trade history to SQLite/PostgreSQL
- **Advanced Analytics**: Sharpe ratio, win rate, average trade duration
- **Multi-timeframe view**: Show daily, weekly, monthly stats
- **Trade journal**: Add notes and tags to trades

## Support

For issues or questions:
1. Check logs: `journalctl -u pine-runner.service -f`
2. Verify service: `systemctl status pine-runner.service`
3. Review heartbeat: `cat /var/run/pine/worker_heartbeat.json`
4. Check this README for common troubleshooting steps
