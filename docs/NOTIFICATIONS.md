# Discord Notifications

The Rooney Capital trading system includes real-time Discord notifications for trade alerts, system monitoring, and daily performance summaries.

## Features

### Trade Alerts
- **Entry Notifications**: Instant alerts when trades are opened
  - Symbol and position size
  - Entry price
  - IBS value at entry
  - ML filter score
- **Exit Notifications**: Alerts when trades are closed
  - Entry and exit prices
  - P&L (dollars and percentage)
  - Exit reason (IBS exit, stop loss, etc.)
  - Trade duration

### System Alerts
- **Startup**: System startup confirmation with symbol list
- **Errors**: Critical error notifications with component details
- **Health Checks**: Periodic system health status (can be scheduled)

### Daily Summaries
- Total P&L for the day
- Number of trades executed
- Win rate percentage
- Best and worst trade
- Symbols traded

## Setup

### 1. Create Discord Webhook

1. Open your Discord server
2. Go to Server Settings → Integrations → Webhooks
3. Click "New Webhook"
4. Name it (e.g., "Rooney Trading Bot")
5. Select the channel where notifications should appear
6. Copy the webhook URL

### 2. Configure the System

Add the Discord webhook URL to your runtime configuration file:

**Option A: Add to config file (`/opt/pine/config/runtime.yml`)**

```yaml
discord_webhook_url: "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE"
```

**Option B: Set as environment variable**

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE"
```

### 3. Restart the Service

```bash
sudo systemctl restart pine-runner.service
```

You should immediately receive a "System Started" notification in your Discord channel.

## Notification Types

### Trade Entry (Green)
Sent when a new position is opened:
```
🟢 Trade Entry: ES

Symbol: ES
Side: LONG
Size: 1 contracts
Entry Price: $4523.50
IBS: 0.145
ML Score: 0.723
```

### Trade Exit
Color-coded by profitability (green for profit, red for loss):
```
✅ Trade Exit: ES

Symbol: ES
Side: LONG
Entry Price: $4523.50
Exit Price: $4528.75
P&L: $262.50 (+0.12%)
Exit Reason: IBS exit
IBS at Exit: 0.856
Duration: 2.3 hours
```

### Daily Summary
Sent at end of day (or manually triggered):
```
📊 Daily Summary - 2024-01-15

Total P&L: $1,250.00
Trades: 5
Win Rate: 80.0%
Best Trade: $450.00
Worst Trade: -$125.00
Symbols: ES, NQ, RTY
```

### System Alerts
Information, warnings, and errors:
```
ℹ️ System Started
Rooney Capital trading system started
Symbols: ES, NQ, RTY, YM

⚠️ Warning
High volatility detected

🚨 System Error
Component: ML Filter
Details: Model bundle not found
```

## Manual Daily Summary

To send a daily summary on demand:

```python
from runner.live_worker import LiveWorker

# Assuming you have a worker instance
worker.send_daily_summary()
```

Or via a separate script:

```python
from utils.discord_notifier import DiscordNotifier
from utils.trades_db import TradesDB
from datetime import datetime

# Initialize notifier
notifier = DiscordNotifier("YOUR_WEBHOOK_URL")

# Get today's trades from database
db = TradesDB()
today = datetime.now().date()
start = datetime.combine(today, datetime.min.time())
end = datetime.combine(today, datetime.max.time())
trades = db.get_trades_between(start, end)

# Calculate and send summary
if trades:
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100
    symbols = list(set(t["symbol"] for t in trades))

    notifier.send_daily_summary(
        total_pnl=total_pnl,
        num_trades=len(trades),
        win_rate=win_rate,
        best_trade=max(t["pnl"] for t in trades),
        worst_trade=min(t["pnl"] for t in trades),
        symbols_traded=symbols,
    )
```

## Scheduling Daily Summaries

To automatically send daily summaries at market close (e.g., 4 PM ET), add a cron job:

```bash
# Edit crontab
crontab -e

# Add line to send summary at 4 PM ET (adjust for your timezone)
0 16 * * 1-5 /opt/pine/venv/bin/python /opt/pine/scripts/send_daily_summary.py
```

Create `/opt/pine/scripts/send_daily_summary.py`:

```python
#!/usr/bin/env python
"""Send daily trading summary to Discord."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.discord_notifier import DiscordNotifier
from utils.trades_db import TradesDB
from datetime import datetime

def main():
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL not set")
        return 1

    notifier = DiscordNotifier(webhook_url)
    db = TradesDB()

    # Get today's trades
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())
    trades = db.get_trades_between(start, end)

    if not trades:
        print("No trades today")
        return 0

    # Calculate summary
    total_pnl = sum(t["pnl"] for t in trades)
    winning = [t for t in trades if t["pnl"] > 0]
    win_rate = (len(winning) / len(trades)) * 100
    symbols = list(set(t["symbol"] for t in trades))

    # Send notification
    success = notifier.send_daily_summary(
        total_pnl=total_pnl,
        num_trades=len(trades),
        win_rate=win_rate,
        best_trade=max(t["pnl"] for t in trades),
        worst_trade=min(t["pnl"] for t in trades),
        symbols_traded=symbols,
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

Make it executable:

```bash
chmod +x /opt/pine/scripts/send_daily_summary.py
```

## Customization

The Discord notifier supports custom notifications:

```python
from utils.discord_notifier import get_notifier

notifier = get_notifier()

# Custom alert
notifier.send_custom(
    title="Custom Alert",
    description="Something important happened",
    fields=[
        {"name": "Detail 1", "value": "Value 1", "inline": True},
        {"name": "Detail 2", "value": "Value 2", "inline": True},
    ],
    color=16776960,  # Orange
)

# Simple text message
notifier.send_custom(content="Quick update: All systems operational")
```

## Troubleshooting

### No notifications appearing

1. **Verify webhook URL is correct**:
   ```bash
   # Check config
   cat /opt/pine/config/runtime.yml | grep discord

   # Or check environment variable
   echo $DISCORD_WEBHOOK_URL
   ```

2. **Check service logs**:
   ```bash
   journalctl -u pine-runner.service -n 100 | grep -i discord
   ```

   Should see:
   ```
   Discord notifier initialized
   System Started notification sent
   ```

3. **Test webhook manually**:
   ```bash
   curl -X POST "YOUR_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"content": "Test message"}'
   ```

### Webhook returns 404

The webhook URL has been deleted or is invalid. Create a new webhook in Discord and update your configuration.

### Too many notifications

You can temporarily disable notifications by removing the `discord_webhook_url` from config and restarting:

```bash
# Comment out in config file
sudo nano /opt/pine/config/runtime.yml
# Add # before discord_webhook_url line

# Restart
sudo systemctl restart pine-runner.service
```

## Best Practices

1. **Separate Channels**: Create different channels for different notification types
   - `#trades` for entry/exit notifications
   - `#system-alerts` for errors and health checks
   - `#daily-summaries` for end-of-day reports

2. **Webhook Security**: Keep webhook URLs private (they allow posting to your Discord)

3. **Rate Limits**: Discord has rate limits on webhooks. The notifier includes basic error handling, but avoid excessive custom notifications.

4. **Mobile Notifications**: Configure Discord mobile app notification settings to get instant alerts on your phone.

## API Reference

### DiscordNotifier

```python
from utils.discord_notifier import DiscordNotifier

notifier = DiscordNotifier(webhook_url)
```

**Methods:**

- `send_trade_entry(symbol, side, price, size, ibs, ml_score)` - Trade entry notification
- `send_trade_exit(symbol, side, entry_price, exit_price, size, pnl, pnl_percent, exit_reason, ibs, duration_hours)` - Trade exit notification
- `send_daily_summary(total_pnl, num_trades, win_rate, best_trade, worst_trade, symbols_traded, date)` - Daily performance summary
- `send_system_alert(title, message, alert_type, fields)` - System alert (info/warning/error)
- `send_error(error_message, component, details)` - Error notification
- `send_health_check(status, uptime, services, recent_activity)` - Health check notification
- `send_custom(content, title, description, fields, color)` - Custom notification

See `src/utils/discord_notifier.py` for full documentation.
