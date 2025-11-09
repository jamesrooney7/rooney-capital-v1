# Multi-Alpha System - Quick Start Guide

This guide will help you get the multi-alpha trading system running on your server.

---

## Prerequisites

- Ubuntu/Debian Linux server
- Python 3.10+
- Redis server
- Databento API key
- TradersPost webhook URLs (optional, for live trading)

---

## Step 1: Install Dependencies

### 1.1 Install System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv redis-server git
```

### 1.2 Verify Redis is Running

```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
redis-cli ping  # Should return "PONG"
```

### 1.3 Install Python Dependencies

```bash
cd /opt/pine/rooney-capital-v1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2: Configure Environment Variables

### 2.1 Create `.env` File

```bash
cd /opt/pine/rooney-capital-v1
cp .env.example .env
nano .env  # or vim .env
```

### 2.2 Fill in Your Credentials

```bash
# ============================================================================
# Databento Configuration
# ============================================================================
DATABENTO_API_KEY=your_actual_databento_api_key_here

# ============================================================================
# TradersPost Configuration (optional for paper trading)
# ============================================================================
TRADERSPOST_IBS_A_WEBHOOK=https://your-traderspost-webhook-url-ibs-a
TRADERSPOST_IBS_B_WEBHOOK=https://your-traderspost-webhook-url-ibs-b
TRADERSPOST_BREAKOUT_WEBHOOK=https://your-traderspost-webhook-url-breakout
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter` in nano).

### 2.3 Protect the `.env` File

```bash
chmod 600 .env
```

---

## Step 3: Verify Configuration

### 3.1 Check Contract Map Exists

```bash
ls -la Data/Databento_contract_map.yml
```

If missing, you need to create it. See MULTI_ALPHA_ARCHITECTURE.md for details.

### 3.2 Check ML Models Exist

```bash
ls -la models/*.pkl
```

You should see files like:
- `6A_rf_model.pkl`
- `6C_rf_model.pkl`
- `CL_rf_model.pkl`
- etc.

### 3.3 Verify Configuration File

```bash
cat config.multi_alpha.yml | head -30
```

---

## Step 4: Load Environment Variables

**IMPORTANT**: You must export environment variables before running any component.

```bash
cd /opt/pine/rooney-capital-v1
export $(grep -v '^#' .env | xargs)
```

Verify it worked:
```bash
echo $DATABENTO_API_KEY  # Should show your API key
```

---

## Step 5: Start the Data Hub

The data hub connects to Databento and publishes market data to Redis.

### 5.1 Start in Foreground (for testing)

```bash
python3 -m src.data_hub.data_hub_main --config config.multi_alpha.yml
```

You should see:
```
INFO - Connected to Redis at localhost:6379
INFO - DataHub initialized for dataset=GLBX.MDP3 products=['ES.FUT', 'NQ.FUT', ...]
INFO - subscribing to trades:parent [...]
INFO - Data Hub running. Press Ctrl+C to stop.
```

### 5.2 Test Data Hub (optional)

In another terminal, watch Redis for published bars:

```bash
redis-cli
> SUBSCRIBE market:ES:1min
```

You should see bars being published every minute.

Press `Ctrl+C` to stop the data hub when testing is complete.

---

## Step 6: Start a Strategy Worker

Each strategy runs in its own process.

### 6.1 Load Environment Variables (in new terminal)

```bash
cd /opt/pine/rooney-capital-v1
export $(grep -v '^#' .env | xargs)
```

### 6.2 Start IBS A Strategy Worker

```bash
python3 -m src.runner.strategy_worker --strategy "ibs_a" --config config.multi_alpha.yml
```

You should see:
```
INFO - Starting strategy worker for 'ibs_a'...
INFO - Loaded ML model for 6A: 30 features, threshold=0.5
INFO - Loaded ML model for 6C: 19 features, threshold=0.55
...
INFO - Loading historical data for warmup...
INFO - Requesting daily warmup data from Databento (250 days)...
INFO - 6A: converted 250 daily bars, looking for feed '6A_day'
INFO - 6A: loaded 250 daily warmup bars
...
INFO - Historical warmup mode ENABLED on all strategies (fast mode)
INFO - Running Cerebro event loop...
INFO - Monitoring 57 feeds for warmup drain...
INFO - Warmup backlog drained - disabling warmup mode for full processing
```

The worker is now running and waiting for live bars from the data hub!

---

## Step 7: Verify System is Working

### 7.1 Check Redis Subscriptions

```bash
redis-cli client list | grep -c subscribe
```

Should show at least 2 (data hub publishing + strategy worker subscribing).

### 7.2 Check Logs

Data hub should show:
```
INFO - Publishing bar to market:ES:1min
INFO - Publishing bar to market:NQ:1min
...
```

Strategy worker should show:
```
INFO - Received bar for 6A: 2025-11-09 20:15:00
INFO - Received bar for 6C: 2025-11-09 20:15:00
...
```

### 7.3 Monitor Strategy Heartbeat

```bash
# If heartbeat file is configured
cat /tmp/heartbeat_ibs_a.json
```

Should show updated timestamp and broker values.

---

## Step 8: Running Multiple Strategies

You can run multiple strategy workers concurrently.

### Terminal 1: Data Hub
```bash
export $(grep -v '^#' .env | xargs)
python3 -m src.data_hub.data_hub_main --config config.multi_alpha.yml
```

### Terminal 2: IBS A Worker
```bash
export $(grep -v '^#' .env | xargs)
python3 -m src.runner.strategy_worker --strategy "ibs_a" --config config.multi_alpha.yml
```

### Terminal 3: IBS B Worker
```bash
export $(grep -v '^#' .env | xargs)
python3 -m src.runner.strategy_worker --strategy "ibs_b" --config config.multi_alpha.yml
```

All workers share the same data hub and can run simultaneously!

---

## Step 9: Production Deployment with systemd

For production, use systemd to manage services automatically.

### 9.1 Create Data Hub Service

```bash
sudo nano /etc/systemd/system/pine-datahub.service
```

```ini
[Unit]
Description=Pine Data Hub
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=linuxuser
WorkingDirectory=/opt/pine/rooney-capital-v1
EnvironmentFile=/opt/pine/rooney-capital-v1/.env
ExecStart=/opt/pine/rooney-capital-v1/venv/bin/python3 -m src.data_hub.data_hub_main --config config.multi_alpha.yml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 9.2 Create Strategy Worker Service

```bash
sudo nano /etc/systemd/system/pine-worker@.service
```

```ini
[Unit]
Description=Pine Strategy Worker (%i)
After=network.target redis.service pine-datahub.service
Requires=redis.service pine-datahub.service

[Service]
Type=simple
User=linuxuser
WorkingDirectory=/opt/pine/rooney-capital-v1
EnvironmentFile=/opt/pine/rooney-capital-v1/.env
ExecStart=/opt/pine/rooney-capital-v1/venv/bin/python3 -m src.runner.strategy_worker --strategy "%i" --config config.multi_alpha.yml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 9.3 Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start data hub
sudo systemctl enable pine-datahub.service
sudo systemctl start pine-datahub.service

# Start strategy workers
sudo systemctl enable pine-worker@ibs_a.service
sudo systemctl start pine-worker@ibs_a.service

sudo systemctl enable pine-worker@ibs_b.service
sudo systemctl start pine-worker@ibs_b.service
```

### 9.4 Check Service Status

```bash
# Check data hub
sudo systemctl status pine-datahub.service

# Check IBS A worker
sudo systemctl status pine-worker@ibs_a.service

# Check IBS B worker
sudo systemctl status pine-worker@ibs_b.service
```

### 9.5 View Logs

```bash
# Data hub logs
sudo journalctl -u pine-datahub.service -f

# IBS A worker logs
sudo journalctl -u pine-worker@ibs_a.service -f

# IBS B worker logs
sudo journalctl -u pine-worker@ibs_b.service -f
```

---

## Troubleshooting

### Issue: "DATABENTO_API_KEY not found"

**Solution**: Make sure you exported environment variables:
```bash
export $(grep -v '^#' .env | xargs)
```

Or for systemd services, ensure `EnvironmentFile` points to correct `.env` path.

---

### Issue: "Redis connection refused"

**Solution**: Start Redis:
```bash
sudo systemctl start redis-server
redis-cli ping  # Should return PONG
```

---

### Issue: "Authentication failed: CRAM reply bucket ID malformed"

**Solution**: Your Databento API key is empty or invalid. Check:
```bash
echo $DATABENTO_API_KEY  # Should show your key
```

If empty, re-export from `.env`:
```bash
export $(grep -v '^#' .env | xargs)
```

---

### Issue: "No module named 'databento'"

**Solution**: Install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Issue: Data hub running but strategy worker not receiving bars

**Solution**:
1. Check Redis subscriptions: `redis-cli client list`
2. Verify data hub is publishing: `redis-cli SUBSCRIBE market:ES:1min`
3. Check strategy worker logs for subscription messages
4. Ensure both are using same Redis host:port

---

### Issue: "IndexError: array index out of range" in safe_div.py

**Solution**: This is fixed in commit e82ff38. Make sure you have the latest code:
```bash
git pull origin claude/multi-alpha-trading-system-011CUxeA5kyA32m5JSnv24vG
```

---

## Next Steps

1. **Monitor the system** for at least 1 trading session
2. **Verify trades** are being sent to TradersPost (if enabled)
3. **Add more strategies** by creating new sections in `config.multi_alpha.yml`
4. **Set up monitoring** with health checks and alerts
5. **Review logs** daily for errors or warnings

---

## Support

- **Architecture**: See MULTI_ALPHA_ARCHITECTURE.md
- **Migration Status**: See MIGRATION_STATUS.md
- **Legacy System**: See README.md and SYSTEM_GUIDE.md
- **Issues**: Check GitHub issues or create a new one

---

## Safety Reminders

⚠️ **Always paper trade first!** Set `POLICY_KILLSWITCH=true` in production.

⚠️ **Never commit `.env` to git!** It's already in `.gitignore`.

⚠️ **Monitor daily**: Check logs, positions, and TradersPost daily.

⚠️ **Have a backup plan**: Know how to manually close positions if needed.
