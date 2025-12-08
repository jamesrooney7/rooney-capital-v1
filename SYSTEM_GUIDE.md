# Rooney Capital Trading System - Complete Guide

**Last Updated:** 2025-12-04
**Current Configuration:** 9 instruments, max 2 positions, $2,500 daily stop loss

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Current Configuration](#current-configuration)
3. [File Structure](#file-structure)
4. [Configuration Management](#configuration-management)
5. [Portfolio Optimization](#portfolio-optimization)
6. [Deployment Workflow](#deployment-workflow)
7. [Operations & Monitoring](#operations--monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Key Files Reference](#key-files-reference)
10. [Live Operations (Strategy Factory)](#live-operations-strategy-factory)

---

## System Overview

### What It Does

The Rooney Capital system is an automated futures trading platform that:
- **Ingests** live market data from Databento (CME Globex)
- **Evaluates** Internal Bar Strength (IBS) signals across multiple futures instruments
- **Filters** trades using trained Random Forest ML models
- **Manages** portfolio-level risk with position limits and daily stop loss
- **Routes** orders to TradersPost webhook for execution via Tradovate

### Architecture

```
Databento API → Live Data Bridge → Backtrader Engine → IBS Strategy + ML Veto → Portfolio Coordinator → TradersPost → Tradovate
                                                                                 ↓
                                                                          Discord Alerts
                                                                          SQLite Trades DB
                                                                          Dashboard
```

### Key Components

1. **Data Ingestion** (`src/runner/databento_bridge.py`)
   - Subscribes to Databento Live API
   - Aggregates ticks into 1-minute bars
   - Feeds data into Backtrader

2. **Strategy Logic** (`src/strategy/factory_adapter.py`)
   - Implements Strategy Factory patterns (ATRBuyDip, AvgHLRangeIBS, etc.)
   - Calculates technical indicators and features
   - Applies ML meta-labeling filter
   - Manages position entry/exit on 15-minute bars

3. **ML Models** (`models/factory/`)
   - ML meta-labeling models per symbol/strategy combination
   - Naming: `{SYMBOL}_{STRATEGY}_ml_meta_labeling_final_model.pkl`
   - Trained using walk-forward optimization
   - Filter signals with probability threshold (typically 0.5)

4. **Portfolio Coordinator** (`src/runner/portfolio_coordinator.py`)
   - Enforces max_positions limit (currently 2)
   - Tracks daily P&L
   - Triggers $2,500 daily stop loss
   - Prevents race conditions with atomic slot reservation

5. **Order Routing** (`src/runner/traderspost_client.py`)
   - Sends orders to TradersPost webhook
   - 1-hour GTD (Good Till Date) time-in-force
   - Includes symbol, action, quantity, price

6. **Notifications** (`src/utils/discord_notifier.py`)
   - Trade entries/exits
   - Daily summaries (P&L, win rate, profit factor)
   - Emergency alerts (stop loss hit)

7. **Trade Database** (`src/utils/trades_db.py`)
   - SQLite database at `/opt/pine/runtime/trades.db`
   - Stores all trade history
   - Used by dashboard for real-time display

---

## Current Configuration

### Instruments (9 Total)

| Category | Instruments |
|----------|-------------|
| **Currencies** (6) | 6A (AUD), 6B (GBP), 6C (CAD), 6M (MXN), 6N (NZD), 6S (CHF) |
| **Energy** (1) | CL (Crude Oil) |
| **Metals** (2) | HG (Copper), SI (Silver) |

**Removed from previous config:** ES, NQ, RTY, YM (equity indices), GC (gold), NG (natural gas), PL (platinum), 6E (euro)

### Portfolio Constraints

- **Max Positions:** 2 concurrent positions
- **Daily Stop Loss:** $2,500 (all positions exited, no new trades until next day)
- **Starting Capital:** $250,000 (paper trading)
- **Slippage:** 4 ticks round-trip per instrument (2 ticks per side)

### Performance (Train/Test Split)

| Metric | Train (2023) | Test (2024) |
|--------|--------------|-------------|
| **Sharpe Ratio** | 11.56 | 10.48 |
| **CAGR** | 94.87% | 90.74% |
| **Max Drawdown** | $3,249 | $6,296 |
| **Generalization** | - | 90.7% |

---

## File Structure

### Configuration Files (SINGLE SOURCE OF TRUTH)

```
/opt/pine/rooney-capital-v1/
├── config.yml                          # 🔴 PRIMARY CONFIG - EDIT THIS!
│                                       # Contains: symbols, portfolio settings, credentials
├── .env                                # Environment variables (API keys, webhooks)
├── Data/
│   └── Databento_contract_map.yml      # Contract specifications (tick sizes, datasets)
└── models/factory/
    ├── CL_ATRBuyDip_ml_meta_labeling_final_model.pkl      # ML model per symbol/strategy
    ├── CL_ATRBuyDip_ml_meta_labeling_executive_summary.txt
    ├── ES_AvgHLRangeIBS_ml_meta_labeling_final_model.pkl
    └── ... (14 models total)
```

**IMPORTANT:** `config/portfolio_optimization.json` has been **REMOVED**. All configuration is now in `config.yml`.

### Key Source Files

```
src/
├── runner/
│   ├── live_worker.py                 # Main orchestrator (loads config, starts strategies)
│   ├── portfolio_coordinator.py       # Portfolio-level risk management
│   ├── databento_bridge.py            # Live data ingestion from Databento
│   └── traderspost_client.py          # Order routing to TradersPost
├── strategy/
│   ├── factory_adapter.py             # Strategy Factory adapter for live trading
│   ├── ibs_strategy.py                # Legacy IBS strategy (deprecated)
│   └── contract_specs.py              # Tick sizes and values per instrument
├── models/
│   ├── factory_loader.py              # Strategy Factory model loading
│   └── loader.py                      # Legacy model loading utilities
└── utils/
    ├── discord_notifier.py            # Discord alerts and summaries
    └── trades_db.py                   # SQLite trade database

research/
├── portfolio_optimizer_greedy_train_test.py  # Portfolio optimizer with --update-config
├── portfolio_simulator.py             # Portfolio backtesting using actual trades
├── train_rf_three_way_split.py       # Legacy IBS model training (deprecated)
└── ml_meta_labeling/                  # Strategy Factory ML training pipeline

scripts/
├── launch_worker.py                   # Worker launcher (used by systemd)
└── clean_for_new_optimizer.sh         # Cleanup script before deploying new config

dashboard/
└── app.py                             # Streamlit dashboard for monitoring
```

### Runtime Files

```
/opt/pine/runtime/
├── trades.db                          # SQLite trade history
├── backups/                           # Dashboard backups
└── /var/run/pine/
    └── worker_heartbeat.json          # Heartbeat file (process health check)
```

---

## Configuration Management

### config.yml Structure

```yaml
# Core paths
contract_map: Data/Databento_contract_map.yml
models_path: models/factory  # Strategy Factory ML models

# Instruments to trade
symbols:
  - 6A
  - 6B
  - 6C
  - 6M
  - 6N
  - 6S
  - CL
  - HG
  - SI

# API credentials (use environment variables)
databento_api_key: ${DATABENTO_API_KEY}
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}
discord_webhook_url: ${DISCORD_WEBHOOK_URL}
killswitch: ${POLICY_KILLSWITCH}

# Portfolio management
portfolio:
  max_positions: 2
  daily_stop_loss: 2500

# Cash and runtime settings
starting_cash: 250000
load_historical_warmup: true
historical_lookback_days: 252
backfill: true
backfill_days: 4

# Per-instrument settings (optional)
contracts:
  6A:
    size: 1
  CL:
    size: 1
  # ... etc
```

### Environment Variables (.env)

```bash
DATABENTO_API_KEY=your_api_key_here
TRADERSPOST_WEBHOOK_URL=https://your-webhook-url
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
POLICY_KILLSWITCH=false
PINE_RUNTIME_CONFIG=/opt/pine/rooney-capital-v1/config.yml
```

---

## Portfolio Optimization

### Overview

The portfolio optimizer finds the best combination of instruments and max_positions to maximize Sharpe ratio while respecting drawdown constraints.

### Running the Optimizer

```bash
cd /opt/pine/rooney-capital-v1

# Run with auto-update flag (RECOMMENDED)
python3 research/portfolio_optimizer_greedy_train_test.py \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --daily-stop-loss 2500 \
    --update-config  # <-- Auto-updates config.yml!
```

**What it does:**

1. Loads existing trade data from `results/` directory (fast, no backtest re-running)
2. Trains on 2023 data: removes instruments one-by-one until max drawdown < $5,000
3. Tests multiple max_positions values (1-4)
4. Validates on 2024 data (out-of-sample)
5. **Creates backup:** `config_backup_TIMESTAMP.yml`
6. **Updates config.yml** with optimal symbols and max_positions
7. Saves results to: `results/greedy_optimization_TIMESTAMP.json`

### Features

- ✅ **Greedy instrument removal:** Removes worst performers until constraint met
- ✅ **Train/test split:** Avoids overfitting
- ✅ **4-tick slippage:** Realistic execution costs
- ✅ **$2,500 daily stop loss:** Applied to portfolio
- ✅ **Hourly position tracking:** Accurate overlapping position simulation
- ✅ **Breach events:** Counts how many times drawdown exceeded $6,000

### Output Example

```
====================================================================================
OPTIMIZATION COMPLETE
====================================================================================

🏆 Optimal: max_positions=2, 9 instruments
📈 Train Sharpe: 11.560 (DD: $3,249)
📉 Test Sharpe: 10.480 (DD: $6,296)
📊 Generalization: 90.7%

✅ Successfully updated config.yml
   - Symbols: ['6A', '6B', '6C', '6M', '6N', '6S', 'CL', 'HG', 'SI']
   - Max positions: 2
   - Daily stop loss: $2,500

🎉 Config updated! You can now deploy with:
   git add config.yml && git commit -m 'Update portfolio config'
   git push && sudo systemctl restart pine-runner.service
```

---

## Deployment Workflow

### Step 1: Pull Latest Changes

```bash
cd /opt/pine/rooney-capital-v1
git pull origin claude/instrument-filters-analysis-011CUnCr2DhCHJdJpanur45q
```

### Step 2: Verify Configuration

```bash
# Check symbols
cat config.yml | grep -A 10 "symbols:"

# Check portfolio settings
cat config.yml | grep -A 3 "portfolio:"

# Should show:
# portfolio:
#   max_positions: 2
#   daily_stop_loss: 2500
```

### Step 3: Stop Service & Clean Data

```bash
# Stop the trading service
sudo systemctl stop pine-runner.service

# Clean old trade data (creates backup first)
sudo bash scripts/clean_for_new_optimizer.sh
```

**What cleanup does:**
- Backs up trades.db to `/opt/pine/backups/pre-optimizer-TIMESTAMP/`
- Removes old trades.db
- Cleans heartbeat and state files

### Step 4: Restart Service

```bash
# Start with clean state
sudo systemctl start pine-runner.service

# Monitor startup logs
sudo journalctl -u pine-runner.service -f
```

### Step 5: Verify Successful Startup

Watch for these log messages:

```
✅ Portfolio config loaded from config.yml: max_positions=X, daily_stop_loss=$XXXX
✅ Portfolio coordinator initialized successfully
✅ Loaded N symbols from config
✅ Strategy Factory model loaded for CL_ATRBuyDip (threshold: 0.5)
... (should see model loaded for each portfolio instrument)
✅ Discord notifier initialized
✅ TradersPost client initialized
```

**Red flags to watch for:**

```
❌ No portfolio configuration in config.yml
❌ Failed to load model for [SYMBOL]
❌ No portfolio optimization config found  # (old JSON file reference)
```

---

## Operations & Monitoring

### Service Management

```bash
# Start
sudo systemctl start pine-runner.service

# Stop
sudo systemctl stop pine-runner.service

# Restart
sudo systemctl restart pine-runner.service

# Status
sudo systemctl status pine-runner.service

# View live logs
sudo journalctl -u pine-runner.service -f

# View only trades
sudo journalctl -u pine-runner.service -f | grep -E "(LONG|SHORT|FILL|Position)"

# View only errors
sudo journalctl -u pine-runner.service -f | grep -i error
```

### Dashboard

```bash
# Start dashboard
cd /opt/pine/rooney-capital-v1/dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Access at:
http://YOUR_SERVER_IP:8501
```

**Dashboard shows:**
- Current open positions (from database, not logs)
- Recent completed trades (last 20)
- Daily P&L summary
- Win rate and profit factor

### Trade Database Queries

```bash
# Total trade count
sqlite3 /opt/pine/runtime/trades.db "SELECT COUNT(*) FROM trades;"

# Recent trades
sqlite3 /opt/pine/runtime/trades.db \
  "SELECT symbol, entry_time, exit_time, pnl FROM trades ORDER BY exit_time DESC LIMIT 10;"

# P&L by symbol
sqlite3 /opt/pine/runtime/trades.db \
  "SELECT symbol, COUNT(*) as trades, SUM(pnl) as total_pnl FROM trades GROUP BY symbol;"

# Open positions
sqlite3 /opt/pine/runtime/trades.db \
  "SELECT symbol, entry_price, entry_time FROM trades WHERE exit_time IS NULL;"
```

### Discord Notifications

You'll receive:

1. **Trade Entry:** When position opens
   - Symbol, direction (LONG/SHORT), entry price

2. **Trade Exit:** When position closes
   - P&L, exit price, duration

3. **Daily Summary:** First trade of new day triggers summary for previous day
   - Total P&L
   - Number of trades
   - Win rate
   - Profit factor
   - Best/worst trades
   - Symbols traded

4. **Emergency Alerts:** When daily stop loss hits
   - Daily P&L
   - Open positions being closed
   - Time triggered

### Health Checks

```bash
# Verify only ONE process running
ps aux | grep -E "launch_worker|src.runner.main" | grep -v grep
# Should show exactly ONE process

# Check heartbeat file (updated every 30s)
cat /var/run/pine/worker_heartbeat.json
# Should show recent timestamp

# Verify models loaded
sudo journalctl -u pine-runner.service -n 200 | grep -i "model loaded" | wc -l
# Should match the number of portfolio instruments

# Check portfolio coordinator
sudo journalctl -u pine-runner.service -n 100 | grep -i "portfolio"
# Should show: "Portfolio config loaded from config.yml: max_positions=2..."
```

### Daily Checklist

1. ✅ **Service running:** `sudo systemctl status pine-runner`
2. ✅ **All models loaded:** Check logs for "model loaded" messages (one per portfolio instrument)
3. ✅ **No errors:** `sudo journalctl -u pine-runner -n 200 | grep -i error`
4. ✅ **Max 2 positions:** Check dashboard or Discord
5. ✅ **Dashboard accessible:** Open in browser
6. ✅ **Single process:** `ps aux | grep launch_worker | grep -v grep | wc -l` returns 1

---

## Troubleshooting

### Issue: Multiple Positions Opening (More Than 2)

**Symptom:** 3+ positions open simultaneously
**Cause:** Old `config/portfolio_optimization.json` file or config.yml not read correctly

**Fix:**
```bash
# Verify portfolio_optimization.json is deleted
ls config/portfolio_optimization.json
# Should return: No such file or directory

# Verify config.yml has portfolio section
cat config.yml | grep -A 3 "portfolio:"

# Restart service
sudo systemctl restart pine-runner.service

# Check logs
sudo journalctl -u pine-runner.service -n 50 | grep "Portfolio config loaded"
# Should see: "Portfolio config loaded from config.yml: max_positions=2"
```

### Issue: Service Won't Start

**Symptom:** `systemctl status pine-runner` shows failed
**Cause:** Configuration error or missing dependencies

**Fix:**
```bash
# Check detailed error
sudo journalctl -u pine-runner.service -n 50

# Common issues:
# 1. Missing .env file
ls /opt/pine/rooney-capital-v1/.env

# 2. Invalid config.yml syntax
python3 -c "import yaml; yaml.safe_load(open('config.yml'))"

# 3. Missing models
ls models/factory/*_ml_meta_labeling_final_model.pkl | wc -l
# Should return 14 (one per symbol/strategy combination)

# 4. Port already in use (if running multiple instances)
ps aux | grep launch_worker | grep -v grep
# Should show only ONE or ZERO
```

### Issue: No Models Loading

**Symptom:** Logs show "Failed to load model for [SYMBOL]"
**Cause:** Model files missing or corrupted

**Fix:**
```bash
# Check which models exist
ls -lh models/factory/*_ml_meta_labeling_final_model.pkl

# Verify model directory structure
# Should have: models/factory/{SYMBOL}_{STRATEGY}_ml_meta_labeling_final_model.pkl
# Example: models/factory/CL_ATRBuyDip_ml_meta_labeling_final_model.pkl

# Check model count (should be 14)
ls models/factory/*_ml_meta_labeling_final_model.pkl | wc -l

# If missing, check git status or re-pull from GitHub
git status models/factory/
```

### Issue: TradersPost Orders Not Appearing

**Symptom:** Webhooks send successfully but don't show in TradersPost
**Cause:** TradersPost strategy disabled or webhook URL wrong

**Fix:**
```bash
# 1. Verify webhook URL in .env
cat .env | grep TRADERSPOST_WEBHOOK_URL

# 2. Test webhook manually
curl -X POST $TRADERSPOST_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"ticker":"ES","action":"buy","quantity":1,"price":4500}'

# 3. Check TradersPost strategy is ENABLED in their UI
# (Log in to traderspost.io and verify strategy status)
```

### Issue: Dashboard Shows No Data

**Symptom:** Dashboard blank or shows "No trades"
**Cause:** trades.db empty or not accessible

**Fix:**
```bash
# Check database exists
ls -lh /opt/pine/runtime/trades.db

# Check trade count
sqlite3 /opt/pine/runtime/trades.db "SELECT COUNT(*) FROM trades;"

# If 0, system hasn't made any trades yet (wait for signals)
# Or database was recently cleaned

# Verify dashboard can access database
cd dashboard
python3 -c "import sqlite3; conn = sqlite3.connect('/opt/pine/runtime/trades.db'); print('OK')"
```

### Issue: High Memory Usage

**Symptom:** System slow, high memory consumption
**Cause:** Multiple processes or data accumulation

**Fix:**
```bash
# Check for multiple processes
ps aux | grep python | grep -E "launch_worker|runner"

# Kill duplicates if found
sudo systemctl stop pine-runner.service
sudo pkill -f launch_worker
sudo systemctl start pine-runner.service

# Check database size
ls -lh /opt/pine/runtime/trades.db

# If very large (>100MB), consider archiving old trades:
# (Backup first!)
cp /opt/pine/runtime/trades.db /opt/pine/backups/trades_backup_$(date +%Y%m%d).db
sqlite3 /opt/pine/runtime/trades.db "DELETE FROM trades WHERE entry_time < date('now', '-3 months');"
```

---

## Key Files Reference

### Configuration

| File | Purpose | Edit? |
|------|---------|-------|
| `config.yml` | Main configuration - symbols, portfolio settings, credentials | ✅ YES |
| `.env` | Environment variables (API keys, webhooks) | ✅ YES |
| `Data/Databento_contract_map.yml` | Contract specs (tick sizes, datasets) | ⚠️ RARELY |

### Source Code (Don't Edit Unless Development)

| File | Purpose |
|------|---------|
| `src/runner/live_worker.py` | Main orchestrator, loads config, starts strategies |
| `src/runner/portfolio_coordinator.py` | Portfolio risk management (max positions, daily stop) |
| `src/strategy/factory_adapter.py` | Strategy Factory trading logic (15-min bars) |
| `src/models/factory_loader.py` | Strategy Factory ML model loading |
| `src/runner/traderspost_client.py` | Order routing to TradersPost |
| `src/utils/discord_notifier.py` | Discord notifications |
| `models/factory/` | Production ML meta-labeling models (14 total) |

### Scripts (Operations)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `research/portfolio_optimizer_greedy_train_test.py` | Portfolio optimization with --update-config | After collecting new trade data |
| `scripts/clean_for_new_optimizer.sh` | Clean old trade data before deployment | Before deploying new config |
| `scripts/launch_worker.py` | Worker launcher (called by systemd) | Never directly (systemd uses it) |
| `dashboard/app.py` | Streamlit dashboard | Run manually for monitoring |

### Runtime Files (System-Generated)

| File | Purpose |
|------|---------|
| `/opt/pine/runtime/trades.db` | SQLite trade history |
| `/var/run/pine/worker_heartbeat.json` | Process health check |
| `/opt/pine/backups/` | Trade database backups |

### System Service

| Item | Location |
|------|----------|
| **Service file** | `/etc/systemd/system/pine-runner.service` |
| **Service name** | `pine-runner.service` |
| **Working directory** | `/opt/pine/rooney-capital-v1` |
| **Runs as user** | `root` |

---

## Quick Command Reference

```bash
# === SERVICE MANAGEMENT ===
sudo systemctl start pine-runner.service
sudo systemctl stop pine-runner.service
sudo systemctl restart pine-runner.service
sudo systemctl status pine-runner.service
sudo journalctl -u pine-runner.service -f

# === CONFIGURATION ===
cd /opt/pine/rooney-capital-v1
cat config.yml | grep -A 10 "symbols:"
cat config.yml | grep -A 3 "portfolio:"
cat .env

# === DEPLOYMENT ===
cd /opt/pine/rooney-capital-v1
cp config.yml /tmp/config.yml.backup && cp .env /tmp/.env.backup
git pull origin claude/show-strategy-results-01AeKC416rG5ZpWkoxbJmoxS
git lfs pull
cp /tmp/config.yml.backup config.yml && cp /tmp/.env.backup .env
sudo systemctl restart pine-runner.service

# === MONITORING ===
ps aux | grep "runner.main" | grep -v grep  # Should show ONE process
sqlite3 /opt/pine/runtime/trades.db "SELECT COUNT(*) FROM trades;"
sudo journalctl -u pine-runner.service | grep -i "model loaded" | wc -l

# === EMERGENCY STOP ===
sed -i 's/POLICY_KILLSWITCH=false/POLICY_KILLSWITCH=true/' /opt/pine/rooney-capital-v1/.env
sudo systemctl restart pine-runner.service

# === DASHBOARD ===
cd /opt/pine/rooney-capital-v1/dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# === PORTFOLIO OPTIMIZATION ===
cd /opt/pine/rooney-capital-v1
python3 research/portfolio_optimizer_greedy_train_test.py \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 --update-config
```

---

## Live Operations (Strategy Factory)

This section covers day-to-day operations for running the Strategy Factory live trading system.

### Server Details

| Item | Value |
|------|-------|
| **Hostname** | rooney-capital-v2 |
| **SSH User** | root |
| **Working Directory** | /opt/pine/rooney-capital-v1 |
| **Git Branch** | claude/show-strategy-results-01AeKC416rG5ZpWkoxbJmoxS |

### Quick Start: Deploy and Run

```bash
# 1. SSH to server
ssh root@your-server-ip

# 2. Navigate to project
cd /opt/pine/rooney-capital-v1

# 3. Pull latest changes (backup config first)
cp config.yml /tmp/config.yml.backup
cp .env /tmp/.env.backup
git pull origin claude/show-strategy-results-01AeKC416rG5ZpWkoxbJmoxS
git lfs pull  # Required for ML model files
cp /tmp/config.yml.backup config.yml
cp /tmp/.env.backup .env

# 4. Verify code is updated
git log --oneline -1

# 5. Restart the service
sudo systemctl restart pine-runner.service

# 6. Verify it's running
sudo systemctl status pine-runner.service
```

### Systemd Service Commands

```bash
# Start service
sudo systemctl start pine-runner.service

# Stop service
sudo systemctl stop pine-runner.service

# Restart service
sudo systemctl restart pine-runner.service

# Check status
sudo systemctl status pine-runner.service

# Enable on boot
sudo systemctl enable pine-runner.service
```

### Monitoring Logs

```bash
# Stream logs in real-time
sudo journalctl -u pine-runner.service -f

# View recent logs (last hour)
sudo journalctl -u pine-runner.service --since "1 hour ago"

# View logs since service start
sudo journalctl -u pine-runner.service -b

# View recent trade executions
sudo journalctl -u pine-runner.service | grep -E "BUY executed|SELL executed" | tail -20

# View entry signals (including ML-blocked ones)
sudo journalctl -u pine-runner.service | grep -E "ENTRY SIGNAL|Entry blocked" | tail -30

# View errors only
sudo journalctl -u pine-runner.service | grep -i "error" | tail -50

# View strategy bar processing
sudo journalctl -u pine-runner.service | grep "Bar.*signal=" | tail -20
```

### Checking Worker Status

```bash
# Check service status
sudo systemctl status pine-runner.service

# Check if process is running (shows PID and start time)
ps aux | grep "runner.main" | grep -v grep

# Example output:
# root     1629144  3.0  0.4 6950576 546836 ?  Sl  14:37  0:04 python3 -m runner.main
#                                                          ^^^^^^ start time

# Check heartbeat file
cat /var/run/pine/worker_heartbeat.json
```

### Force Killing a Stuck Worker

If `sudo systemctl stop` doesn't work:

```bash
# 1. Find the PID
ps aux | grep "runner.main" | grep -v grep

# 2. Force kill by PID
sudo kill -9 <PID>

# 3. Or force kill all runner processes
sudo pkill -9 -f "runner.main"

# 4. Verify it's dead
ps aux | grep "runner.main" | grep -v grep
# (should show nothing)

# 5. Start fresh
sudo systemctl start pine-runner.service
```

### Full Redeploy Workflow

When you need to update code and restart:

```bash
# 1. Stop current worker
sudo systemctl stop pine-runner.service

# 2. If that doesn't work, force kill
sudo pkill -9 -f "runner.main"

# 3. Backup config files
cp config.yml /tmp/config.yml.backup
cp .env /tmp/.env.backup

# 4. Pull latest code
git pull origin claude/show-strategy-results-01AeKC416rG5ZpWkoxbJmoxS
git lfs pull

# 5. Restore config files
cp /tmp/config.yml.backup config.yml
cp /tmp/.env.backup .env

# 6. Verify the update
git log --oneline -1

# 7. Start fresh
sudo systemctl start pine-runner.service

# 8. Monitor logs for errors
sudo journalctl -u pine-runner.service -f
```

### Emergency Stop (Killswitch)

To immediately stop all trading:

```bash
# Enable killswitch
sed -i 's/POLICY_KILLSWITCH=false/POLICY_KILLSWITCH=true/' /opt/pine/rooney-capital-v1/.env
sudo systemctl restart pine-runner.service

# To re-enable trading
sed -i 's/POLICY_KILLSWITCH=true/POLICY_KILLSWITCH=false/' /opt/pine/rooney-capital-v1/.env
sudo systemctl restart pine-runner.service
```

### Common Issues and Solutions

#### Issue: "Failed to resolve symbol: CLZ5" (or similar)

**Cause:** Contract has expired. CL/HG/NG contracts expire around the 20th of the month BEFORE the contract month.

**Solution:** The contract selector should auto-select the correct contract. If not:
1. Check `src/runner/contract_selector.py` has latest fixes
2. Pull latest code and restart

#### Issue: Worker shows old start date after restart

**Cause:** Old process wasn't killed before starting new one.

**Solution:**
```bash
# Force kill ALL runner processes
sudo pkill -9 -f "runner.main"
# Start fresh
sudo systemctl start pine-runner.service
```

#### Issue: No diagnostic output in logs

**Cause:** Running old code that doesn't have diagnostic logging.

**Solution:**
1. Pull latest code: `git pull origin claude/show-strategy-results-01AeKC416rG5ZpWkoxbJmoxS`
2. Run `git lfs pull` to ensure ML models are present
3. Restart: `sudo systemctl restart pine-runner.service`

#### Issue: Orders filling at wrong prices

**Diagnostic steps:**
1. Check the diagnostic logs for order execution:
   ```bash
   sudo journalctl -u pine-runner.service | grep "executed at" | tail -10
   ```
2. Compare executed price vs bar data
3. If executed price matches Bar open instead of previous bar close, there may be a timing issue

### Log Access

Logs are managed by journalctl (systemd):

```bash
# Stream live logs
sudo journalctl -u pine-runner.service -f

# View last 100 lines
sudo journalctl -u pine-runner.service -n 100

# View logs since specific time
sudo journalctl -u pine-runner.service --since "2 hours ago"

# Export logs to file
sudo journalctl -u pine-runner.service --since today > /tmp/today.log
```

### Diagnostic Log Format

Strategy bar processing logs show:
```
ES: Bar 1159 @ 2025-12-08 16:45:00 | signal=False, last_10_signals=0, close=6853.25, ML_features=10/10
```

This shows:
- **Bar number** and **timestamp**
- **Signal status** (True/False)
- **Recent signals** count
- **Close price**
- **ML features** availability (e.g., 10/10 = all features calculated)

---

## Next Steps

1. **Monitor first few days:** Watch for any issues with new 9-instrument portfolio
2. **Verify TradersPost integration:** Ensure orders route correctly with 1-hour GTD
3. **Check Discord summaries:** Daily reports should show portfolio metrics
4. **Quarterly re-optimization:** Run optimizer every 3 months with new trade data
5. **Review performance:** Compare live results vs backtest expectations (Sharpe ~8-10)

---

## Support & Resources

- **GitHub Issues:** Report bugs or request features
- **Documentation:** This file (SYSTEM_GUIDE.md) is the single source of truth
- **Configuration:** All settings in `config.yml` (no more JSON files!)
- **Model Training:** See `research/ml_meta_labeling/` for Strategy Factory models
- **Portfolio Optimization:** See `research/portfolio_optimizer_greedy_train_test.py`

**Last Updated:** 2025-12-04
**Configuration:** 9 instruments, 2 max positions, $2,500 daily stop
