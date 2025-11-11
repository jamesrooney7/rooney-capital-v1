# Current Portfolio Configuration

**⚠️ NOTE:** This guide is deprecated. For the complete and up-to-date system guide, see **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**

---

## Overview

Current optimized portfolio configuration validated with train/test split (2023 train, 2024 test).

## Configuration Details

**Instruments (9 total):**
- **Currencies (6):** 6A (AUD), 6B (GBP), 6C (CAD), 6M (MXN), 6N (NZD), 6S (CHF)
- **Energy (1):** CL (Crude Oil)
- **Metals (2):** HG (Copper), SI (Silver)

**Removed from previous:** ES, NQ, RTY, YM, GC, NG, PL, 6E

**Portfolio Constraints:**
- Max positions: 2
- Daily stop loss: $2,500

**Performance (Train/Test Split):**
- Train Sharpe (2023): 11.56 | Test Sharpe (2024): 10.48
- Test CAGR: 90.74%
- Test Max Drawdown: $6,296
- Generalization: 90.7%

## File Structure (Single Source of Truth)

**IMPORTANT:** Configuration has been unified into `config.yml` only!

```
/opt/pine/rooney-capital-v1/
├── config.yml                          # 🔴 PRIMARY CONFIG - All settings here!
│                                       #    (symbols, portfolio, credentials)
├── .env                                # Environment variables (API keys)
├── Data/
│   └── Databento_contract_map.yml     # Contract specifications
└── scripts/
    ├── launch_worker.py                # Worker launcher (used by systemd)
    ├── clean_for_new_optimizer.sh      # Cleanup script before deployment
    └── reset_dashboard.py              # Dashboard reset tool
```

**Note:** `config/portfolio_optimization.json` has been **REMOVED**. All configuration is now in `config.yml`.

## Systemd Service

**Service:** `pine-runner.service`
- **Location:** `/etc/systemd/system/pine-runner.service`
- **Config file:** `/opt/pine/rooney-capital-v1/config.yml`
- **Env file:** `/opt/pine/rooney-capital-v1/.env`
- **Status:** Enabled (auto-starts on boot)

## Common Commands

### Service Management
```bash
# Start trading system
sudo systemctl start pine-runner

# Stop trading system
sudo systemctl stop pine-runner

# Restart trading system
sudo systemctl restart pine-runner

# View status
sudo systemctl status pine-runner

# View live logs
sudo journalctl -u pine-runner -f
```

### Verify Configuration
```bash
# Check which instruments are configured
cat /opt/pine/rooney-capital-v1/config.yml | grep -A 12 "symbols:"

# Should show 9 instruments: 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI

# Check portfolio settings
cat /opt/pine/rooney-capital-v1/config.yml | grep -A 3 "portfolio:"

# Should show:
# portfolio:
#   max_positions: 2
#   daily_stop_loss: 2500
```

### Dashboard Management
```bash
# Reset dashboard (backup old trades, start fresh)
cd /opt/pine/rooney-capital-v1
python3 scripts/reset_dashboard.py

# Start Streamlit dashboard
cd dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Check Running Processes
```bash
# Verify only ONE trading process is running
ps aux | grep -E "src.runner.main|launch_worker" | grep -v grep

# Should see only ONE process: launch_worker.py
```

## Cleanup (One-Time Setup)

If migrating from old configuration:

```bash
cd /opt/pine/rooney-capital-v1
./scripts/cleanup_server.sh
```

This script will:
1. ✅ Archive old config files from `/opt/pine/runtime/`
2. ✅ Remove duplicate `rooney-trading.service`
3. ✅ Update `pine-runner.service` to use correct paths
4. ✅ Update `.gitignore` for ML models and analysis files

## Troubleshooting

### Issue: Only 11-12 instruments loading instead of 16

**Cause:** Service using old config from `/opt/pine/runtime/config.yml`

**Fix:**
```bash
# Verify service is using correct config
sudo systemctl cat pine-runner | grep PINE_RUNTIME_CONFIG

# Should show: /opt/pine/rooney-capital-v1/config.yml
# If not, run cleanup script or manually update service
```

### Issue: Duplicate trading processes

**Cause:** Multiple services running or orphaned processes

**Fix:**
```bash
# Stop all services
sudo systemctl stop pine-runner
sudo systemctl stop rooney-trading 2>/dev/null || true

# Kill orphaned processes
pkill -f "src.runner.main"
pkill -f "launch_worker"

# Start only pine-runner
sudo systemctl start pine-runner
```

### Issue: Permission errors for heartbeat file

**Cause:** `/var/run/pine/` owned by wrong user

**Fix:**
```bash
sudo chown -R linuxuser:linuxuser /var/run/pine/
sudo systemctl restart pine-runner
```

## Migration from Configuration A

Configuration A (old):
- 10 instruments
- Max 4 positions
- Higher Sharpe (16.23) but only tested on 2023-2024
- Potentially overfit

Configuration B (current):
- 16 instruments (better diversification)
- Max 2 positions (more conservative)
- Lower Sharpe (7.88) but validated on out-of-sample 2022-2024
- More robust for live trading

## Important Notes

1. **Slippage is included** in portfolio simulator results
2. **Test period Sharpe (7.88)** is more realistic than train period (10.55)
3. **Max 2 positions** prevents over-concentration
4. **16 instruments** provide diversification across asset classes
5. **Configuration B was generated on Nov 3, 2024** from portfolio optimizer

## Contact

For issues, check logs:
```bash
sudo journalctl -u pine-runner -n 100 --no-pager
```
