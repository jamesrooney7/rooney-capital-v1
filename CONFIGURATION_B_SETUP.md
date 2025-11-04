# Configuration B Setup Guide

## Overview

Configuration B is the optimized portfolio configuration validated on out-of-sample data (2022-2024).

## Configuration Details

**Instruments (16 total):**
- **Currencies (7):** 6A, 6B, 6C, 6E, 6M, 6N, 6S
- **Equities (4):** ES, NQ, RTY, YM
- **Commodities (5):** CL, GC, HG, NG, PL

**Portfolio Constraints:**
- Max positions: 2 (conservative risk management)
- Daily stop loss: $2,500

**Performance (Out-of-Sample 2022-2024):**
- Sharpe Ratio: 7.88
- CAGR: 67.4%
- Max Drawdown: -0.61%

## File Structure (Single Source of Truth)

```
/opt/pine/rooney-capital-v1/
├── config.yml                          # Runtime config (16 instruments)
├── .env                                # Environment variables
├── config/
│   └── portfolio_optimization.json     # Portfolio optimization results
└── scripts/
    ├── launch_worker.py                # Worker launcher (used by systemd)
    ├── reset_dashboard.py              # Dashboard reset tool
    └── cleanup_server.sh               # Server cleanup script
```

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
cat /opt/pine/rooney-capital-v1/config.yml | grep -A 20 "symbols"

# Should show all 16: 6A, 6B, 6C, 6E, 6M, 6N, 6S, CL, ES, GC, HG, NG, NQ, PL, RTY, YM
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
