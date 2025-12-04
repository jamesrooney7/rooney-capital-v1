# Rooney Capital - Monitoring Guide

**⚠️ NOTE:** This guide is maintained for quick reference. For the complete system guide, see **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**

**Current Configuration:** 9 instruments, 2 max positions, $2,500 daily stop loss

---

## Quick Status Check

### Check Service Status
```bash
sudo systemctl status pine-runner
```

### Verify All 9 Instruments Are Active
```bash
sudo journalctl -u pine-runner -n 100 | grep "Model loaded" | wc -l
```
Should return **9**

### View Live Logs (All Activity)
```bash
sudo journalctl -u pine-runner -f
```

### View Only Trading Signals
```bash
sudo journalctl -u pine-runner -f | grep -E "(LONG|SHORT|Position|Trade|FILL)"
```

### View Errors Only
```bash
sudo journalctl -u pine-runner -f | grep -i error
```

## Dashboard

### Start Dashboard
```bash
cd /opt/pine/rooney-capital-v1/dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then access at: `http://YOUR_SERVER_IP:8501`

### Check Current Trade Count
```bash
sqlite3 /opt/pine/runtime/trades.db "SELECT COUNT(*) FROM trades;"
```

### View Recent Trades
```bash
sqlite3 /opt/pine/runtime/trades.db "SELECT symbol, entry_time, exit_time, pnl FROM trades ORDER BY exit_time DESC LIMIT 10;"
```

### Daily P&L Summary
```bash
sqlite3 /opt/pine/runtime/trades.db "SELECT symbol, COUNT(*) as trades, SUM(pnl) as total_pnl FROM trades GROUP BY symbol;"
```

## Configuration Verification

### Verify 9 Instruments in Config
```bash
grep "symbols:" -A 12 /opt/pine/rooney-capital-v1/config.yml
```

Should list: 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI

### Check Portfolio Settings
```bash
grep -A 3 "portfolio:" /opt/pine/rooney-capital-v1/config.yml
```
Should show:
```yaml
portfolio:
  max_positions: 2
  daily_stop_loss: 2500
```

### Verify Active Instruments in Logs
```bash
sudo journalctl -u pine-runner -n 200 | grep "Model loaded"
```
Should show 9 instruments: 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI

## Performance Monitoring

### Check Position Coordinator Status
```bash
sudo journalctl -u pine-runner -n 100 | grep -i "portfolio"
```

### View Daily Stop Loss Status
```bash
sudo journalctl -u pine-runner -n 200 | grep -i "stop loss"
```

### Count Active Positions
```bash
sudo journalctl -u pine-runner -n 50 | grep "Current positions:"
```

## Troubleshooting

### Restart Service
```bash
sudo systemctl restart pine-runner
```

### View Last 200 Log Lines
```bash
sudo journalctl -u pine-runner -n 200
```

### Check for Duplicate Processes
```bash
ps aux | grep -E "python.*runner|python.*launch_worker" | grep -v grep
```
Should show **only ONE** process

### Verify No Old Services Running
```bash
sudo systemctl list-units | grep -i rooney
```
Should show **only** `pine-runner.service` (NOT rooney-trading.service)

## Current Configuration Targets

Based on train/test optimization (2023 train, 2024 test):

- **Train Sharpe Ratio**: 11.56 | **Test Sharpe Ratio**: 10.48
- **Test CAGR**: 90.74%
- **Test Max Drawdown**: $6,296
- **Generalization**: 90.7%
- **Instruments**: 9 (6 currencies, 1 energy, 2 metals)
  - Currencies: 6A, 6B, 6C, 6M, 6N, 6S
  - Energy: CL
  - Metals: HG, SI
- **Max Concurrent Positions**: 2
- **Daily Stop Loss**: $2,500

## Key Files

- **Config**: `/opt/pine/rooney-capital-v1/config.yml`
- **Contract Map**: `/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml`
- **Models**: `/opt/pine/rooney-capital-v1/src/models/`
- **Trades DB**: `/opt/pine/runtime/trades.db`
- **Service File**: `/etc/systemd/system/pine-runner.service`
- **Backups**: `/opt/pine/runtime/backups/`

## Daily Checklist

1. ✅ Verify service is running: `sudo systemctl status pine-runner`
2. ✅ Check all 9 instruments loaded: `sudo journalctl -u pine-runner -n 100 | grep "Model loaded" | wc -l` (should be 9)
3. ✅ Verify portfolio config: `sudo journalctl -u pine-runner -n 100 | grep "Portfolio config loaded from config.yml"` (should show max_positions=2)
4. ✅ Monitor for errors: `sudo journalctl -u pine-runner -n 200 | grep -i error`
5. ✅ Check position count: Max should be 2 concurrent
6. ✅ Review dashboard for daily P&L
7. ✅ Verify no duplicate processes: `ps aux | grep python | grep runner` (should be 1)

## Important Notes

- **DO NOT** modify model files in `/opt/pine/rooney-capital-v1/src/models/`
- **DO NOT** run multiple trading services simultaneously
- **ALWAYS** verify only one `pine-runner` process is active
- Dashboard resets and backups are stored in `/opt/pine/runtime/backups/`
- Configuration B is live trading - monitor closely during first few days
