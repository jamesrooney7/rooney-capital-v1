# Configuration B - Monitoring Guide

## Quick Status Check

### Check Service Status
```bash
sudo systemctl status pine-runner
```

### Verify All 16 Instruments Are Active
```bash
sudo journalctl -u pine-runner -n 100 | grep "Model loaded" | wc -l
```
Should return **16**

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

### Verify 16 Instruments in Config
```bash
grep "symbols:" -A 20 /opt/pine/rooney-capital-v1/config.yml
```

### Check Portfolio Settings
```bash
grep -E "(max_positions|daily_stop_loss)" /opt/pine/rooney-capital-v1/config.yml
```
Should show:
- `max_positions: 2`
- `daily_stop_loss: 2500`

### Verify Active Instruments in Logs
```bash
sudo journalctl -u pine-runner -n 500 | grep "Trading symbols:"
```
Should list all 16: 6A, 6B, 6C, 6E, 6M, 6N, 6S, CL, ES, GC, HG, NG, NQ, PL, RTY, YM

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

## Configuration B Targets

Based on out-of-sample test results (2022-2024):

- **Test Sharpe Ratio**: 7.88
- **Test CAGR**: 67.4%
- **Instruments**: 16 (7 currencies, 4 equities, 5 commodities)
- **Max Concurrent Positions**: 2
- **Daily Stop Loss**: $2,500
- **Risk per Trade**: ~$200-500 per instrument

## Key Files

- **Config**: `/opt/pine/rooney-capital-v1/config.yml`
- **Contract Map**: `/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml`
- **Models**: `/opt/pine/rooney-capital-v1/src/models/`
- **Trades DB**: `/opt/pine/runtime/trades.db`
- **Service File**: `/etc/systemd/system/pine-runner.service`
- **Backups**: `/opt/pine/runtime/backups/`

## Daily Checklist

1. ✅ Verify service is running: `sudo systemctl status pine-runner`
2. ✅ Check all 16 instruments loaded: `sudo journalctl -u pine-runner -n 100 | grep "Model loaded" | wc -l`
3. ✅ Monitor for errors: `sudo journalctl -u pine-runner -n 200 | grep -i error`
4. ✅ Check position count: Max should be 2 concurrent
5. ✅ Review dashboard for daily P&L
6. ✅ Verify no duplicate processes: `ps aux | grep python | grep runner`

## Important Notes

- **DO NOT** modify model files in `/opt/pine/rooney-capital-v1/src/models/`
- **DO NOT** run multiple trading services simultaneously
- **ALWAYS** verify only one `pine-runner` process is active
- Dashboard resets and backups are stored in `/opt/pine/runtime/backups/`
- Configuration B is live trading - monitor closely during first few days
