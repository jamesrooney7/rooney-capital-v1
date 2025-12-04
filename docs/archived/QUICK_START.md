# 🚀 QUICK START - Launch Portfolio System NOW

## Step 1: Pull Latest Code

```bash
cd /opt/pine/rooney-capital-v1
git pull origin main
```

## Step 2: Verify Environment

**Check your webhook is pointing to PAPER account:**
```bash
cat .env | grep TRADERSPOST_WEBHOOK_URL
# Should point to paper/sim account, NOT live money
```

**Check killswitch is OFF:**
```bash
cat .env | grep KILLSWITCH
# Should be: POLICY_KILLSWITCH=false
```

## Step 3: Launch System

**Open Terminal 1 - System Launch:**
```bash
cd /opt/pine/rooney-capital-v1
./launch_portfolio_system.sh
```

This will:
- ✅ Run all pre-flight checks
- ✅ Show portfolio configuration
- ✅ Start the system with logging
- ✅ Display key events to watch for

## Step 4: Monitor (Optional)

**Open Terminal 2 - Real-Time Monitor:**
```bash
cd /opt/pine/rooney-capital-v1
./monitor_portfolio.sh
```

This shows a live dashboard with:
- Portfolio coordinator status
- Current open positions
- Orders placed
- Stop loss events
- System health

## What You Should See (First 5 Minutes)

### Terminal 1 Output:
```
✓ Virtual environment exists
✓ Portfolio config exists
✓ Runtime config exists
✓ Environment file exists
✓ DATABENTO_API_KEY configured
✓ TRADERSPOST_WEBHOOK_URL configured
✓ Killswitch disabled (trading enabled)

📊 Portfolio Configuration:
    "max_positions": 4,
    "daily_stop_loss": 2500.0,
    "symbols": ["6A", "6B", "6C", "6N", "6S", "CL", "ES", "PL", "RTY", "SI"],
    "n_symbols": 10

🎯 Starting system...

INFO - Initializing Portfolio Coordinator (max_positions=4, daily_stop_loss=$2,500)
INFO - Filtered to only optimized symbols: ['6A', '6B', '6C', '6N', '6S', 'CL', 'ES', 'PL', 'RTY', 'SI']
INFO - Loading ML model for ES...
INFO - Loading ML model for 6A...
[... more model loading ...]
INFO - Databento connection established
INFO - Loading historical data (252 days)...
INFO - Starting live data stream...
INFO - System ready - monitoring for setups
```

### Terminal 2 Dashboard:
```
================================================================================
📊 PORTFOLIO SYSTEM DASHBOARD
================================================================================
Time: 2024-12-09 18:30:45
✓ System is RUNNING

================================================================================
PORTFOLIO COORDINATOR STATUS
================================================================================
✓ Portfolio Coordinator initialized
  max_positions: 4
  daily_stop_loss: $2,500.00

Current Open Positions:
  No positions opened yet

Entry Blocks (max positions): 0

================================================================================
TRADING ACTIVITY
================================================================================
Orders Placed: 0 (Buy: 0, Sell: 0)

================================================================================
RISK MANAGEMENT
================================================================================
✓ No stop loss events

Daily P&L Updates:
  Daily P&L: $0.00

================================================================================
SYSTEM HEALTH
================================================================================
✓ No errors
✓ No warnings
```

## Key Things to Watch

### ✅ Good Signs:
- Portfolio Coordinator initializes with max_positions=4
- Only 10 symbols active (6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI)
- ML models load successfully
- Databento connection established
- Historical data loads
- Live stream starts

### ⚠️ Warning Signs:
- Errors during startup
- ML models failing to load
- Connection failures
- Missing portfolio config

### 🚨 Stop Immediately If:
- Portfolio Coordinator doesn't initialize
- System trades wrong symbols
- Errors prevent startup

## During Trading Hours

### When a Setup is Detected:
```
INFO - ES: IBS setup detected at 5850.00
INFO - ES: ML prediction: BUY (confidence: 0.85)
INFO - Portfolio Coordinator: Can open position (1/4 positions used)
INFO - ES: PLACING BUY ORDER at 5850.00
INFO - ES: Order filled at 5850.25
INFO - ES: Position opened, registered with portfolio coordinator (1/4 open)
```

### When Max Positions Reached:
```
INFO - CL: IBS setup detected at 72.50
INFO - CL: ML prediction: BUY (confidence: 0.78)
⛔ CL ENTRY BLOCKED BY PORTFOLIO: Max positions (4) reached
INFO - Current open: ES, RTY, 6A, SI
```

### When Daily Stop Loss Hits:
```
🚨 PORTFOLIO STOP LOSS TRIGGERED 🚨
INFO - Daily P&L: -$2,520.00
INFO - Closing all 3 open positions immediately
INFO - ES: Emergency exit at 5845.00
INFO - RTY: Emergency exit at 2020.50
INFO - SI: Emergency exit at 24.15
INFO - No new entries allowed until next trading day
```

## Manual Log Monitoring (Alternative to Dashboard)

```bash
# Watch all activity
tail -f logs/portfolio_system_*.log

# Watch portfolio events only
tail -f logs/portfolio_system_*.log | grep -i "portfolio"

# Watch orders only
tail -f logs/portfolio_system_*.log | grep -i "PLACING.*ORDER"

# Watch for problems
tail -f logs/portfolio_system_*.log | grep -iE "error|warning|blocked|stop loss"
```

## How to Stop

**Graceful shutdown:**
```bash
# In Terminal 1, press Ctrl+C
```

**Force stop:**
```bash
kill $(pgrep -f "runner.main")
```

**Emergency killswitch:**
```bash
# Edit .env
echo "POLICY_KILLSWITCH=true" >> .env

# Restart (will run but not place orders)
./launch_portfolio_system.sh
```

## Troubleshooting

### "Virtual environment not found"
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### "Portfolio config not found"
```bash
# Config should have been created
ls -la config/portfolio_optimization.json

# If missing, re-pull
git pull origin main
```

### "Import errors"
```bash
source venv/bin/activate
PYTHONPATH=src python3 -c "from runner.live_worker import LiveWorker; print('OK')"
```

### "Databento connection failed"
```bash
# Check API key
cat .env | grep DATABENTO_API_KEY

# Test connection
curl -u "$DATABENTO_API_KEY:" https://hist.databento.com/v0/metadata.list_datasets
```

## What's Running?

Your system now has:
- ✅ **Portfolio Coordinator** - Managing max 4 positions, $2,500 stop loss
- ✅ **10 Optimized Symbols** - Only trading 6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI
- ✅ **ML Filtering** - Every setup checked by random forest model
- ✅ **Real-time Data** - Databento live streaming
- ✅ **Paper Execution** - Orders sent to TradersPost → Paper broker

## Next Steps

1. **Watch for 24 hours** - Observe behavior during full trading day
2. **Check performance** - Compare to expected metrics (10-30% monthly)
3. **Verify constraints** - Ensure max positions and stop loss work
4. **Monitor logs** - Look for any errors or unusual behavior
5. **Track in spreadsheet** - Log daily trades, P&L, win rate

## Expected Daily Activity

- **Trades:** 1-4 per day (portfolio-level)
- **Positions:** 0-4 open at any time
- **Avg P&L:** $500-$1,500 per day
- **Stop loss:** Should rarely hit (maybe 1-2x per month)

## Remember

Your backtest showed:
- 📊 Sharpe 16.23
- 💰 +1,087% in 2 years
- 📉 Max DD -3.69%

Expect **30-50% degradation** in live results. Even half of backtest performance would be exceptional.

---

**System is designed, optimized, and ready. Time to see it work in real-time!** 🚀

Good luck! Watch closely and enjoy seeing your strategy come to life! 📈
