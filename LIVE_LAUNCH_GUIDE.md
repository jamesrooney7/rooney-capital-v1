# 🚀 LIVE LAUNCH GUIDE - Portfolio Optimization System

## ⚠️ CRITICAL: PAPER TRADING FIRST

**DO NOT GO LIVE WITH REAL MONEY IMMEDIATELY**

This guide walks you through launching in **PAPER TRADING MODE** first. You must observe for **minimum 2-4 weeks** before considering live trading.

---

## 📋 Pre-Flight Checklist

### Step 1: Pull Latest Code

```bash
cd /opt/pine/rooney-capital-v1
git pull origin main

# Verify you're on main with latest commits
git log --oneline -5
```

**Expected:** You should see the portfolio optimization commits (PortfolioCoordinator, export_portfolio_config, etc.)

### Step 2: Verify Configuration Files

```bash
# Check portfolio config exists
cat config/portfolio_optimization.json

# Should show:
# - max_positions: 4
# - daily_stop_loss: 2500.0
# - symbols: [6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI]
```

**IMPORTANT:** The config.yml currently has MORE symbols than the optimized portfolio. The PortfolioCoordinator will automatically filter to only trade the 10 optimized symbols.

### Step 3: Set Paper Trading Mode

```bash
# Check your .env file
cat .env | grep KILLSWITCH
cat .env | grep TRADERSPOST
```

**For Paper Trading:**
- `POLICY_KILLSWITCH=false` (allow trading)
- `TRADERSPOST_WEBHOOK_URL` should point to your **paper trading broker account**

**Verify TradersPost Settings:**
1. Log into TradersPost
2. Ensure your broker connection is in **PAPER/SIM mode**
3. Copy the webhook URL for paper account
4. Update `.env` with paper trading webhook

### Step 4: Verify Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Test imports
PYTHONPATH=src python3 -c "from runner.portfolio_coordinator import PortfolioCoordinator; print('✓ PortfolioCoordinator OK')"
PYTHONPATH=src python3 -c "from runner.live_worker import LiveWorker; print('✓ LiveWorker OK')"

# Both should print OK without errors
```

### Step 5: Verify ML Models Exist

```bash
# Check that you have trained models for all 10 symbols
ls -la src/models/*/rf_model.pkl

# Should see models for: 6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI
```

### Step 6: Verify Historical Data Access

```bash
# Test Databento API key
cat .env | grep DATABENTO_API_KEY

# Verify contract map exists
ls -la Data/Databento_contract_map.yml
```

---

## 🎯 Launch Commands

### Option A: Manual Start (Recommended for First Launch)

```bash
cd /opt/pine/rooney-capital-v1
source venv/bin/activate

# Set PYTHONPATH and run
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PINE_RUNTIME_CONFIG="$PWD/config.yml"

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start the worker with logging
python -m src.runner.main 2>&1 | tee logs/launch_$(date +%Y%m%d_%H%M%S).log
```

**What to Watch For:**
1. ✅ Preflight checks pass
2. ✅ ML models load for all 10 symbols
3. ✅ Databento connection established
4. ✅ Historical data loads (252 days × 10 symbols)
5. ✅ Portfolio Coordinator initializes with max_positions=4
6. ✅ Symbol list filtered to 10 optimized symbols
7. ✅ Live data stream starts
8. ⚠️ Check for any errors or warnings

### Option B: Using Start Script

```bash
cd /opt/pine/rooney-capital-v1

# Use the existing start script
./scripts/start.sh
```

### Option C: Systemd Service (Production)

```bash
# Start service
sudo systemctl start rooney-trading

# Check status
sudo systemctl status rooney-trading

# View logs
sudo journalctl -u rooney-trading -f
```

---

## 📊 Monitoring Commands

### Real-Time Log Monitoring

```bash
# Follow main logs
tail -f logs/worker_YYYYMMDD.log

# Filter for portfolio coordinator events
tail -f logs/worker_YYYYMMDD.log | grep -i "portfolio"

# Filter for trade entries
tail -f logs/worker_YYYYMMDD.log | grep -i "PLACING.*ORDER"

# Filter for position management
tail -f logs/worker_YYYYMMDD.log | grep -E "(ENTRY BLOCKED|Max positions|stopped out)"
```

### Health Checks

```bash
# Check heartbeat (system is alive)
cat /var/run/pine/worker_heartbeat.json
watch -n 5 cat /var/run/pine/worker_heartbeat.json

# Check process is running
ps aux | grep "runner.main"

# Check system resources
top -p $(pgrep -f "runner.main")
```

### Portfolio Status Checks

```bash
# Check how many positions are open (from logs)
tail -f logs/worker_*.log | grep -i "n_positions"

# Check daily P&L tracking
tail -f logs/worker_*.log | grep -i "daily_pnl"

# Check for stop loss events
tail -f logs/worker_*.log | grep -i "STOP LOSS"
```

### Discord Notifications

If you have Discord webhook configured, you'll receive:
- Trade notifications (entries/exits)
- Daily P&L updates
- **🚨 CRITICAL: Portfolio stop loss alerts**
- System errors and warnings

---

## 🎯 Expected Behavior - First 24 Hours

### During Market Hours (6pm - 5pm CT)

**Position Opening:**
```
17:30 CT - ES: Setup detected, ML passed
17:30 CT - Portfolio Coordinator: Can open position (0/4 positions used)
17:30 CT - ES: PLACING BUY ORDER at 5850.00
17:32 CT - ES: Position opened, registered with portfolio coordinator (1/4 open)
```

**Position Rejected (Max Positions):**
```
09:15 CT - CL: Setup detected, ML passed
09:15 CT - ⛔ CL ENTRY BLOCKED BY PORTFOLIO: Max positions (4) reached
09:15 CT - Current open: ES, RTY, 6A, SI
```

**Daily Stop Loss Hit:**
```
14:20 CT - 🚨 PORTFOLIO STOP LOSS TRIGGERED 🚨
14:20 CT - Daily P&L: -$2,520.00
14:20 CT - Closing all 3 open positions immediately
14:20 CT - ES: Emergency exit at 5845.00 (loss: $-250)
14:20 CT - RTY: Emergency exit at 2020.50 (loss: $-100)
14:20 CT - SI: Emergency exit at 24.15 (loss: $-200)
14:20 CT - No new entries allowed until next trading day
```

**Daily Reset:**
```
18:00 CT - New trading day: 2024-12-10
18:00 CT - Portfolio Coordinator: Daily P&L reset to $0.00
18:00 CT - Portfolio Coordinator: Stop loss status cleared
18:00 CT - Ready for new positions (0/4 open)
```

### What Numbers to Expect (Paper Trading)

Based on backtest performance with realistic degradation:

**Daily:**
- **Trades per day:** 1-4 trades (portfolio-level)
- **Avg daily P&L:** $500 - $1,500 (vs $1,492 in backtest)
- **Win rate:** 55-65% of days (vs 64% in backtest)
- **Max daily loss:** $1,000 - $2,500 (stop loss at $2,500)

**Weekly:**
- **Avg weekly return:** 3-8% (vs 10%+ in backtest)
- **Positive weeks:** 60-75% (vs 100% in backtest)

**Monthly:**
- **Expected return:** 10-20% (vs 50%+ in backtest 2024)
- **Max drawdown:** $5,000 - $15,000 (vs $10,957 in backtest)

**Red Flags:**
- ⚠️ Daily losses exceeding $3,000 consistently
- ⚠️ Win rate below 45% after 20+ trades
- ⚠️ Drawdown exceeding $20,000
- ⚠️ Multiple stop loss hits per week (>2)
- ⚠️ Position coordinator errors or deadlocks

---

## 🔍 Key Monitoring Points

### 1. Portfolio Constraints Working?

**Test Case:** Wait until 4 positions are open

**Expected Behavior:**
```
✅ System rejects new entries with message: "Max positions (4) reached"
✅ Existing positions continue to be monitored
✅ New entries allowed only after a position closes
```

**How to Verify:**
```bash
# Watch for these log patterns
tail -f logs/worker_*.log | grep -E "(ENTRY BLOCKED|Max positions)"
```

### 2. Daily Stop Loss Working?

**Test Case:** If daily losses approach -$2,500

**Expected Behavior:**
```
✅ All positions closed immediately when -$2,500 hit
✅ Discord alert sent: "🚨 PORTFOLIO STOP LOSS HIT"
✅ No new entries allowed for rest of day
✅ Resets at 6pm CT (start of next session)
```

**How to Verify:**
```bash
# Watch for stop loss events
tail -f logs/worker_*.log | grep -i "STOP LOSS"
```

### 3. Symbol Filtering Working?

**Test Case:** Check which symbols are being traded

**Expected Behavior:**
```
✅ Only 10 symbols active: 6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI
✅ Other symbols (6E, 6M, GC, HG, NQ, YM, NG) are filtered out
```

**How to Verify:**
```bash
# Check startup logs for symbol filtering
grep "filtered_symbols\|Filtered to only" logs/worker_*.log
```

### 4. ML Models Working?

**Test Case:** Every potential entry should have ML prediction

**Expected Behavior:**
```
✅ Setup detected → ML prediction made → Pass/fail decision
✅ Logs show: "ML Features: [...]" and "ML Prediction: BUY/PASS"
❌ No entries without ML check
```

**How to Verify:**
```bash
# All entries should have ML prediction
grep "PLACING.*ORDER" logs/worker_*.log -B 5 | grep "ML"
```

---

## 📈 Performance Tracking

### Daily Checklist

**Every trading day, log:**
1. Number of trades executed
2. Win rate (wins / total trades)
3. Daily P&L
4. Largest win / loss
5. Max open positions
6. Any stop loss hits
7. Any errors or warnings

**Template:**
```
Date: 2024-12-09
Trades: 3 (2 wins, 1 loss)
Win Rate: 66.7%
Daily P&L: +$1,240
Largest Win: +$620 (ES)
Largest Loss: -$180 (CL)
Max Open: 3 positions
Stop Loss: None
Errors: None
Notes: Strong day, no issues
```

### Weekly Review

**Every Sunday:**
1. Calculate weekly return %
2. Compare to backtest expectations
3. Review any unusual behavior
4. Check drawdown vs backtest
5. Assess whether performance is within expected range

### Red Alert Thresholds

**Stop paper trading and investigate if:**
- ❌ **Weekly loss >10%** (suggests major issue)
- ❌ **Drawdown >$25,000** (2.5x backtest max DD)
- ❌ **Win rate <40%** after 30+ trades
- ❌ **Daily stop loss hit 3+ times in one week**
- ❌ **Portfolio coordinator failures** (positions not tracked)
- ❌ **Execution errors >5%** of orders

---

## 🛑 How to Stop the System

### Graceful Shutdown

```bash
# Find the process
ps aux | grep "runner.main"

# Send SIGTERM (graceful)
kill $(pgrep -f "runner.main")

# Or use systemctl
sudo systemctl stop rooney-trading
```

### Emergency Killswitch

**If you need to stop trading immediately:**

```bash
# Set killswitch in .env
echo "POLICY_KILLSWITCH=true" >> .env

# Restart the worker (will not place new trades)
sudo systemctl restart rooney-trading
```

**Or manually:**
1. Edit `.env`: Set `POLICY_KILLSWITCH=true`
2. Restart the system
3. System will run but not place orders

---

## ✅ Go/No-Go Decision Checklist

**After 2-4 weeks of paper trading, evaluate:**

### Go Live If:
- ✅ **No critical errors** during entire paper trading period
- ✅ **Win rate 50-70%** (realistic range)
- ✅ **Avg return 10-30% monthly** (conservative but profitable)
- ✅ **Max DD < $20,000** (under 2x backtest)
- ✅ **Portfolio constraints working perfectly** (max positions, stop loss)
- ✅ **Execution quality good** (fills near expected prices)
- ✅ **System stability 100%** (no crashes or hangs)

### Do NOT Go Live If:
- ❌ Any critical errors or system failures
- ❌ Win rate < 45%
- ❌ Weekly losses consistently
- ❌ Drawdown > $25,000
- ❌ Portfolio constraints not working
- ❌ Daily stop loss not triggering correctly
- ❌ Position tracking issues
- ❌ You don't understand the results

---

## 🚀 Transition to Live Trading

**When you're ready (after successful paper trading):**

### Step 1: Update Broker Settings

1. Log into TradersPost
2. **Switch to LIVE broker account** (not paper)
3. Copy new webhook URL
4. Update `.env`:
   ```bash
   TRADERSPOST_WEBHOOK_URL=https://traderspost.io/webhooks/YOUR-LIVE-WEBHOOK
   ```

### Step 2: Reduce Position Sizing (Optional)

Start with **50% capital allocation**:

Edit `config.yml`:
```yaml
starting_cash: 125000  # Half of $250k
```

This reduces risk while you validate live execution.

### Step 3: Final Checks

```bash
# Verify live webhook
cat .env | grep TRADERSPOST

# Verify killswitch is OFF
cat .env | grep KILLSWITCH
# Should be: POLICY_KILLSWITCH=false

# Test connection to TradersPost
curl -X POST "$TRADERSPOST_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"test": "ping"}'
```

### Step 4: Launch Live

```bash
cd /opt/pine/rooney-capital-v1
sudo systemctl restart rooney-trading

# Watch logs very carefully for first hour
sudo journalctl -u rooney-trading -f
```

### Step 5: Monitor Closely

**First week of live trading:**
- Check logs multiple times per day
- Verify every trade in your broker
- Compare P&L between system and broker
- Watch for execution slippage
- Be ready to kill switch if needed

---

## 📞 Emergency Contacts

**If something goes wrong:**

1. **Stop the system** (see "How to Stop" section)
2. **Enable killswitch** (POLICY_KILLSWITCH=true)
3. **Close open positions manually** in your broker
4. **Review logs** to understand what happened
5. **Don't restart until issue is identified and fixed**

---

## 📚 Additional Resources

- **Integration Guide:** `PORTFOLIO_INTEGRATION_GUIDE.md`
- **Performance Analysis:** Run `research/portfolio_performance_analysis.py`
- **Code Review:** See `src/runner/portfolio_coordinator.py` for constraint logic
- **TradersPost Docs:** https://traderspost.io/docs
- **Databento Docs:** https://docs.databento.com

---

## 🎯 Summary: Your Launch Steps

```bash
# 1. Pull code
cd /opt/pine/rooney-capital-v1
git pull origin main

# 2. Verify config
cat config/portfolio_optimization.json
cat .env | grep -E "KILLSWITCH|TRADERSPOST"

# 3. Test imports
source venv/bin/activate
PYTHONPATH=src python3 -c "from runner.live_worker import LiveWorker; print('✓')"

# 4. Start system (paper trading)
./scripts/start.sh

# 5. Monitor logs
tail -f logs/worker_*.log | grep -E "(PORTFOLIO|STOP LOSS|Max positions)"

# 6. Watch for 2-4 weeks

# 7. Evaluate performance vs expectations

# 8. Go live (if all checks pass) or investigate issues
```

---

**Remember:** Your backtest showed exceptional results (Sharpe 16.23, 100% monthly win rate). **Expect 30-50% degradation** in live trading. If you achieve even **half** of the backtest performance, you'll have an outstanding strategy.

**Good luck, and trade carefully!** 🚀📈
