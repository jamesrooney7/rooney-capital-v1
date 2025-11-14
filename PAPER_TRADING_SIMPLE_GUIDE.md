# 🎯 PAPER TRADING SETUP - ES, NQ, RTY, YM (Simplified)

This guide sets up paper trading for your 4 trained models, skipping portfolio optimization.

---

## ✅ Pre-Flight Checklist

### 1. Verify Your Trained Models

```bash
cd /opt/pine/rooney-capital-v1

# Check models exist
ls -lh src/models/ES_rf_model.pkl
ls -lh src/models/NQ_rf_model.pkl
ls -lh src/models/RTY_rf_model.pkl
ls -lh src/models/YM_rf_model.pkl

# All 4 should exist from your recent training
```

### 2. Create/Update config.yml

```bash
# Copy example config
cp config.example.yml config.yml

# Edit it
nano config.yml
```

**Update these sections:**

```yaml
# Add YM to symbols list
symbols:
  - ES
  - NQ
  - RTY
  - YM  # <-- ADD THIS

# Add YM contract specs (after RTY section)
contracts:
  ES:
    size: 1
    commission: 4.00
    margin: 13200
    multiplier: 50
  NQ:
    size: 1
    commission: 4.20
    margin: 15400
    multiplier: 20
  RTY:
    size: 2
    commission: 4.50
    margin: 8800
    multiplier: 50
  YM:  # <-- ADD THIS
    size: 1
    commission: 4.00
    margin: 11000
    multiplier: 5
```

### 3. Set Up Environment Variables

```bash
# Create .env file if it doesn't exist
cp .env.example .env

# Edit .env
nano .env
```

**Required variables:**

```bash
# Databento API key (for market data)
DATABENTO_API_KEY=your_databento_key_here

# TradersPost webhook (PAPER TRADING URL!)
TRADERSPOST_WEBHOOK_URL=https://traderspost.io/webhooks/YOUR_PAPER_WEBHOOK

# Killswitch (set to false to allow trading)
POLICY_KILLSWITCH=false

# Optional: Discord notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK
```

**CRITICAL:** Make sure `TRADERSPOST_WEBHOOK_URL` points to your **PAPER/SIM account**, NOT live!

### 4. Verify TradersPost Paper Account

1. Log into https://traderspost.io
2. Go to Broker Connections
3. Ensure you have a **PAPER/SIMULATION** account connected
4. Copy the webhook URL for that paper account
5. Paste it into `.env` as `TRADERSPOST_WEBHOOK_URL`

### 5. Test Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Test imports
python3 -c "from runner.live_worker import LiveWorker; print('✅ LiveWorker OK')"
python3 -c "from models.loader import load_model_bundle; print('✅ Model loader OK')"

# Test model loading
python3 -c "from models.loader import load_model_bundle; load_model_bundle('ES', 'src/models'); print('✅ ES model loads')"
python3 -c "from models.loader import load_model_bundle; load_model_bundle('NQ', 'src/models'); print('✅ NQ model loads')"
python3 -c "from models.loader import load_model_bundle; load_model_bundle('RTY', 'src/models'); print('✅ RTY model loads')"
python3 -c "from models.loader import load_model_bundle; load_model_bundle('YM', 'src/models'); print('✅ YM model loads')"
```

All should print ✅ without errors.

---

## 🚀 Launch Paper Trading

### Option A: Manual Start (Recommended for First Time)

```bash
cd /opt/pine/rooney-capital-v1
source venv/bin/activate

# Set environment
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PINE_RUNTIME_CONFIG="$PWD/config.yml"

# Load .env variables
set -a && source .env && set +a

# Create logs directory
mkdir -p logs

# Start with live logging
python -m runner.main 2>&1 | tee logs/paper_trading_$(date +%Y%m%d_%H%M%S).log
```

**What You Should See:**

```
[INFO] ============================================================
[INFO] ROONEY CAPITAL TRADING SYSTEM - PREFLIGHT CHECKS
[INFO] ============================================================
[INFO] ✓ Config loaded from config.yml
[INFO] ✓ Databento API key found
[INFO] ✓ TradersPost webhook configured
[INFO] ✓ Killswitch: DISABLED (trading allowed)
[INFO] ✓ Symbols to trade: ES, NQ, RTY, YM
[INFO] ✓ ML model loaded: ES (30 features, threshold=0.50)
[INFO] ✓ ML model loaded: NQ (30 features, threshold=0.50)
[INFO] ✓ ML model loaded: RTY (30 features, threshold=0.50)
[INFO] ✓ ML model loaded: YM (30 features, threshold=0.50)
[INFO] ✓ Databento connection successful
[INFO] ✓ TradersPost webhook test successful
[INFO]
[INFO] ============================================================
[INFO] ALL PREFLIGHT CHECKS PASSED - STARTING LIVE TRADING
[INFO] ============================================================
[INFO] Loading historical data (252 days x 4 symbols)...
[INFO] Subscribing to live data streams...
[INFO] System ready - monitoring for signals...
```

### Option B: Background with nohup

```bash
cd /opt/pine/rooney-capital-v1
source venv/bin/activate

# Set environment
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PINE_RUNTIME_CONFIG="$PWD/config.yml"

# Load .env
set -a && source .env && set +a

# Start in background
nohup python -m runner.main > logs/paper_trading.log 2>&1 &

# Get PID
echo $! > logs/paper_trading.pid
```

**Monitor logs:**
```bash
tail -f logs/paper_trading.log
```

---

## 📊 Monitoring Your Paper Trading

### Real-Time Log Monitoring

```bash
# Follow main log
tail -f logs/paper_trading.log

# Filter for trade signals
tail -f logs/paper_trading.log | grep -i "PLACING.*ORDER"

# Filter for ML decisions
tail -f logs/paper_trading.log | grep -i "ML.*prediction"

# Filter for exits
tail -f logs/paper_trading.log | grep -i "exit"
```

### Health Checks

```bash
# Check process is running
ps aux | grep "runner.main" | grep -v grep

# Check heartbeat
cat /var/run/pine/worker_heartbeat.json

# Watch heartbeat updates (every 30 seconds)
watch -n 5 cat /var/run/pine/worker_heartbeat.json
```

---

## 🎯 Expected Paper Trading Behavior

### What to Expect (Based on Your Test Results)

**ES:**
- Trades/month: ~50-60 (593/year ÷ 12)
- Win rate: ~73%
- Avg P&L/trade: $224 ($531k / 2,372 trades)
- Monthly P&L: ~$11k-13k

**NQ:**
- Trades/month: ~55-60
- Win rate: ~68%
- Avg P&L/trade: $334
- Monthly P&L: ~$18k-20k

**RTY:**
- Trades/month: ~58-62
- Win rate: ~70%
- Avg P&L/trade: $129
- Monthly P&L: ~$7k-8k

**YM:**
- Trades/month: ~56-60
- Win rate: ~71%
- Avg P&L/trade: $152
- Monthly P&L: ~$8k-9k

**TOTAL (all 4 symbols):**
- **Combined trades/month**: ~220-240
- **Combined monthly P&L**: ~$44k-50k
- **Daily P&L**: ~$2k-2.5k

### Red Flags to Watch For

⚠️ **Stop and investigate if:**
- Win rate drops below 50% after 100 trades
- Daily losses exceed $5,000 consistently
- Execution errors >5% of orders
- Models stop making predictions (errors loading)
- System crashes or hangs

---

## 📈 Performance Tracking Template

Create a simple tracking spreadsheet or log file:

```bash
# Daily tracking
echo "Date,Symbol,Trades,Wins,Losses,PnL,Notes" > paper_trading_results.csv

# Add entries daily
echo "2024-11-14,ES,3,2,1,+450,Normal day" >> paper_trading_results.csv
echo "2024-11-14,NQ,2,1,1,+120,Light volume" >> paper_trading_results.csv
echo "2024-11-14,RTY,4,3,1,+280,Strong signals" >> paper_trading_results.csv
echo "2024-11-14,YM,2,2,0,+310,Perfect day" >> paper_trading_results.csv
```

### Weekly Review Checklist

Every Sunday, review:

1. **Total trades**: Compare to expected (~50-60/week all symbols)
2. **Win rate**: Should be 65-75%
3. **Weekly P&L**: Should be +$10k-12k
4. **Largest loss**: Should be <$500/trade
5. **System stability**: Any crashes or errors?
6. **Execution quality**: Are fills at expected prices?

---

## 🛑 How to Stop Paper Trading

### Graceful Stop

```bash
# Find process
ps aux | grep "runner.main" | grep -v grep

# Kill gracefully (using PID from output)
kill <PID>

# Or if you used nohup:
kill $(cat logs/paper_trading.pid)
```

### Emergency Killswitch

```bash
# Edit .env
nano .env

# Change:
POLICY_KILLSWITCH=true

# Restart (it will run but not place trades)
```

---

## ✅ Validation Timeline

### Week 1-2: Basic Validation (50-100 trades)
- **Goal**: Ensure system is stable and executing correctly
- **Abort if**: Win rate <40%, system crashes, execution errors

### Week 3-4: Moderate Confidence (100-200 trades)
- **Goal**: Validate performance is in expected range
- **Abort if**: Win rate <50%, consistent losses, major errors

### Week 5-8: High Confidence (300-500 trades)
- **Goal**: Gather statistical evidence performance is real
- **Continue if**: Win rate >60%, Sharpe >1.5, consistent profitability

### Month 3-4: Final Validation (600-1000 trades)
- **Goal**: Prove robustness across different market conditions
- **Go live if**: All metrics stable and within expected range

---

## 🚀 Next Steps

1. **Start paper trading** using commands above
2. **Monitor daily** for first 2 weeks
3. **Track metrics** in spreadsheet
4. **Review weekly** to ensure performance matches expectations
5. **After 3-4 months** with good results, consider live trading

---

## 💡 Key Differences from Backtest

**Expect these degradations from your test results:**

| Metric | Test Results | Expected Paper | Reason |
|--------|-------------|----------------|---------|
| Sharpe | 3.2-3.6 | 2.0-2.8 | Execution costs, slippage |
| Win Rate | 68-73% | 60-68% | Model degradation |
| Monthly Return | ~50-70% | ~15-30% | More conservative in live |
| Max DD | 4-11% | 8-15% | Real-world volatility |

**If you achieve even 50% of test performance, you have an excellent strategy.**

---

## 📞 Support

If you encounter issues:
1. Check logs first: `tail -100 logs/paper_trading.log`
2. Verify .env configuration
3. Test model loading manually
4. Check TradersPost paper account is active
5. Review this guide's troubleshooting sections

Good luck with paper trading! 🚀
