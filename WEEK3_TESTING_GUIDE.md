# Week 3: Multi-Alpha Testing & Validation Guide

## Overview

This guide walks you through testing and validating the multi-alpha system with **IBS A** and **IBS B** running independently.

**Timeline:** 5-7 days
**Goal:** Validate that both strategies run independently without interference
**Location:** Server at `/opt/pine/rooney-capital-v1`

---

## Phase 3.1: Integration Tests (Day 1)

### Step 1: Pull Latest Changes

On your server:

```bash
cd /opt/pine/rooney-capital-v1
git fetch origin
git pull origin claude/multi-alpha-architecture-design-011CUukP4GDwseKxc9VirVNm
```

### Step 2: Verify Files

Check that all required files are present:

```bash
# Week 3 integration test
ls -lh tests/test_integration_week3.py

# Updated test config
ls -lh config.test.yml

# Portfolio optimization results
ls -lh config/portfolio_optimization_ibs_a.json
ls -lh config/portfolio_optimization_ibs_b.json

# Multi-alpha example config
ls -lh config.multi_alpha.example.yml
```

### Step 3: Run Week 3 Integration Tests

This will test:
- Both IBS A and IBS B configurations load correctly
- Strategies have independent instruments and constraints
- ML models load for each strategy's instruments
- Portfolio coordinators are independent
- Optimization results match configurations

```bash
cd /opt/pine/rooney-capital-v1
python tests/test_integration_week3.py
```

**Expected Output:**
```
================================================================================
# WEEK 3 INTEGRATION TESTS: Multi-Alpha System (IBS A & IBS B)
================================================================================

================================================================================
TEST 1: Multi-Alpha Configuration Loading
================================================================================
✅ Loaded config from: config.test.yml
✅ Found strategy: ibs_a
✅ Found strategy: ibs_b

📊 IBS A Configuration:
   Enabled: True
   Instruments: ['6A', '6C', '6M', 'CL', 'GC', 'HG']
   Max positions: 2
   Daily stop loss: $2,500
   Starting cash: $150,000

📊 IBS B Configuration:
   Enabled: True
   Instruments: ['6B', '6N', '6S', 'SI', 'YM']
   Max positions: 3
   Daily stop loss: $2,500
   Starting cash: $150,000

✅ Both strategies configured correctly

================================================================================
TEST 2: Strategy Independence
================================================================================
IBS A instruments (6): ['6A', '6C', '6M', 'CL', 'GC', 'HG']
IBS B instruments (5): ['6B', '6N', '6S', 'SI', 'YM']
✅ No instrument overlap - strategies are fully independent
✅ Different max_positions: IBS A=2, IBS B=3
✅ Different broker accounts configured

✅ Strategies are independent

================================================================================
TEST 3: ML Model Loading Per Strategy
================================================================================

📦 IBS A ML Models:
   ✅ 6A: threshold=0.540, features=30
   ✅ 6C: threshold=0.520, features=30
   ✅ 6M: threshold=0.510, features=30
   ✅ CL: threshold=0.550, features=30
   ✅ GC: threshold=0.530, features=30
   ✅ HG: threshold=0.525, features=30

   Loaded 6/6 models for IBS A

📦 IBS B ML Models:
   ✅ 6B: threshold=0.515, features=30
   ✅ 6N: threshold=0.505, features=30
   ✅ 6S: threshold=0.520, features=30
   ✅ SI: threshold=0.535, features=30
   ✅ YM: threshold=0.545, features=30

   Loaded 5/5 models for IBS B

✅ ML model loading verified

================================================================================
TEST SUMMARY
================================================================================
✅ PASS: Multi-Alpha Config Loading
✅ PASS: Strategy Independence
✅ PASS: ML Model Loading Per Strategy
✅ PASS: Portfolio Coordinator Independence
✅ PASS: Strategy Registration & Loading
✅ PASS: Optimization Results Verification
--------------------------------------------------------------------------------
Results: 6/6 tests passed
✅ ALL TESTS PASSED!

Next steps:
1. Deploy to server (/opt/pine/rooney-capital-v1)
2. Set up paper trading webhooks
3. Start both strategies and monitor
```

**If tests fail:**
- Check that ML models exist in `src/models/` for all instruments
- Verify config files match expected format
- Check Python environment has all dependencies

---

## Phase 3.2: Paper Trading Setup (Days 2-3)

### Step 1: Set Up Paper Trading Webhooks

Create two **separate** TradersPost webhook URLs in **paper trading mode**:

1. **IBS A Webhook:**
   - Go to TradersPost → Create new webhook
   - Name: "IBS A Paper Trading"
   - Mode: **Paper Trading**
   - Copy webhook URL

2. **IBS B Webhook:**
   - Go to TradersPost → Create new webhook
   - Name: "IBS B Paper Trading"
   - Mode: **Paper Trading**
   - Copy webhook URL

### Step 2: Update Environment Variables

Add webhook URLs to your server's `.env` file:

```bash
cd /opt/pine/rooney-capital-v1
nano .env
```

Add these lines:

```bash
# IBS A Webhook (Paper Trading)
TRADERSPOST_IBS_A_WEBHOOK=https://webhooks.traderspost.io/trading/webhook/your-ibs-a-webhook-id

# IBS B Webhook (Paper Trading)
TRADERSPOST_IBS_B_WEBHOOK=https://webhooks.traderspost.io/trading/webhook/your-ibs-b-webhook-id

# Databento API Key (if not already set)
DATABENTO_API_KEY=your_databento_api_key
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 3: Create Production Config

Copy the example config and customize:

```bash
cd /opt/pine/rooney-capital-v1
cp config.multi_alpha.example.yml config.multi_alpha.yml
```

Verify the config looks correct:

```bash
cat config.multi_alpha.yml | grep -A 10 "ibs_a:" | head -15
cat config.multi_alpha.yml | grep -A 10 "ibs_b:" | head -15
```

### Step 4: Test Strategy Worker Startup

Test starting IBS A worker (dry run):

```bash
python -m src.runner.strategy_worker --strategy ibs_a --config config.multi_alpha.yml --help
```

This should show the help message without errors.

### Step 5: Create Startup Scripts

Create script to start IBS A:

```bash
nano scripts/start_ibs_a.sh
```

Add:

```bash
#!/bin/bash
# Start IBS A Strategy Worker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start IBS A worker
python -m src.runner.strategy_worker \
    --strategy ibs_a \
    --config config.multi_alpha.yml \
    2>&1 | tee -a logs/ibs_a_worker.log
```

Create script to start IBS B:

```bash
nano scripts/start_ibs_b.sh
```

Add:

```bash
#!/bin/bash
# Start IBS B Strategy Worker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start IBS B worker
python -m src.runner.strategy_worker \
    --strategy ibs_b \
    --config config.multi_alpha.yml \
    2>&1 | tee -a logs/ibs_b_worker.log
```

Make scripts executable:

```bash
chmod +x scripts/start_ibs_a.sh
chmod +x scripts/start_ibs_b.sh
```

Create logs directory:

```bash
mkdir -p logs
```

### Step 6: Start Data Hub (if not already running)

Check if data hub is running:

```bash
ps aux | grep data_hub_main
```

If not running, start it:

```bash
python -m src.data_hub.data_hub_main --config config.multi_alpha.yml &
```

Verify it's running:

```bash
# Check process
ps aux | grep data_hub_main

# Check Redis
redis-cli ping
# Should output: PONG
```

---

## Phase 3.3: Run Paper Trading (Days 4-5)

### Step 1: Start IBS A Worker

In a new terminal/tmux pane:

```bash
cd /opt/pine/rooney-capital-v1
./scripts/start_ibs_a.sh
```

**Watch for:**
- ✅ "Loaded config from config.multi_alpha.yml"
- ✅ "Strategy: ibs_a"
- ✅ "Instruments: 6A, 6C, 6M, CL, GC, HG"
- ✅ "Loaded ML models for 6 instruments"
- ✅ "Connected to Redis"
- ✅ "Subscribed to market data channels"

### Step 2: Start IBS B Worker

In another terminal/tmux pane:

```bash
cd /opt/pine/rooney-capital-v1
./scripts/start_ibs_b.sh
```

**Watch for:**
- ✅ "Loaded config from config.multi_alpha.yml"
- ✅ "Strategy: ibs_b"
- ✅ "Instruments: 6B, 6N, 6S, SI, YM"
- ✅ "Loaded ML models for 5 instruments"
- ✅ "Connected to Redis"
- ✅ "Subscribed to market data channels"

### Step 3: Monitor Both Strategies

**Check logs in real-time:**

```bash
# Terminal 1: IBS A logs
tail -f logs/ibs_a_worker.log

# Terminal 2: IBS B logs
tail -f logs/ibs_b_worker.log
```

**Watch for signals:**
- Entry signals for respective instruments
- ML veto decisions
- Position tracking
- Order submissions to TradersPost

### Step 4: Verify Independence

**Test 1: Stop IBS A, verify IBS B continues**

```bash
# Stop IBS A (Ctrl+C in its terminal)
# Check IBS B logs - should continue running normally
tail -f logs/ibs_b_worker.log
```

**Test 2: Restart IBS A, verify no interference**

```bash
# Restart IBS A
./scripts/start_ibs_a.sh

# Check both logs - both should run normally
```

**Test 3: Check position tracking**

Both strategies should track positions independently:
- IBS A: max 2 concurrent positions (from 6A, 6C, 6M, CL, GC, HG)
- IBS B: max 3 concurrent positions (from 6B, 6N, 6S, SI, YM)

### Step 5: Monitor TradersPost

Check both paper trading accounts:

1. **IBS A account:**
   - Should receive orders only for: 6A, 6C, 6M, CL, GC, HG
   - Max 2 positions at once

2. **IBS B account:**
   - Should receive orders only for: 6B, 6N, 6S, SI, YM
   - Max 3 positions at once

---

## Phase 3.4: Validation Checklist (Days 6-7)

Run for 48 hours minimum before going to production. Check:

### Independence ✅

- [ ] Both strategies run simultaneously without crashes
- [ ] Stopping/restarting one doesn't affect the other
- [ ] Each strategy only trades its configured instruments
- [ ] Position limits respected independently (IBS A: 2, IBS B: 3)

### Performance ✅

- [ ] Latency <200ms from bar to signal
- [ ] Memory stable over 48 hours (no leaks)
- [ ] No missing bars or data loss
- [ ] Redis memory usage acceptable

### Correctness ✅

- [ ] ML models loading correctly per strategy
- [ ] Entry/exit signals match expected logic
- [ ] Orders sent to correct webhook (IBS A vs IBS B)
- [ ] Position tracking accurate

### Monitoring ✅

- [ ] Logs show clear separation (ibs_a vs ibs_b)
- [ ] Heartbeat files updating (if implemented)
- [ ] TradersPost shows separate accounts with correct symbols

---

## Troubleshooting

### Problem: "Strategy 'ibs_a' not found in config"

**Solution:** Check config file has `ibs_a:` (not `ibs:`):

```bash
cat config.multi_alpha.yml | grep "ibs_a:"
cat config.multi_alpha.yml | grep "ibs_b:"
```

### Problem: "ML model not found for symbol X"

**Solution:** Verify models exist:

```bash
ls -lh src/models/6A_best.json
ls -lh src/models/6A_rf_model.pkl
```

All symbols should have both `.json` and `.pkl` files.

### Problem: "Cannot connect to Redis"

**Solution:** Start Redis:

```bash
sudo systemctl start redis
redis-cli ping  # Should output: PONG
```

### Problem: "Orders not appearing in TradersPost"

**Solution:** Check webhook configuration:

```bash
# Verify environment variables set
echo $TRADERSPOST_IBS_A_WEBHOOK
echo $TRADERSPOST_IBS_B_WEBHOOK

# Check logs for webhook calls
grep "TradersPost" logs/ibs_a_worker.log
```

### Problem: Strategies interfering with each other

**Solution:** Check that they use different instruments:

```bash
python -c "
from src.config.config_loader import load_config
config = load_config('config.multi_alpha.yml')
print('IBS A:', config.strategies['ibs_a'].instruments)
print('IBS B:', config.strategies['ibs_b'].instruments)
print('Overlap:', set(config.strategies['ibs_a'].instruments) & set(config.strategies['ibs_b'].instruments))
"
```

Should show no overlap.

---

## Next Steps After Validation

Once all validation checks pass:

### Option A: Continue Paper Trading

Keep running in paper mode to gather more data before going live.

### Option B: Production Deployment

1. **Create production webhooks** (live mode)
2. **Update environment variables** with live webhook URLs
3. **Restart both strategies** with live configuration
4. **Monitor closely** for first 24 hours

---

## Rollback Plan

If critical issues arise:

1. **Stop both strategies:**
   ```bash
   pkill -f "strategy_worker.*ibs_a"
   pkill -f "strategy_worker.*ibs_b"
   ```

2. **Close any open positions** in TradersPost manually

3. **Investigate logs:**
   ```bash
   tail -100 logs/ibs_a_worker.log
   tail -100 logs/ibs_b_worker.log
   ```

4. **Fix issues** and re-test before restarting

---

## Success Criteria

Before moving to production, confirm:

- ✅ 48+ hours of stable paper trading
- ✅ Both strategies operating independently
- ✅ No crashes or memory leaks
- ✅ Signals match expected behavior
- ✅ Performance metrics acceptable (<200ms latency)
- ✅ All validation tests passing

---

## Summary

**Phase 3.1 (Day 1):** Run integration tests ✅
**Phase 3.2 (Days 2-3):** Set up paper trading environment ✅
**Phase 3.3 (Days 4-5):** Run both strategies, verify independence ✅
**Phase 3.4 (Days 6-7):** Validation checklist, decide on production ✅

**Total Duration:** 5-7 days
**End State:** Multi-alpha system validated and ready for production deployment

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**For:** Week 3 - Multi-Alpha Testing & Validation
