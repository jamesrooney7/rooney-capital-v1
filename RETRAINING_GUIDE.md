# Complete Model Retraining Guide

**Date:** 2025-11-11
**Branch:** `claude/fix-model-loader-subdirectories-011CUyeWHsJABoCDcyptEv4H`
**Reason:** Critical lookahead bias fixes applied

---

## 🚨 Why Retrain?

Three critical bugs were fixed that affected training data quality:

1. **Lookahead bias in feature capture** - Features were captured from Bar N+1 instead of Bar N (inflated performance by 10-30%)
2. **Missing warmup in generate_portfolio_backtest_data.py** - Reference symbols had incomplete cross-asset features
3. **Missing warmup in extract_training_data.py** - Training data had NaN cross-asset features for first ~252 bars

**All 18 models must be retrained with corrected data.**

---

## 📋 Step 1: Pull Latest Code from GitHub

```bash
# SSH into your server
cd /opt/pine/rooney-capital-v1

# Fetch latest changes
git fetch origin

# Checkout the branch with fixes
git checkout claude/fix-model-loader-subdirectories-011CUyeWHsJABoCDcyptEv4H

# Pull latest commits
git pull origin claude/fix-model-loader-subdirectories-011CUyeWHsJABoCDcyptEv4H

# Verify you have the latest code
git log --oneline -5
```

**Expected output:** You should see commits like:
- `24d784a Fix: Add warmup period for reference symbols in training data extraction`
- `fe3807b Add comprehensive audit of extract_training_data.py`
- `f1adc15 Fix critical lookahead bias in training data extraction`

---

## 📋 Step 2: Verify Fixed Code

```bash
# Verify feature capture fix (should show intraday_ago=-1)
grep -n "intraday_ago=-1" research/extract_training_data.py

# Verify warmup fix in extract_training_data.py (should show warmup_start_date)
grep -n "warmup_start_date" research/extract_training_data.py | head -5

# Verify warmup fix in generate_portfolio_backtest_data.py
grep -n "warmup_start_date" research/generate_portfolio_backtest_data.py | head -5
```

**Expected:** All three greps should return results showing the fixes are present.

---

## 📋 Step 3: Complete Retraining Process

You need to retrain all 18 symbols with fresh training data.

### 18 Symbols to Retrain:
```
ES, NQ, RTY, YM, GC, SI, HG, CL, NG, PL, 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
```

### Parallel Execution Recommendation (16 cores, 125GB RAM)

**Option 1: Run 4 batches of 4-5 symbols** (RECOMMENDED)
- Each training job can use 3-4 cores internally
- Leaves headroom for system operations
- Estimated time per batch: 2-4 hours
- Total time: 8-16 hours

**Option 2: Run 6 jobs in parallel**
- More aggressive parallelization
- May cause some contention
- Estimated time: 6-10 hours

**Option 3: Sequential (1 at a time)**
- Safest approach
- Full cores available per job
- Estimated time: 18-36 hours

---

## 📋 Step 4A: Batch Training (RECOMMENDED for 16 cores)

### Batch 1 (Equity Indices - 4 symbols)
```bash
cd /opt/pine/rooney-capital-v1

# Run 4 in parallel (using & to background)
nohup python3 research/train_rf_three_way_split.py --symbol ES --bo-trials 50 > logs/ES_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol NQ --bo-trials 50 > logs/NQ_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol RTY --bo-trials 50 > logs/RTY_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol YM --bo-trials 50 > logs/YM_training.log 2>&1 &

# Monitor progress
ps aux | grep train_rf_three_way_split.py | grep -v grep
tail -f logs/ES_training.log  # Watch ES progress (Ctrl+C to exit)
```

**Wait for Batch 1 to complete** (check with `ps aux | grep train_rf` - should show no processes)

### Batch 2 (Metals - 4 symbols)
```bash
nohup python3 research/train_rf_three_way_split.py --symbol GC --bo-trials 50 > logs/GC_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol SI --bo-trials 50 > logs/SI_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol HG --bo-trials 50 > logs/HG_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol PL --bo-trials 50 > logs/PL_training.log 2>&1 &

# Monitor
ps aux | grep train_rf_three_way_split.py | grep -v grep
tail -f logs/GC_training.log
```

**Wait for Batch 2 to complete**

### Batch 3 (Energy - 2 symbols)
```bash
nohup python3 research/train_rf_three_way_split.py --symbol CL --bo-trials 50 > logs/CL_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol NG --bo-trials 50 > logs/NG_training.log 2>&1 &

# Monitor
ps aux | grep train_rf_three_way_split.py | grep -v grep
tail -f logs/CL_training.log
```

**Wait for Batch 3 to complete**

### Batch 4 (Currencies - 8 symbols)
```bash
# Run in two sub-batches (4 at a time)
nohup python3 research/train_rf_three_way_split.py --symbol 6A --bo-trials 50 > logs/6A_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6B --bo-trials 50 > logs/6B_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6C --bo-trials 50 > logs/6C_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6E --bo-trials 50 > logs/6E_training.log 2>&1 &

# Wait for these 4 to finish, then:
nohup python3 research/train_rf_three_way_split.py --symbol 6J --bo-trials 50 > logs/6J_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6M --bo-trials 50 > logs/6M_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6N --bo-trials 50 > logs/6N_training.log 2>&1 &
nohup python3 research/train_rf_three_way_split.py --symbol 6S --bo-trials 50 > logs/6S_training.log 2>&1 &

# Monitor
ps aux | grep train_rf_three_way_split.py | grep -v grep
tail -f logs/6A_training.log
```

---

## 📋 Step 4B: Alternative - Single Batch Script

Create a helper script to run all in sequence:

```bash
cat > retrain_all.sh << 'EOF'
#!/bin/bash
SYMBOLS="ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S"

for symbol in $SYMBOLS; do
    echo "========================================"
    echo "Training $symbol at $(date)"
    echo "========================================"
    python3 research/train_rf_three_way_split.py --symbol $symbol --bo-trials 50

    if [ $? -eq 0 ]; then
        echo "✅ $symbol training completed successfully"
    else
        echo "❌ $symbol training FAILED"
    fi
done

echo "========================================"
echo "All training complete at $(date)"
echo "========================================"
EOF

chmod +x retrain_all.sh

# Run it
nohup ./retrain_all.sh > logs/retrain_all.log 2>&1 &

# Monitor
tail -f logs/retrain_all.log
```

---

## 📋 Step 5: Monitor Training Progress

### Check running processes:
```bash
# See which symbols are training
ps aux | grep train_rf_three_way_split.py | grep -v grep

# See CPU/memory usage
top -b -n 1 | grep python3
```

### Check logs:
```bash
# View recent training output
tail -50 logs/ES_training.log

# Follow live output
tail -f logs/ES_training.log

# Check for errors
grep -i error logs/*.log
```

### Check for completed models:
```bash
# List trained models
ls -lh results/*/src/models/*/*.pkl

# Check metadata
cat results/ES_optimization/src/models/ES/ES_rf_metadata.json | jq .
```

---

## 📋 Step 6: Verify Training Results

After each symbol completes, verify the outputs:

```bash
# Check model file exists
ls -lh results/ES_optimization/src/models/ES/ES_rf_model.pkl

# Check metadata
cat results/ES_optimization/src/models/ES/ES_rf_metadata.json

# Check training data was generated
ls -lh data/training/ES_transformed_features.csv

# Check test results (should be in the log)
grep -A 10 "PHASE 3" logs/ES_training.log
```

**Expected for NG:** Test results should now be LOWER than before (due to removing lookahead bias), but should MATCH the backtest results we generate later.

---

## 📋 Step 7: After All Training Complete

### Verify all 18 models were created:
```bash
# Count model files
find results/ -name "*_rf_model.pkl" | wc -l
# Should show: 18

# List all models with sizes
find results/ -name "*_rf_model.pkl" -exec ls -lh {} \;
```

### Check for any failures:
```bash
# Check logs for errors
grep -i "error\|failed" logs/*_training.log

# Check which symbols completed
for sym in ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S; do
    if [ -f "results/${sym}_optimization/src/models/${sym}/${sym}_rf_model.pkl" ]; then
        echo "✅ $sym - COMPLETE"
    else
        echo "❌ $sym - MISSING"
    fi
done
```

---

## 📋 Step 8: Compare Old vs New Results

Once retraining is complete, compare performance:

### Expected Changes:
- **Training test results:** 10-30% LOWER than before (removing lookahead bias)
- **Real-time backtest results:** Should now MATCH training test results
- **Feature quality:** Cross-asset features fully populated (no NaN values)

### Example for NG:
```bash
# Check new training results (from log)
grep -A 20 "PHASE 3.*2023.*2024" logs/NG_training.log

# Expected: ~$45k-60k instead of previous $75k (due to lookahead fix)
# But this should now MATCH the backtest when we regenerate it!
```

---

## 🎯 Expected Timeline (16 cores)

**Per symbol:** ~2-3 hours
**Per batch (4 symbols):** ~2-4 hours
**Total time (all 18):** 8-16 hours

---

## ⚠️ Troubleshooting

### Training hangs or uses 100% memory:
```bash
# Check memory usage
free -h

# Kill stuck process
ps aux | grep train_rf_three_way_split.py
kill <PID>

# Reduce parallel jobs (run 2-3 at a time instead of 4)
```

### Import errors:
```bash
# Verify you're in the right directory
pwd  # Should be /opt/pine/rooney-capital-v1

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Re-activate virtual environment if needed
source venv/bin/activate  # or conda activate rooney-capital
```

### Data file not found:
```bash
# Check data files exist
ls -lh data/resampled/ES_hourly.csv
ls -lh data/resampled/ES_daily.csv

# If missing, you may need to regenerate resampled data
```

---

## 📝 Summary Commands

```bash
# 1. Pull latest code
git pull origin claude/fix-model-loader-subdirectories-011CUyeWHsJABoCDcyptEv4H

# 2. Verify fixes
grep -n "intraday_ago=-1" research/extract_training_data.py

# 3. Start training (batch approach - 4 at a time)
cd /opt/pine/rooney-capital-v1
mkdir -p logs

# Batch 1 - Indices
for sym in ES NQ RTY YM; do
    nohup python3 research/train_rf_three_way_split.py --symbol $sym --bo-trials 50 > logs/${sym}_training.log 2>&1 &
done

# Monitor
watch 'ps aux | grep train_rf_three_way_split.py | grep -v grep'

# When batch 1 done, run batch 2, etc.
```

---

**Last Updated:** 2025-11-11
**Status:** Ready to retrain all 18 models with corrected, unbiased training data
