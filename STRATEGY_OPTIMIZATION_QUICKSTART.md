# Strategy Optimization Quick Start Guide

## TL;DR

```bash
# 1. Optimize base strategy parameters for ES
./research/run_full_strategy_optimization.sh ES

# 2. Extract training data with optimized parameters
python research/extract_training_data.py \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31

# 3. Train ML model
python research/train_rf_three_way_split.py \
    --symbol ES \
    --rs-trials 120 \
    --bo-trials 300
```

## What Just Happened?

### Before (Old Process)
```
Manual parameters → Extract trades → Train ML → Deploy
```
**Problem:** Not enough trades! (~10k instead of 20k)

### After (New Process)
```
Auto-optimize parameters → Extract trades → Train ML → Deploy
                ↓
        Maximize trade volume!
```

## New Files You'll See

### Input
- `data/resampled/ES_hourly.csv` - Your existing hourly data

### Output
- `config/strategy_params.json` - **Optimized parameters** (auto-used everywhere)
- `optimization_results/ES/` - Detailed results and reports

## Key Commands

### Optimize Parameters (Run First!)
```bash
./research/run_full_strategy_optimization.sh ES
```
**Duration:** 2-4 hours
**Output:** `config/strategy_params.json` updated

### Extract Training Data (Uses Optimized Params)
```bash
python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31
```
**Duration:** 1-2 hours
**Output:** `data/training/ES_transformed_features.csv`

### Train ML Model (Existing Process)
```bash
python research/train_rf_three_way_split.py --symbol ES
```
**Duration:** 2-3 hours
**Output:** `src/models/ES_rf_model.pkl`

## What Gets Optimized?

| Parameter | Old Value | Optimized Range | Why? |
|-----------|-----------|-----------------|------|
| IBS entry | 0.2 | 0.15 - 0.35 | More/fewer entries |
| IBS exit | 0.8 | 0.65 - 0.80 | Earlier/later exits |
| Stop loss | 1.0 ATR | 2.5 - 3.5 ATR | Wider stops = fewer stop-outs |
| Take profit | 0.5 ATR | 1.0 - 2.5 ATR | Let winners run |
| Max hold | 8 hrs | 6 - 15 hrs | More time = more targets hit |

**Goal:** Find the sweet spot that generates **3,000-5,000 trades** with Sharpe > 0.25 and win rate > 48%

## Checking Results

### Did It Work?
```bash
cat optimization_results/ES/reports/final_approval_decision.json
```

Look for:
```json
{
  "final_decision": {
    "decision": "APPROVED",  // ← You want this!
    "for_ml_layer": true,
    "message": "All evaluations passed. GOOD: 4200 total OOS trades..."
  }
}
```

### What Parameters Did It Choose?
```bash
cat config/strategy_params.json | jq '.ES'
```

Example output:
```json
{
  "ibs_entry_high": 0.25,     // ← Optimized!
  "ibs_exit_low": 0.75,       // ← Optimized!
  "stop_atr_mult": 3.0,       // ← Optimized!
  "target_atr_mult": 2.0,     // ← Optimized!
  "max_holding_bars": 10,     // ← Optimized!
  "_total_oos_trades": 4200,  // ← Trade count achieved
  "_decision": "APPROVED"     // ← Ready for ML!
}
```

## What If It Fails?

### "Insufficient trades" Error

**Problem:** Parameters too strict

**Fix:** Edit `research/optimize_base_strategy_params.py`, make ranges wider:
```python
'ibs_entry_high': [0.20, 0.25, 0.30, 0.35, 0.40],  # Added 0.40
'ibs_exit_low': [0.60, 0.65, 0.70, 0.75],  # Added 0.60
```

### "REJECTED" Decision

**Problem:** Failed stability or validation checks

**Fix:** Check detailed reports:
```bash
# See which check failed
cat optimization_results/ES/reports/final_approval_decision.json

# Check stability
cat optimization_results/ES/analysis/parameter_stability_analysis.json

# Check held-out performance
cat optimization_results/ES/heldout/heldout_evaluation.json
```

## Running on Multiple Symbols

```bash
# Optimize all symbols (runs one at a time)
for SYMBOL in ES NQ RTY YM; do
    ./research/run_full_strategy_optimization.sh $SYMBOL
done
```

**Note:** Each symbol takes 2-4 hours, so this could run overnight.

## Integration with Existing Workflow

### Your Old Workflow
```bash
# 1. Extract data (manual parameters)
python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31

# 2. Train ML
python research/train_rf_three_way_split.py --symbol ES
```

### Your New Workflow
```bash
# 0. FIRST TIME ONLY: Optimize parameters
./research/run_full_strategy_optimization.sh ES

# 1. Extract data (auto-uses optimized parameters from config)
python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31

# 2. Train ML (same as before)
python research/train_rf_three_way_split.py --symbol ES
```

**Key Difference:** Step 0 is new, but only needs to run once (or annually).

## When to Re-Optimize

- **First time** for each symbol
- **Annually** (parameters may drift)
- **After major market regime change** (e.g., COVID-19, Fed policy shift)
- **When live Sharpe degrades >30%** from backtest expectations

## What Happens Under the Hood?

1. **Walk-Forward Optimization** (2016-2020)
   - Trains on 2011-2015, tests on 2016
   - Trains on 2011-2016, tests on 2017
   - ... continues through 2020
   - Tests 100 parameter combinations per window using Bayesian optimization

2. **Stability Analysis**
   - Checks if optimal parameters are consistent across windows
   - Calculates median and standard deviation
   - Flags unstable parameters

3. **Held-Out Validation** (2021-2024)
   - Tests median parameters on completely unseen data
   - Ensures no overfitting
   - Calculates validation efficiency

4. **Final Decision**
   - Combines all checks
   - APPROVED if passes all criteria
   - Writes to `config/strategy_params.json`

## Pro Tips

### Speed Up Optimization
```python
# Edit research/optimize_base_strategy_params.py
OPTUNA_CONFIG = {
    'n_trials': 50,  # Reduce from 100 (faster but less thorough)
}
```

### Get More Trades
```python
# Edit research/optimize_base_strategy_params.py
'ibs_entry_high': [0.25, 0.30, 0.35, 0.40, 0.45],  # More permissive
'ibs_exit_low': [0.55, 0.60, 0.65, 0.70],  # Exit earlier
```

### Override for Testing
```bash
# Extract with specific parameters (ignores config)
python research/extract_training_data.py \
    --symbol ES \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --ibs-entry-high 0.30 \
    --ibs-exit-low 0.70
```
*(Note: You'd need to add these CLI args to extract_training_data.py)*

## Files Cheat Sheet

| File | What It Does |
|------|-------------|
| `research/run_full_strategy_optimization.sh` | **Master script** - Run this! |
| `config/strategy_params.json` | **Optimized params** - Auto-read by everything |
| `research/extract_training_data.py` | **Extract trades** - Now uses config |
| `optimization_results/ES/reports/final_approval_decision.json` | **Check success** - Read this first |
| `src/strategy/strategy_params_loader.py` | **Config loader** - Used by all scripts |

## Monitoring Optimization Progress

While optimization is running:
```bash
# Watch progress
tail -f optimization_results/ES/windows/window_1_optimization_history.csv

# Check current trial count
wc -l optimization_results/ES/windows/window_1_optimization_history.csv
```

## Next Steps After Optimization

1. **Review Results**
   ```bash
   cat optimization_results/ES/reports/final_approval_decision.json
   ```

2. **Extract Training Data**
   ```bash
   python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31
   ```

3. **Train ML Model**
   ```bash
   python research/train_rf_three_way_split.py --symbol ES
   ```

4. **Deploy to Production**
   - Model automatically uses optimized parameters from config
   - No code changes needed!

## Questions?

See `BASE_STRATEGY_OPTIMIZATION_README.md` for detailed documentation.

---

**Remember:** The optimization only needs to run **once per symbol** (or annually). After that, all subsequent steps automatically use the optimized parameters!
