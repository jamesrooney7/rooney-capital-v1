# Strategy Automation Pipeline Guide

## Overview

The automated pipeline (`research/auto_optimize_strategy.py`) handles **Steps 2-6** of strategy development:

1. ❌ **Strategy Code** - Manual (you write it)
2. ✅ **Backtest** - Automated verification
3. ✅ **Feature Extraction** - Automated
4. ✅ **ML Training** - Automated (parallel)
5. ✅ **Portfolio Optimization** - Automated
6. ✅ **Results Tracking** - Automated

**Time Savings:** ~2 hours → ~10 minutes + wait time per strategy

---

## Prerequisites

### 1. Strategy Code Must Exist

Create your strategy class first:

```python
# src/strategy/breakout_strategy.py
from src.strategy.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def should_enter_long(self) -> bool:
        # Your logic here
        pass

    def should_enter_short(self) -> bool:
        pass

    def should_exit(self) -> bool:
        pass

    def get_features_snapshot(self) -> Dict[str, Any]:
        pass
```

### 2. Backtest Trade CSVs Must Exist

Run backtests manually first to generate trade files:

```bash
# For each symbol:
python research/backtest_runner.py --symbol ES --start 2010-01-01 --end 2025-01-01

# Or use a loop:
for symbol in ES NQ RTY YM 6A 6B 6C CL GC SI; do
    python research/backtest_runner.py --symbol $symbol --start 2010-01-01 --end 2025-01-01
done
```

This creates: `results/breakout_optimization/{SYMBOL}_rf_best_trades.csv`

---

## Quick Start

### Basic Usage (All 18 Symbols)

```bash
python research/auto_optimize_strategy.py --strategy breakout
```

This will:
1. ✓ Verify trade CSVs exist for all 18 symbols
2. ✓ Extract features for all 18 symbols
3. ✓ Train 18 ML models in parallel (16 jobs)
4. ✓ Run greedy portfolio optimizer
5. ✓ Add results to `results/all_optimizations.json`

### Custom Symbol List

```bash
python research/auto_optimize_strategy.py \
    --strategy breakout \
    --symbols ES NQ RTY CL GC SI
```

### Custom Parallelism (Smaller Server)

```bash
python research/auto_optimize_strategy.py \
    --strategy breakout \
    --parallel-jobs 8
```

### Custom Constraints

```bash
python research/auto_optimize_strategy.py \
    --strategy breakout \
    --max-positions 3 \
    --max-dd-limit 7500 \
    --initial-capital 200000 \
    --daily-stop-loss 3000
```

---

## Pipeline Details

### Step 2: Backtest Verification

**What it does:**
- Checks if trade CSV files exist for all requested symbols
- Aborts if any are missing

**Expected files:**
```
results/{STRATEGY}_optimization/ES_rf_best_trades.csv
results/{STRATEGY}_optimization/NQ_rf_best_trades.csv
...
```

**If missing:**
```
ERROR: Missing trade files for: ES, NQ
Please run backtests first before continuing.

Example commands:
  python research/backtest_runner.py --symbol ES --start 2010-01-01
  python research/backtest_runner.py --symbol NQ --start 2010-01-01
```

### Step 3: Feature Extraction

**What it does:**
- Runs `research/extract_training_data.py` for each symbol
- Saves to `data/training/{SYMBOL}_transformed_features.csv`
- Skips if features already exist

**Output:**
```
[1/18] Extracting features: ES
  ✓ Features extracted: ES_transformed_features.csv
[2/18] Extracting features: NQ
  ✓ Features already exist: NQ_transformed_features.csv
...
```

### Step 4: ML Training (Parallel)

**What it does:**
- Runs `research/rf_cpcv_random_then_bo.py` for each symbol in parallel
- Uses your server's parallel approach (xargs)
- Copies models to `src/models/` automatically

**Settings:**
- Feature selection end: 2020-12-31
- Holdout start: 2023-01-01
- Random search trials: 25
- Bayesian optimization trials: 65
- CPCV folds: 5
- Test folds: 2
- Embargo days: 2

**Output:**
```
[2025-11-08 15:42:00] Training ES...
[2025-11-08 15:42:00] Training NQ...
[2025-11-08 15:42:00] Training RTY...
...
[2025-11-08 15:52:00] ✓ Completed ES
  → Models copied to src/models/
[2025-11-08 15:53:00] ✓ Completed NQ
...
Models created: 18/18
```

### Step 5: Portfolio Optimization

**What it does:**
- Runs greedy optimizer on all trained models
- Train period: 2023-01-01 to 2023-12-31
- Test period: 2024-01-01 to 2024-12-31
- Saves to `results/all_optimizations.json`

**Output:**
```
Train period: 2023-01-01 to 2023-12-31
Test period: 2024-01-01 to 2024-12-31
Constraint: Max DD < $5,000

TESTING max_positions = 1
...
TESTING max_positions = 4

BEST CONFIGURATION:
  Optimal Symbols: ES, NQ, CL, GC, SI
  Max Positions: 2
  Test Sharpe: 10.4
  Test Max DD: $4,200
```

---

## Expected Runtime

| Step | Time (18 symbols) |
|------|-------------------|
| Backtest verification | < 1 second |
| Feature extraction | 5-10 minutes |
| ML training (parallel) | 20-40 minutes |
| Portfolio optimization | 3-5 minutes |
| **Total** | **~30-60 minutes** |

*With 16 parallel jobs on 125GB RAM server*

---

## Output Files

### Individual Strategy Results

```
results/breakout_optimization/
├── ES_rf_best_trades.csv         (from backtest)
├── ES_best.json                  (ML model metadata)
├── ES_rf_model.pkl               (trained model)
├── optimization_ES.log           (ML training log)
├── NQ_rf_best_trades.csv
├── NQ_best.json
├── NQ_rf_model.pkl
...
├── greedy_optimization_breakout_TIMESTAMP.json  (portfolio results)
```

### Consolidated Results

```
results/all_optimizations.json
```

All strategies ranked by test Sharpe:

```json
[
  {
    "strategy_name": "breakout",
    "test_sharpe": 10.4,
    "optimal_symbols": ["ES", "NQ", "CL", "GC", "SI"],
    "max_positions": 2,
    ...
  },
  {
    "strategy_name": "ibs_a",
    "test_sharpe": 11.2,
    ...
  }
]
```

### Models Deployed

```
src/models/
├── ES_best.json           ✅ Automatically copied
├── ES_rf_model.pkl        ✅ Automatically copied
├── NQ_best.json
├── NQ_rf_model.pkl
...
```

---

## Example: Full Workflow for New Strategy

### 1. Write Strategy Code (10 minutes)

```python
# src/strategy/breakout_strategy.py
class BreakoutStrategy(BaseStrategy):
    # Your implementation
    pass
```

### 2. Run Backtests (Manual, ~30 minutes)

```bash
for symbol in ES NQ RTY YM 6A 6B 6C 6E 6J 6M 6N 6S CL NG GC SI HG PL; do
    python research/backtest_runner.py --symbol $symbol --start 2010-01-01 &
done
wait
```

### 3. Run Automation Pipeline (~45 minutes)

```bash
python research/auto_optimize_strategy.py --strategy breakout
```

**Wait for completion...**

### 4. Check Results

```bash
cat results/all_optimizations.json | jq '.[] | select(.strategy_name=="breakout")'
```

### 5. Deploy (Next Steps)

If results are good:
- Create `config/portfolio_optimization_breakout.json`
- Update `config.multi_alpha.yml` with breakout strategy
- Deploy to production

---

## Troubleshooting

### "Strategy code not found"

```
ERROR: Strategy code not found: src/strategy/breakout_strategy.py
Please create the strategy class first before running pipeline.
```

**Solution:** Create the strategy class first.

### "Missing trade files"

```
ERROR: Missing trade files for: ES, NQ
Please run backtests first before continuing.
```

**Solution:** Run backtests to generate trade CSVs.

### "ML training failed"

Check individual symbol logs:
```bash
cat results/breakout_optimization/optimization_ES.log
```

Common issues:
- Not enough data in date range
- Missing features in training data
- Memory issues (reduce `--parallel-jobs`)

### "No ML models created"

Check that feature extraction succeeded:
```bash
ls -lh data/training/*_transformed_features.csv
```

### Pipeline timeout

For slower servers:
- Reduce `--parallel-jobs`
- Run fewer symbols at once
- Split into batches

---

## Command Reference

### All Arguments

```bash
python research/auto_optimize_strategy.py \
    --strategy STRATEGY_NAME          # Required: strategy name
    --symbols ES NQ CL ...            # Optional: specific symbols (default: all 18)
    --parallel-jobs 16                # Optional: parallel ML jobs (default: 16)
    --max-positions 4                 # Optional: max positions (default: 4)
    --max-dd-limit 5000               # Optional: max DD limit (default: 5000)
    --initial-capital 150000          # Optional: capital (default: 150000)
    --daily-stop-loss 2500            # Optional: daily stop (default: 2500)
```

### Examples

**Test on 6 symbols with lower parallelism:**
```bash
python research/auto_optimize_strategy.py \
    --strategy breakout \
    --symbols ES NQ CL GC SI HG \
    --parallel-jobs 6
```

**More aggressive constraints:**
```bash
python research/auto_optimize_strategy.py \
    --strategy momentum \
    --max-dd-limit 7500 \
    --daily-stop-loss 3500 \
    --initial-capital 200000
```

---

## Future Enhancements

### Phase 2: Template-Based Generation

Auto-generate strategy code from templates:

```bash
python research/auto_optimize_strategy.py \
    --strategy breakout_20 \
    --template breakout \
    --params "lookback=20,volume_confirm=true"
```

### Phase 3: LLM Code Generation

Generate strategy from description:

```bash
python research/auto_optimize_strategy.py \
    --strategy custom_ma \
    --generate-from "Buy when 10MA crosses above 50MA with ADX>25"
```

---

## Next Steps

After pipeline completes:

1. **Review results:**
   ```bash
   cat results/all_optimizations.json | jq '.'
   ```

2. **Create config files** (I'll help with this)

3. **Update multi-alpha config** (I'll help with this)

4. **Deploy to production**

---

## Tips

✅ **Always run backtests first** - Pipeline checks but won't generate them

✅ **Start small** - Test on 3-5 symbols first, then scale up

✅ **Monitor resources** - Adjust `--parallel-jobs` based on your server

✅ **Check logs** - Individual symbol logs in `results/{STRATEGY}_optimization/`

✅ **Version control models** - Commit `src/models/*.json` and `*.pkl` to git
