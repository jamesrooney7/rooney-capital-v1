# Proper Testing Workflow (No Leakage)

## Overview

This document describes the correct workflow to optimize portfolio parameters while maintaining strict separation between training and testing.

## Timeline

```
2010-2020: ML Training (hyperparameters + features)
2021:      Threshold Optimization
2022-2024: HOLDOUT (never touched until final test)
```

## Workflow Steps

### STAGE 1: Train Individual Models (2010-2020)

Train all 11 models using 2010-2020 for hyperparameter optimization:

```bash
# This uses the batch script you already have
# Just verify the dates are correct (train_end=2020-12-31)
./research/batch_train_all_symbols.sh  # Or your existing script
```

**Output**: Models in `src/models/` with optimized hyperparameters

---

### STAGE 2: Threshold Optimization (2021)

The training script already does this automatically in Phase 2:
- Uses 2021 data (threshold period)
- Finds optimal threshold for each model
- Saves to `*_best.json` under "Prod_Threshold"

**Output**: Each model has its optimal threshold (typically 0.50-0.55)

---

### STAGE 3: Portfolio Optimization (2010-2021 ONLY)

Now optimize portfolio parameters on the TRAINING+THRESHOLD period (2010-2021):

```bash
# Step 1: Run backtests on 2010-2021 (TRAINING PERIOD ONLY)
python research/batch_backtest_all_symbols.sh \
    --start 2010-01-01 \
    --end 2021-12-31 \
    --output-dir results/training_period_2010_2021

# Step 2: Optimize portfolio on TRAINING PERIOD results
python research/portfolio_optimizer_greedy.py \
    --results-dir results/training_period_2010_2021 \
    --max-dd-limit 9000 \
    --max-breach-events 2 \
    --min-positions 1 \
    --max-positions 10 \
    --output results/portfolio_config_from_training.csv

# Step 3: LOCK IN the configuration
# Extract best configuration
python -c "
import pandas as pd
df = pd.read_csv('results/portfolio_config_from_training.csv')
best = df.iloc[0]
print('='*60)
print('LOCKED PORTFOLIO CONFIGURATION (from training period):')
print('='*60)
print(f'Max Positions: {int(best[\"max_positions\"])}')
print(f'Symbols: {best[\"symbols\"]}')
print(f'Expected Sharpe (training): {best[\"sharpe\"]:.3f}')
print('='*60)
print('')
print('⚠️  DO NOT CHANGE THESE PARAMETERS AFTER SEEING TEST RESULTS!')
print('')
" > LOCKED_PORTFOLIO_CONFIG.txt

cat LOCKED_PORTFOLIO_CONFIG.txt
```

**Output**: Locked configuration file with symbols + max_positions

**CRITICAL**: Write down these numbers and DO NOT change them after seeing test results!

---

### STAGE 4: Final Holdout Test (2022-2024)

Now test the LOCKED configuration on unseen data:

```bash
# Step 1: Run backtests on HOLDOUT period (2022-2024)
python research/batch_backtest_all_symbols.sh \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --output-dir results/holdout_2022_2024

# Step 2: Simulate portfolio with LOCKED configuration
# Extract the locked max_positions from your config
LOCKED_MAX_POSITIONS=$(grep "Max Positions:" LOCKED_PORTFOLIO_CONFIG.txt | awk '{print $3}')

python research/portfolio_simulator.py \
    --results-dir results/holdout_2022_2024 \
    --max-positions $LOCKED_MAX_POSITIONS \
    --daily-stop-loss 2500 \
    --initial-cash 250000 \
    --output results/FINAL_UNBIASED_RESULTS.csv

# Step 3: View TRUE unbiased Sharpe ratio
python -c "
import pandas as pd
df = pd.read_csv('results/FINAL_UNBIASED_RESULTS.csv')
best = df[df['max_positions'] == $LOCKED_MAX_POSITIONS].iloc[0]
print('')
print('='*60)
print('FINAL UNBIASED RESULTS (2022-2024 Holdout)')
print('='*60)
print(f'Portfolio Sharpe: {best[\"sharpe_ratio\"]:.3f}')
print(f'CAGR: {best[\"cagr\"]*100:.2f}%')
print(f'Max Drawdown: \${best[\"max_drawdown_dollars\"]:,.2f}')
print(f'Profit Factor: {best[\"profit_factor\"]:.2f}')
print('='*60)
print('')
print('This is your TRUE, unbiased performance estimate.')
print('Expected to be 15-25% lower than previous biased estimate.')
print('')
"
```

**Output**: TRUE unbiased Sharpe ratio (expected: 10.9-12.3)

---

## Key Principles

### ✅ DO:
- Optimize portfolio params on 2010-2021 (training period)
- Lock configuration BEFORE looking at 2022-2024
- Test locked config on 2022-2024 (holdout)
- Accept whatever Sharpe you get on holdout

### ❌ DON'T:
- Run portfolio optimizer on 2022-2024 data
- Change max_positions after seeing holdout results
- Re-optimize anything after seeing test performance
- Cherry-pick symbols based on holdout performance

---

## Expected Results

### Training Period (2010-2021):
- Portfolio Sharpe: ~12-15 (optimized on this data)
- This tells you: "What configuration works best in training"
- **Lock in**: symbols + max_positions

### Holdout Period (2022-2024):
- Portfolio Sharpe: ~10.9-12.3 (with locked config)
- This tells you: "True out-of-sample performance"
- **This is your real estimate**

The holdout will be 15-25% lower due to:
1. No optimization on this data (good!)
2. Different market conditions
3. Model naturally less effective out-of-sample

---

## Summary Command Sequence

```bash
# 1. Train models (2010-2020)
./batch_train_all_symbols.sh

# 2. Backtest training period (2010-2021)
./batch_backtest_all_symbols.sh --start 2010-01-01 --end 2021-12-31 --output-dir results/training_2010_2021

# 3. Optimize portfolio on training period
python research/portfolio_optimizer_greedy.py \
    --results-dir results/training_2010_2021 \
    --output results/portfolio_config_training.csv

# 4. LOCK configuration (write it down!)
# Extract max_positions and symbols, save to file

# 5. Backtest holdout period (2022-2024)
./batch_backtest_all_symbols.sh --start 2022-01-01 --end 2024-12-31 --output-dir results/holdout_2022_2024

# 6. Test with LOCKED config
python research/portfolio_simulator.py \
    --results-dir results/holdout_2022_2024 \
    --max-positions <LOCKED_VALUE> \
    --output results/FINAL_RESULTS.csv
```

---

## Validation

After this workflow:
- ✅ No parameter optimized on test data
- ✅ Clean temporal separation
- ✅ Portfolio config from training only
- ✅ True unbiased estimate from holdout

This is the CORRECT way to do it!
