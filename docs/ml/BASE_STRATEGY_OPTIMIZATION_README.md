# Base Strategy Parameter Optimization System

## Overview

This system optimizes the core IBS (Internal Bar Strength) strategy parameters **before** ML training to maximize trade volume while maintaining quality thresholds. This ensures sufficient training data for downstream machine learning models.

**Primary Goal:** Generate 3,000-5,000 out-of-sample trades with win rate > 48% and Sharpe > 0.25

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   BASE STRATEGY OPTIMIZATION                        │
│                                                                     │
│  Input: Hourly OHLCV data (2011-2024)                             │
│  Output: Optimized parameters → config/strategy_params.json       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA EXTRACTION                                   │
│                                                                     │
│  Uses optimized parameters from config                             │
│  Generates ALL trades (no ML filtering)                            │
│  Output: {SYMBOL}_transformed_features.csv                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML OPTIMIZATION                                   │
│                                                                     │
│  Three-way split with CPCV                                         │
│  Optimizes ML threshold and hyperparameters                        │
│  Output: Trained model + metadata                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Run Full Optimization (Recommended)

```bash
# One command to optimize ES
./research/run_full_strategy_optimization.sh ES

# With custom data directory
./research/run_full_strategy_optimization.sh ES --data-dir /path/to/data
```

This will:
1. Run walk-forward optimization (5 windows, 2016-2020)
2. Analyze parameter stability
3. Validate on held-out data (2021-2024)
4. Make final approval decision
5. Write optimized parameters to `config/strategy_params.json`

**Duration:** ~2-4 hours for ES (100 trials × 5 windows)

### Step-by-Step Workflow

If you prefer to run each step individually:

```bash
# 1. Walk-forward optimization
python research/optimize_base_strategy_params.py --symbol ES

# 2. Stability analysis
python research/analyze_strategy_stability.py --symbol ES

# 3. Held-out validation
python research/validate_strategy_heldout.py --symbol ES

# 4. Final decision
python research/finalize_strategy_decision.py --symbol ES
```

## Parameters Optimized

### Fixed Parameters (Not Optimized)
- `ibs_entry_low`: 0.0 (capture all oversold)
- `ibs_exit_high`: 1.0 (capture all overbought exits)
- `atr_period`: 14 (industry standard)
- `auto_close_hour`: 15 (3 PM ET)

### Optimized Parameters

| Parameter | Description | Search Space | Rationale |
|-----------|-------------|--------------|-----------|
| `ibs_entry_high` | IBS entry threshold | [0.15, 0.20, 0.25, 0.30, 0.35] | Higher = more trades |
| `ibs_exit_low` | IBS exit threshold | [0.65, 0.70, 0.75, 0.80] | Lower = more time for target |
| `stop_atr_mult` | Stop loss (ATR multiplier) | [2.5, 3.0, 3.5] | Wider = fewer stop-outs |
| `target_atr_mult` | Take profit (ATR multiplier) | [1.0, 1.5, 2.0, 2.5] | Higher = let winners run |
| `max_holding_bars` | Max holding period (hours) | [6, 8, 10, 12, 15] | Longer = fewer time exits |

**Total search space:** 1,200 combinations (Bayesian optimization tests ~100 per window)

## Walk-Forward Windows

```
Window 1: Train 2011-2015 → Test 2016
Window 2: Train 2011-2016 → Test 2017
Window 3: Train 2011-2017 → Test 2018
Window 4: Train 2011-2018 → Test 2019
Window 5: Train 2011-2019 → Test 2020

Held-Out: 2021-2024 (NEVER used during optimization)
```

## Objective Function

**Volume-Prioritized Scoring:**
```python
score = 0.70 * (num_trades / 2500) + 0.30 * sharpe_ratio

# Bonuses for high volume:
if num_trades >= 3000: score *= 1.3  # 30% bonus
elif num_trades >= 2500: score *= 1.15  # 15% bonus
elif num_trades >= 2000: score *= 1.05  # 5% bonus
```

**Hard Constraints (must pass):**
- Minimum trades: 1,500
- Minimum win rate: 48%
- Minimum Sharpe: 0.20

## Success Criteria

### Walk-Forward Success
- ✓ Total OOS trades ≥ 2,500
- ✓ Mean Walk-Forward Efficiency (WFE) ≥ 0.5
- ✓ At least 3 of 5 windows profitable
- ✓ Mean Sharpe ≥ 0.25
- ✓ Win rate: 48% - 57%

### Parameter Stability
- ✓ Critical params (entry/exit) CV < 0.20
- ✓ Secondary params (stops/targets) CV < 0.30
- ✓ No wild instability across windows

### Held-Out Validation
- ✓ Validation Efficiency (VE) ≥ 0.5
- ✓ Trades ≥ 1,500
- ✓ Win rate within ±5pp of walk-forward mean
- ✓ Max drawdown < 25%

## Output Structure

```
optimization_results/ES/
├── windows/
│   ├── window_1_optimal_params.json        # Best params for window 1
│   ├── window_1_train_metrics.json         # Training performance
│   ├── window_1_test_metrics.json          # Test performance
│   ├── window_1_optimization_history.csv   # All 100 trials
│   └── ... (windows 2-5)
├── analysis/
│   ├── aggregate_walkforward_metrics.json  # Summary across all windows
│   ├── parameter_stability_analysis.json   # CV and stability ratings
│   └── robust_parameter_ranges.json        # Median ± 1 std for each param
├── heldout/
│   ├── heldout_validation_metrics.json     # Performance on 2021-2024
│   └── heldout_evaluation.json             # VE and criteria checks
└── reports/
    └── final_approval_decision.json        # APPROVED/REJECTED + ML config
```

## Configuration File

After successful optimization, parameters are written to `config/strategy_params.json`:

```json
{
  "_comment": "Optimized base strategy parameters for each symbol",
  "_last_updated": "2025-01-19",
  "_optimization_version": "1.0",
  "ES": {
    "ibs_entry_low": 0.0,
    "ibs_entry_high": 0.25,
    "ibs_exit_low": 0.75,
    "ibs_exit_high": 1.0,
    "stop_atr_mult": 3.0,
    "target_atr_mult": 2.0,
    "max_holding_bars": 10,
    "atr_period": 14,
    "auto_close_hour": 15,
    "_optimized": true,
    "_optimization_date": "2025-01-19T10:30:00",
    "_walk_forward_sharpe": 0.35,
    "_heldout_sharpe": 0.32,
    "_total_oos_trades": 4200,
    "_heldout_trades": 2100,
    "_decision": "APPROVED"
  }
}
```

## Integration with Existing Pipeline

### 1. Data Extraction

`extract_training_data.py` now **automatically reads** parameters from `config/strategy_params.json`:

```bash
# Automatically uses optimized parameters for ES
python research/extract_training_data.py \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31
```

No need to pass parameters manually! If no optimized config exists, it uses defaults.

### 2. Live Trading

The `IbsStrategy` class can read parameters from config via `strategy_params_loader.py`:

```python
from strategy.strategy_params_loader import get_strategy_params_for_backtrader

# Load optimized parameters
params = get_strategy_params_for_backtrader('ES')

# Add to Cerebro
cerebro.addstrategy(IbsStrategy, symbol='ES', **params)
```

## Trade Volume Thresholds

### Per-Window Training (In-Sample)
- **< 1,500:** STOP optimization (too restrictive)
- **< 2,000:** Warning (marginal)
- **≥ 2,500:** Target
- **≥ 3,000:** Good

### Per-Window Testing (Out-of-Sample)
- **< 300:** Unreliable (exclude window)
- **< 500:** Marginal reliability
- **≥ 700:** Target (per year)
- **≥ 900:** Good

### Total OOS (2016-2020 combined)
- **< 2,000:** REJECT for ML
- **< 3,000:** Simplified ML (20 features, no ensemble)
- **< 5,000:** Standard ML (25 features)
- **≥ 7,000:** Full ML (30 features, ensemble)

### Held-Out (2021-2024)
- **< 1,000:** Can't validate
- **< 2,000:** High uncertainty
- **≥ 2,800:** Expected baseline
- **≥ 3,500:** Good

## Troubleshooting

### Issue: Window 1 fails with "insufficient trades"

**Cause:** Parameter ranges too restrictive

**Solution:**
1. Edit `research/optimize_base_strategy_params.py`
2. Make parameter ranges more permissive:
   ```python
   'ibs_entry_high': [0.20, 0.25, 0.30, 0.35, 0.40],  # Added 0.40
   'ibs_exit_low': [0.60, 0.65, 0.70, 0.75],  # Added 0.60
   ```
3. Re-run optimization

### Issue: Parameters unstable (high CV)

**Cause:** Market regime changes across windows

**Options:**
1. **Accept it:** Some instability is normal. Check if still APPROVED.
2. **Shorter windows:** Use 1-year test windows instead of anchored
3. **Regime filters:** Add market regime detection

### Issue: Held-out validation fails

**Cause:** Overfitting to 2011-2020 period

**Options:**
1. **Review parameters:** Check if they make economic sense
2. **Simplify:** Reduce number of optimized parameters
3. **Different periods:** Try different train/test splits

## Advanced Usage

### Custom Parameter Ranges

Edit `OPTIMIZATION_PARAMS` in `research/optimize_base_strategy_params.py`:

```python
OPTIMIZATION_PARAMS = {
    'ibs_entry_high': {
        'type': 'categorical',
        'values': [0.20, 0.25, 0.30, 0.35, 0.40],  # Customized
    },
    # ... other params
}
```

### More Trials

Increase Bayesian optimization trials:

```python
OPTUNA_CONFIG = {
    'n_trials': 200,  # Default: 100
    # ... other config
}
```

### Different Windows

Edit `WINDOWS` to use different time periods or shorter/longer windows.

## Performance Benchmarks

**ES (E-mini S&P 500) - Expected Results:**
- Total OOS trades: 3,500-5,000
- Mean Sharpe: 0.25-0.40
- Win rate: 50-54%
- Held-out trades: 1,800-2,500
- Optimization time: 2-4 hours

## Key Differences from ML Optimization

| Aspect | Base Strategy Optimization | ML Optimization |
|--------|---------------------------|-----------------|
| **What's optimized** | Entry/exit thresholds, stops/targets | ML hyperparameters, threshold |
| **Goal** | Maximize trade volume + quality | Maximize Sharpe/DSR |
| **When it runs** | Once per symbol (or annually) | After each data extraction |
| **Filters applied** | None (pure IBS) | ML prediction probability |
| **Time period** | 2011-2020 (train), 2021-2024 (test) | 2010-2018 (train), 2019-2020 (threshold), 2021-2024 (test) |

## Files Reference

### Core Scripts
- `research/optimize_base_strategy_params.py` - Main optimizer (walk-forward)
- `research/analyze_strategy_stability.py` - Parameter stability analysis
- `research/validate_strategy_heldout.py` - Held-out validation
- `research/finalize_strategy_decision.py` - Final approval decision
- `research/run_full_strategy_optimization.sh` - Master orchestrator

### Utilities
- `research/utils/vectorized_backtest.py` - Fast backtest engine (~100x faster than Backtrader)
- `src/strategy/strategy_params_loader.py` - Config file loader

### Configuration
- `config/strategy_params.json` - Optimized parameters for all symbols

## FAQ

**Q: Do I need to re-run this for every symbol?**
A: Yes, each symbol has different optimal parameters.

**Q: How often should I re-optimize?**
A: Annually, or when you notice degraded live performance.

**Q: Can I override optimized parameters?**
A: Yes! In `extract_training_data.py` or live trading, pass parameters explicitly:
```python
cerebro.addstrategy(IbsStrategy, symbol='ES', ibs_entry_high=0.30)
```

**Q: What if optimization is REJECTED?**
A: Review the evaluation reports. Common fixes:
- Make parameter ranges more permissive
- Use different time periods
- Check data quality

**Q: Can I use this with other strategies?**
A: Yes! The vectorized backtest engine is generic. Just modify the entry/exit logic.

## Support

For questions or issues:
1. Check `optimization_results/{SYMBOL}/reports/final_approval_decision.json`
2. Review window-by-window results in `optimization_results/{SYMBOL}/windows/`
3. Consult `BASE_STRATEGY_OPTIMIZATION_README.md` (this file)

## Version History

- **v1.0** (2025-01-19): Initial release
  - Walk-forward optimization (5 windows)
  - Parameter stability analysis
  - Held-out validation
  - Automated approval decision
  - Config file integration
