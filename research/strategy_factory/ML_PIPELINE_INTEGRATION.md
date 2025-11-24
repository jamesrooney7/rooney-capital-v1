# Strategy Factory → ML Pipeline Integration

## Complete End-to-End Workflow

**Status:** 🚧 Ready for Implementation
**Estimated Time:** Phase 1 (5-10 days) → Porting (1-2 weeks) → ML Pipeline (2-4 weeks)

---

## Overview

This document describes how to pipe winning strategies from **Phase 1 (Strategy Factory)** through **Phase 3 (ML Enhancement)** to create production-ready trading systems.

```
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1: Strategy Factory (Raw Discovery)                   │
│ - 15 instruments × 54 strategies = 810 combinations          │
│ - ~30K-40K parameter backtests total                         │
│ - Gate 1 → Walk-Forward → Regime → Stability → MC → FDR     │
│ Output: Top 10 winners per instrument (150 total)           │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PORTING: Strategy Factory → Backtrader (One-Time)           │
│ - Convert BaseStrategy to Backtrader format                 │
│ - Add collect_filter_values() hooks (50+ features)          │
│ - Maintain Phase 1 optimized parameters                     │
│ Output: Backtrader strategies ready for ML                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PHASE 3: ML Enhancement (Per Strategy)                      │
│ For each of 150 winners:                                    │
│ 1. Extract training data (2010-2021) with 50+ features      │
│ 2. Train ML model (420 trials: 120 random + 300 Bayesian)   │
│ 3. Validate on 2022-2024 (target: 2x Sharpe improvement)    │
│ Output: ML-enhanced strategies with trained models          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PORTFOLIO OPTIMIZATION (Final Selection)                    │
│ - Feed all ML-enhanced strategies to portfolio optimizer    │
│ - Find optimal subset meeting drawdown constraints          │
│ - Max 4 concurrent positions, $2,500 daily stop             │
│ Output: Production portfolio configuration                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Instructions

### **Prerequisites**

✅ Phase 1 complete for all 15 instruments
✅ Database populated with results
✅ All instruments validated

---

### **Step 1: Extract Top Winners**

After all Phase 1 runs complete, extract the top performers:

```bash
cd /opt/pine/rooney-capital-v1

# Extract top 10 strategies per instrument
python research/strategy_factory/extract_winners.py \
    --top-n 10 \
    --output ml_pipeline/winners_manifest.json

# Review results
cat ml_pipeline/winners_manifest.json | jq '.winners_by_instrument'
```

**Output:**
```json
{
  "ES": [
    {"strategy_id": 21, "strategy_name": "RSI2_MeanReversion", "sharpe_ratio": 2.45},
    {"strategy_id": 40, "strategy_name": "BuyOn5BarLow", "sharpe_ratio": 2.12},
    ...
  ],
  "NQ": [...],
  ...
}
```

**Expected:** ~150 winners (15 instruments × 10 strategies)
**Unique strategies:** ~20-30 (many strategies win on multiple instruments)

---

### **Step 2: Port Strategies to Backtrader** ⚠️ **MANUAL EFFORT**

For each **unique strategy type** that won:

1. **Create Backtrader implementation**

   ```bash
   # Example: RSI2MeanReversion
   cp research/strategy_factory/strategies/rsi2_mean_reversion.py \
      src/strategy/strategy_factory/rsi2_mean_reversion_bt.py
   ```

2. **Follow porting guide**

   See: `STRATEGY_TO_BACKTRADER_GUIDE.md`

   Key changes:
   - Inherit from `IbsStrategy` (gets `collect_filter_values()`)
   - Convert pandas logic to Backtrader indicators
   - Add `entry_conditions_met()` and `exit_conditions_met()`
   - Use Phase 1 optimized parameters

3. **Test the ported strategy**

   ```bash
   # Quick test on 2023 data
   python research/backtest_runner.py \
       --symbol ES \
       --start 2023-01-01 \
       --end 2023-12-31 \
       --strategy rsi2_mean_reversion_bt \
       --strategy-params '{"rsi_length": 2, "rsi_oversold": 10, "rsi_overbought": 65}'
   ```

**Effort Estimate:**
- Simple strategies (RSI, MA Cross): 1-2 hours each
- Complex strategies (Ichimoku, Fibonacci): 3-4 hours each
- ~20-30 unique strategies to port
- **Total: 1-2 weeks** (can parallelize)

**Pro Tip:** Port incrementally and test each one before moving to next.

---

### **Step 3: Run Automated ML Pipeline**

Once strategies are ported to Backtrader:

```bash
# Run ML pipeline for ALL winners
nohup python research/strategy_factory/run_ml_pipeline.py \
    --manifest ml_pipeline/winners_manifest.json \
    --output-dir ml_pipeline \
    --workers 4 \
    > ml_pipeline/pipeline.log 2>&1 &

echo "ML Pipeline PID: $!"
```

**What this does (per winner):**

1. **Extract Training Data** (2010-2021)
   ```
   ✅ ES_RSI2_training.csv created (5,234 trades, 52 features)
   ```

2. **Train ML Model** (420 trials)
   ```
   Random Search: 120 trials
   Bayesian Optimization: 300 trials
   ✅ ES_RSI2_model.pkl saved
   Best Sharpe (CPCV): 3.24
   Deflated Sharpe: 2.87
   ```

3. **Validate** (2022-2024)
   ```
   Baseline (no ML): Sharpe = 1.45, Trades = 1,234
   With ML Filter:   Sharpe = 3.12, Trades = 487
   ✅ 2.15x Sharpe improvement!
   ```

**Runtime Estimate:**
- Per strategy: 2-4 hours (depends on # features, trials)
- 150 winners × 3 hours = 450 hours
- With 4 parallel workers: ~112 hours = **4-5 days**

**Monitoring:**

```bash
# Watch progress
tail -f ml_pipeline/pipeline.log

# Check results
ls -lh ml_pipeline/models/
ls -lh ml_pipeline/validation/
```

---

### **Step 4: Review ML Results**

```bash
# Load results
python << EOF
import json
with open('ml_pipeline/reports/ml_pipeline_results_YYYYMMDD_HHMMSS.json') as f:
    results = json.load(f)

completed = [r for r in results if r['status'] == 'completed']
print(f"Successfully enhanced: {len(completed)}/150 strategies")

# Show top performers
# TODO: Parse validation metrics and rank
EOF
```

**Selection Criteria:**
- ✅ Validation Sharpe ≥ 2.0 (with ML)
- ✅ Sharpe improvement ≥ 1.5x vs baseline
- ✅ Sufficient trades in validation period (≥100)
- ✅ Consistent across train/validation periods

**Expected:** ~30-50 strategies pass all criteria

---

### **Step 5: Portfolio Optimization**

Feed ML-enhanced strategies to portfolio optimizer:

```bash
# Generate portfolio backtest data for all enhanced strategies
python research/generate_portfolio_backtest_data.py \
    --strategies ml_pipeline/reports/enhanced_strategies.json \
    --output portfolio_data/all_enhanced.csv

# Run portfolio optimizer
python research/optimize_portfolio_positions.py \
    --input portfolio_data/all_enhanced.csv \
    --max-positions 4 \
    --daily-stop-loss 2500 \
    --max-drawdown 9000 \
    --output config/optimal_portfolio.json
```

**Output:**
```json
{
  "symbols": ["ES", "NQ", "6A", "CL", "GC", "RTY"],
  "strategies": {
    "ES": "RSI2_MeanReversion_ML",
    "NQ": "BuyOn5BarLow_ML",
    ...
  },
  "expected_performance": {
    "sharpe_ratio": 12.5,
    "cagr": 0.67,
    "max_drawdown": 7234.56
  }
}
```

---

## Workflow Summary

### Timeline

| Phase | Duration | Parallel? | Output |
|-------|----------|-----------|--------|
| Phase 1 (All instruments) | 5-10 days | ✅ Yes (sequential per instrument) | 150 winners |
| Strategy Porting | 1-2 weeks | ✅ Yes (manual, can split work) | 20-30 Backtrader strategies |
| ML Pipeline | 4-5 days | ✅ Yes (4 workers) | 150 ML models |
| Portfolio Optimization | 1 day | ❌ No (needs all results) | Final config |
| **Total** | **3-4 weeks** | | **Production system** |

### Resource Requirements

**Compute:**
- Phase 1: 8-16 CPU cores, 8GB RAM
- ML Pipeline: 4+ workers, 16GB+ RAM
- Storage: ~50GB for all data/models

**Human Effort:**
- Strategy porting: 40-80 hours (can be done incrementally)
- Review/validation: 10-20 hours
- **Total: 1-2 weeks part-time**

---

## Automation Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `extract_winners.py` | Get top N per instrument | Database | `winners_manifest.json` |
| `run_ml_pipeline.py` | Automate ML for all winners | Manifest | Models + validation |
| `generate_portfolio_backtest_data.py` | Create portfolio input | Enhanced strategies | CSV for optimizer |
| `optimize_portfolio_positions.py` | Find optimal subset | Portfolio data | Final config |

---

## FAQ

### Q: Do I need to port ALL 54 strategies?

**A:** No! Only the ~20-30 unique strategies that actually won in Phase 1.

### Q: Can I run ML pipeline before porting all strategies?

**A:** Yes! Port incrementally and run ML pipeline for each batch:

```bash
# Port RSI2, BuyOn5BarLow, MACross
# Then run ML for just those:
python run_ml_pipeline.py \
    --manifest winners_manifest.json \
    --strategies 21 40 17  # Strategy IDs
```

### Q: What if ML doesn't improve a strategy?

**A:** That's fine! Some strategies may not benefit from ML filtering. The validation step (2022-2024) will show which ones improve and which don't. Only use strategies with ≥1.5x Sharpe improvement.

### Q: How do I know which parameters to use when porting?

**A:** Use the optimized parameters from Phase 1 winner! They're in `winners_manifest.json`:

```json
{
  "strategy_name": "RSI2_MeanReversion",
  "params": {
    "rsi_length": 2,
    "rsi_oversold": 10,
    "rsi_overbought": 65,
    "stop_loss_atr": 1.5,
    "take_profit_atr": 2.0
  }
}
```

### Q: Can I skip portfolio optimization?

**A:** You could, but it's highly recommended! Portfolio optimization:
- Ensures you don't violate broker constraints (max DD, etc.)
- Finds optimal symbol selection (not all 15)
- Balances diversification vs concentration
- Protects against correlation risk

---

## Next Steps

1. ✅ Wait for Phase 1 to complete (all 15 instruments)
2. ✅ Run `extract_winners.py` to get top 10 per instrument
3. ⚠️ Port unique strategies to Backtrader (manual, 1-2 weeks)
4. ✅ Run `run_ml_pipeline.py` (automated, 4-5 days)
5. ✅ Review results, run portfolio optimization
6. 🚀 Deploy to production!

**Questions?** See individual guides:
- `STRATEGY_TO_BACKTRADER_GUIDE.md` - Porting strategies
- `END_TO_END_OPTIMIZATION_GUIDE.md` - ML pipeline details
- `PORTFOLIO_INTEGRATION_GUIDE.md` - Portfolio optimization

---

**You're on track to build a production ML-enhanced multi-strategy system!** 💪
