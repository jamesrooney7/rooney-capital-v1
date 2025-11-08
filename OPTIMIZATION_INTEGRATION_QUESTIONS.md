# Optimization Integration - Questions & Findings

Based on your feedback, here's what I found and what I need your help with:

---

## ✅ What I Found

### 1. ML Model Files Location
**Answer**: ✅ CONFIRMED

ML models are located at:
```
/home/user/rooney-capital-v1/src/models/
├── ES_best.json
├── ES_rf_model.pkl
├── NQ_best.json
├── NQ_rf_model.pkl
├── ... (12 total symbols)
```

**12 symbols have ML models**: ES, NQ, RTY, YM, CL, NG, GC, SI, HG, 6A, 6B, 6E

**Path**: `src/models/` (NOT `src/models/ibs/`)

---

### 2. Full Universe of Instruments
**Source**: `src/strategy/ibs_strategy.py` EXECUTION_SYMBOLS

```python
EXECUTION_SYMBOLS = {
    # Equity indices
    "ES", "NQ", "RTY", "YM",
    # Commodities
    "GC", "SI", "HG", "CL", "NG", "PL",
    # Currencies
    "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S",
    # Grains
    "ZC", "ZS", "ZW"
}
```

**Total**: 18 symbols

---

### 3. Portfolio Optimization Export Tool
**Found**: `research/export_portfolio_config.py`

This script exports greedy search results to JSON:
```bash
python research/export_portfolio_config.py \
    --results results/greedy_optimization_results.csv \
    --output config/portfolio_optimization.json
```

**Inputs**: CSV from greedy optimizer
**Outputs**: JSON config with:
- max_positions
- daily_stop_loss
- symbols (optimized list)
- expected_performance metrics

---

### 4. Previous Portfolio Optimization Results
**Source**: `PORTFOLIO_INTEGRATION_GUIDE.md`

This document mentions:
```json
{
  "portfolio_constraints": {
    "max_positions": 4,
    "daily_stop_loss": 2500.0,
    "symbols": ["6A", "6B", "6C", "6N", "6S", "CL", "ES", "PL", "RTY", "SI"]
  }
}
```

**10 symbols listed** - BUT you said this is NOT the final IBS list

---

## ❓ What I Need Your Help Finding

### Question 1: Actual IBS Greedy Search Results

You mentioned the 10-symbol list in `PORTFOLIO_INTEGRATION_GUIDE.md` is NOT the final optimized list for IBS.

**Where can I find the actual greedy search results?**

Possible locations to check:
- [ ] `results/greedy_optimization_results.csv`
- [ ] `results/ibs_optimization/greedy_*.csv`
- [ ] `config/portfolio_optimization.json`
- [ ] `config/portfolio_optimization_ibs.json`
- [ ] Some other CSV/JSON file with the optimization results

**What I need**:
1. The actual optimized symbol list for IBS
2. The actual optimized max_positions for IBS
3. The actual expected performance (Sharpe, CAGR, etc.)

**Can you help me locate this file?** Or if it doesn't exist, we can:
- Run the greedy optimizer to generate it
- Use your knowledge of which symbols performed best
- Manually specify the list based on your optimization results

---

### Question 2: Per-Instrument Filter Selection

You said: "Same filter settings for each one. Just different filters selected for each instrument."

**I need clarification**:

**Option A**: Filter THRESHOLDS are the same, but which filters are ENABLED differs?

Example:
```
All instruments:
  ibs_entry_high: 0.3  (same value)
  stop_perc: 2.0       (same value)

But:
  ES: enable_rsi=True, enable_atr=False
  NQ: enable_rsi=False, enable_atr=True
```

**Option B**: Something else?

**Where is this stored?**
- In the `{SYMBOL}_best.json` files?
- In a separate configuration file?
- In the strategy code itself?

---

### Question 3: Portfolio Optimization File Purpose

Let me explain what this file is for:

**Purpose**: The `config/portfolio_optimization_ibs.json` file is a **configuration file** that tells the multi-alpha system:

1. **Which symbols to trade** (result of greedy search)
   - Example: Only trade these 10 symbols (not all 18)

2. **How many concurrent positions** (result of greedy search)
   - Example: Max 4 positions open at once

3. **Daily stop loss** (from greedy search optimization)
   - Example: $2,500 daily stop

4. **Expected performance** (from greedy search)
   - Example: Sharpe 14.57, CAGR 84.4%
   - This is for monitoring/validation purposes

**How it's created**:
```bash
# Step 1: Run greedy optimizer (generates CSV)
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results/ibs_optimization/ \
    --output results/greedy_optimization_results.csv

# Step 2: Export best configuration to JSON
python research/export_portfolio_config.py \
    --results results/greedy_optimization_results.csv \
    --output config/portfolio_optimization_ibs.json \
    --row 0  # Select best configuration (row 0)
```

**Result**: A JSON file that the strategy worker reads to know:
- "Only trade these N symbols" (subset of 18)
- "Never exceed M concurrent positions"
- "Stop trading if daily loss exceeds $X"

**Does this make sense?** If you have the greedy search results, I can create this file for you.

---

### Question 4: ML Models for Missing Symbols

I found ML models for 12 symbols:
- ✅ ES, NQ, RTY, YM (indices)
- ✅ CL, NG, GC, SI, HG (commodities/metals)
- ✅ 6A, 6B, 6E (currencies)

Missing ML models for 6 symbols:
- ❌ PL (platinum)
- ❌ 6C, 6J, 6M, 6N, 6S (currencies)
- ❌ ZC, ZS, ZW (grains)

**Questions**:
1. Did you run ML optimization for all 18 symbols, or only 12?
2. If only 12, are those the symbols you actually trade?
3. Or do the other 6 trade without ML filtering?

---

## 🎯 Summary of What I Need

### Immediate (to complete Week 2 integration):

1. **IBS Optimized Symbol List**
   - Help me find the greedy search results file
   - Or tell me the final optimized symbol list
   - Plus max_positions and expected Sharpe/CAGR

2. **Per-Instrument Filter Settings**
   - Clarify what you mean by "different filters selected"
   - Point me to where this is stored (if anywhere)

3. **Confirm ML Model Strategy**
   - 12 symbols with ML models = only trade these?
   - Or 18 symbols (6 without ML filtering)?

### Nice-to-Have (for complete documentation):

4. **Greedy Search Results File**
   - If you have the CSV from greedy optimizer
   - I can properly create the portfolio_optimization_ibs.json

5. **Historical Context**
   - When was greedy search last run?
   - What was the optimization period?
   - Train/test split results?

---

## 🚀 Next Steps

Once you provide the above information, I will:

1. ✅ Create `config/portfolio_optimization_ibs.json` with correct settings
2. ✅ Update `config.multi_alpha.yml` with correct instrument list
3. ✅ Update `strategy_worker.py` to load ML bundles correctly
4. ✅ Create tests to verify optimization integration
5. ✅ Update roadmap with accurate Week 2 tasks

---

## 📍 Where to Look

Here are some commands you can run to help me find the files:

```bash
# Find any greedy search results
find /home/user/rooney-capital-v1 -name "*greedy*.csv" -o -name "*optimization*.csv"

# Find any portfolio configuration files
find /home/user/rooney-capital-v1 -name "*portfolio*.json" -o -name "*config*.json"

# List what's in results directory (if it exists)
ls -la /home/user/rooney-capital-v1/results/

# Check if there's a specific IBS optimization directory
ls -la /home/user/rooney-capital-v1/results/ibs*/

# Find any recent CSVs with results
find /home/user/rooney-capital-v1 -name "*.csv" -mtime -60 | grep -i "result\|optim\|greedy"
```

Or just let me know:
- "The optimized IBS symbols are: [list]"
- "Max positions is: [number]"
- "Expected Sharpe is: [number]"

And I'll create the configuration files from that information!

---

**Ready to help however works best for you!** Let me know what you find or what information you can provide. 🚀
