# ✅ ACTUAL IBS Configuration Found!

## Summary

I found the actual IBS portfolio configuration in the original system. It's stored in **`config.yml`**, NOT in a separate JSON file.

---

## 🎯 Actual IBS Portfolio Configuration

**Source**: `/opt/pine/rooney-capital-v1/config.yml` (lines 43-55)

```yaml
portfolio:
  max_positions: 2          # ← ACTUAL VALUE (not 4)
  daily_stop_loss: 2500
  instruments:              # ← ACTUAL 9 SYMBOLS (not 10)
    - 6A
    - 6B
    - 6C
    - 6M
    - 6N
    - 6S
    - CL
    - HG
    - SI
```

**Key Findings**:
- **9 symbols** (not 10): 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI
- **Max positions: 2** (not 4)
- **Daily stop loss: $2,500** ✓ correct
- **Note**: PL excluded due to slippage issues (mentioned in comments)

---

## 📋 How Original System Works

### Configuration Structure

**All configuration is in `config.yml`** (not separate JSON files):

1. **All symbols to load** (for complete ML feature calculation):
   ```yaml
   symbols:  # 19 total
     - 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
     - CL, ES, GC, HG, NG, NQ, PL, RTY, SI, TLT, YM
   ```

2. **Portfolio instruments** (actually traded):
   ```yaml
   portfolio:
     instruments:  # 9 symbols
       - 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI
   ```

3. **Models path**:
   ```yaml
   models_path: src/models  # ✓ Confirmed
   ```

### Portfolio Optimization Workflow

From `SYSTEM_GUIDE.md`:

**Step 1: Run greedy optimizer**
```bash
python3 research/portfolio_optimizer_greedy_train_test.py \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-breach-events 2 --max-dd 9000 \
    --update-config  # <-- Auto-updates config.yml!
```

**Step 2: Results saved to**:
- `config.yml` (automatically updated)
- `results/greedy_optimization_TIMESTAMP.json` (detailed results)

**What it does**:
1. Greedy instrument removal (removes worst performers)
2. Train/test split validation
3. Applies realistic execution costs (4-tick slippage)
4. Enforces $2,500 daily stop loss
5. **Automatically updates config.yml** with optimal symbols and max_positions

---

## 🔍 ML Models Analysis

### Symbols with ML Models (12 total)

From `src/models/`:
```
✓ ES, NQ, RTY, YM       (indices - 4)
✓ CL, NG, GC, SI, HG    (commodities/metals - 5)
✓ 6A, 6B, 6E            (currencies - 3)
```

### Symbols in Portfolio (9 total)

From `config.yml`:
```
✓ 6A, 6B, 6C            (currencies - 3)
✓ 6M, 6N, 6S            (currencies - 3)
✓ CL, HG, SI            (commodities/metals - 3)
```

### Cross-reference

**Symbols with ML models AND in portfolio**: 6A, 6B, CL, HG, SI (5 symbols)

**Symbols in portfolio WITHOUT ML models**: 6C, 6M, 6N, 6S (4 symbols)

**Question**: Do these 4 symbols (6C, 6M, 6N, 6S) trade without ML filtering?
- Or do they share ML models from 6A/6B?
- Or is there a different ML loading mechanism?

---

## 🔑 Answer to Your Questions

### Q1: Filter Settings
**Answer**: Same filter VALUES, different filters ENABLED per instrument

This is likely controlled by:
- Filter columns passed to IbsStrategy per instrument
- Possibly in the ML optimization metadata (`{SYMBOL}_best.json`)

### Q2: ML Model Location
**Answer**: ✅ Confirmed at `src/models/`

```
src/models/
├── 6A_best.json + 6A_rf_model.pkl
├── 6B_best.json + 6B_rf_model.pkl
├── 6E_best.json + 6E_rf_model.pkl
├── CL_best.json + CL_rf_model.pkl
├── ES_best.json + ES_rf_model.pkl
├── GC_best.json + GC_rf_model.pkl
├── HG_best.json + HG_rf_model.pkl
├── NG_best.json + NG_rf_model.pkl
├── NQ_best.json + NQ_rf_model.pkl
├── RTY_best.json + RTY_rf_model.pkl
├── SI_best.json + SI_rf_model.pkl
└── YM_best.json + YM_rf_model.pkl
```

### Q3: Portfolio Optimization File
**Answer**: It's `config.yml`, not a separate JSON

The greedy optimizer updates `config.yml` directly using the `--update-config` flag.

### Q4: Actual IBS Optimized List
**Answer**: ✅ **9 symbols** (from `config.yml`)

```
6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI
```

With max_positions: 2 (not 4)

---

## 📊 Comparison: Doc vs Reality

| Item | PORTFOLIO_INTEGRATION_GUIDE.md | Actual (config.yml) |
|------|-------------------------------|-------------------|
| Symbols | 10: 6A,6B,6C,6N,6S,CL,ES,PL,RTY,SI | 9: 6A,6B,6C,6M,6N,6S,CL,HG,SI |
| Max Positions | 4 | 2 |
| Daily Stop | $2,500 | $2,500 ✓ |
| Expected Sharpe | 14.574 | Unknown (need results file) |

**Conclusion**: The document shows EXAMPLE results, not the actual current configuration.

---

## 🚀 Next Steps for Multi-Alpha Integration

### 1. Create IBS Portfolio Optimization File

Based on actual `config.yml`:

**File**: `config/portfolio_optimization_ibs.json`

```json
{
  "strategy": "ibs",
  "optimization_metadata": {
    "source": "config.yml (production configuration)",
    "last_updated": "2024-11-08",
    "note": "Current production settings from original system"
  },
  "portfolio_constraints": {
    "max_positions": 2,
    "daily_stop_loss": 2500.0,
    "symbols": ["6A", "6B", "6C", "6M", "6N", "6S", "CL", "HG", "SI"],
    "n_symbols": 9
  },
  "expected_performance": {
    "note": "Metrics from actual greedy optimization not found. May need to re-run optimizer to get Sharpe/CAGR."
  },
  "excluded_symbols": {
    "PL": "Excluded due to slippage issues",
    "ES": "Not in optimized portfolio (has ML model but not selected)",
    "NQ": "Not in optimized portfolio (has ML model but not selected)",
    "RTY": "Not in optimized portfolio (has ML model but not selected)",
    "YM": "Not in optimized portfolio (has ML model but not selected)",
    "GC": "Not in optimized portfolio (has ML model but not selected)",
    "NG": "Not in optimized portfolio (has ML model but not selected)",
    "6E": "Not in optimized portfolio (has ML model but not selected)",
    "6J": "Not in portfolio (no ML model)",
    "ZC": "Not in portfolio (no ML model)",
    "ZS": "Not in portfolio (no ML model)",
    "ZW": "Not in portfolio (no ML model)"
  }
}
```

### 2. Update Multi-Alpha Config

**File**: `config.multi_alpha.yml`

```yaml
strategies:
  ibs:
    enabled: true
    broker_account: ${TRADERSPOST_WEBHOOK_URL}
    starting_cash: 250000  # ← From config.yml
    models_path: src/models/  # ← Confirmed correct
    max_positions: 2  # ← From config.yml (not 4!)
    daily_stop_loss: 2500

    # ACTUAL optimized IBS symbols from config.yml
    instruments:
      - 6A
      - 6B
      - 6C
      - 6M
      - 6N
      - 6S
      - CL
      - HG
      - SI

    strategy_params:
      # Generic parameters (same for all instruments)
      # ML threshold loaded per-instrument from {SYMBOL}_best.json
```

### 3. Strategy Worker ML Loading

The strategy worker should:

1. Load `config.multi_alpha.yml` → get list of 9 instruments
2. For each instrument (e.g., 6A):
   - Try to load `src/models/6A_best.json` + `6A_rf_model.pkl`
   - If exists: Use ML filtering with optimized threshold
   - If NOT exists: Run without ML filtering (or warn/skip)

**Note**: 4 of the 9 instruments (6C, 6M, 6N, 6S) don't have ML models!

### 4. Questions Remaining

1. **How do 6C, 6M, 6N, 6S trade without ML models?**
   - Do they use a shared model?
   - Do they trade without ML filtering?
   - Should we generate ML models for them?

2. **What were the actual greedy optimization results?**
   - Expected Sharpe, CAGR, Max DD?
   - When was it last run?
   - Results file: `results/greedy_optimization_TIMESTAMP.json`?

3. **Per-instrument filter selection**
   - Where is this stored?
   - Is it in the strategy code itself?
   - Or in configuration files?

---

## 📁 Commands for User (Correct Path)

Since you're on `/opt/pine/rooney-capital-v1`:

```bash
# View current production config
cat /opt/pine/rooney-capital-v1/config.yml | grep -A 15 "portfolio:"

# Search for greedy optimization results
find /opt/pine/rooney-capital-v1 -name "*greedy*.json" -o -name "*greedy*.csv"

# Check for results directory
ls -la /opt/pine/rooney-capital-v1/results/ 2>/dev/null || echo "results/ doesn't exist"

# View which symbols have ML models
ls -1 /opt/pine/rooney-capital-v1/src/models/*_best.json | sed 's/.*\///' | sed 's/_best.json//'
```

---

## ✅ Summary

**Found**: Actual IBS configuration in `config.yml`

**Actual Settings**:
- 9 symbols: 6A, 6B, 6C, 6M, 6N, 6S, CL, HG, SI
- Max positions: 2
- Daily stop: $2,500

**ML Models**: 12 symbols have models, but only 5 of the 9 portfolio symbols have models

**Next**: Need to understand how the 4 symbols without models (6C, 6M, 6N, 6S) are handled.

**Status**: Ready to create portfolio_optimization_ibs.json and update multi-alpha config!
