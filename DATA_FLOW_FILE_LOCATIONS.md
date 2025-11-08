# Data Flow & File Location Reference

## Critical Distinction: Git vs Server

This document clarifies what files exist WHERE and how the optimization process works.

---

## 🗂️ File Locations Matrix

| File Type | Location in Git Repo | Location on Server | Gitignored? |
|-----------|---------------------|-------------------|-------------|
| **Source Code** | ✅ `/home/user/rooney-capital-v1/src/` | ✅ `/opt/pine/rooney-capital-v1/src/` | No |
| **Config Files** | ✅ `config.yml`, `config.*.yml` | ✅ `config.yml` | No |
| **ML Models (.json)** | ✅ `src/models/*.json` | ✅ `src/models/*.json` | No |
| **ML Models (.pkl)** | ✅ `src/models/*.pkl` | ✅ `src/models/*.pkl` | No |
| **Original Market Data** | ❌ NOT in git | ✅ On server (location?) | Yes |
| **Strategy Trades** | ❌ NOT in git | ✅ `results/{SYMBOL}_optimization/*.csv` | Yes (*.csv) |
| **Portfolio Optimization Results** | ❌ NOT in git | ✅ `results/greedy_optimization_*.json` | Yes (results/) |

### Key Point
- **Git repo** (`/home/user/rooney-capital-v1`): Code + configs + ML models
- **Server** (`/opt/pine/rooney-capital-v1`): Everything + data + results
- **Results data** (`results/`): Only on server, NOT in git

---

## 📊 Complete Optimization Data Flow

### For IBS Strategy (Current)

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Original Market Data (ON SERVER ONLY)                  │
├─────────────────────────────────────────────────────────────────┤
│ Location: ??? (where is historical Databento data stored?)      │
│ Format: Parquet files? CSV? Database?                          │
│ Used by: Backtesting, ML optimization                          │
│ In Git: ❌ No (too large, gitignored)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Strategy Backtesting (GENERATES TRADES)                │
├─────────────────────────────────────────────────────────────────┤
│ Script: research/backtest_runner.py                            │
│ Inputs: Original market data + IbsStrategy code                │
│ Outputs: Trade-by-trade results per symbol                     │
│                                                                 │
│ Server Location: results/{SYMBOL}_optimization/                │
│   - ES_trades.csv (every trade for ES)                         │
│   - NQ_trades.csv (every trade for NQ)                         │
│   - ... (one per symbol)                                       │
│                                                                 │
│ In Git: ❌ No (gitignored: *.csv, results/)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: ML Optimization (PER INSTRUMENT)                       │
├─────────────────────────────────────────────────────────────────┤
│ Script: research/train_rf_three_way_split.py                   │
│ Inputs: Strategy trades (results/{SYMBOL}_optimization/)       │
│ Process: Train RandomForest on trade features                  │
│ Outputs (PER SYMBOL):                                          │
│   - {SYMBOL}_best.json (metadata: threshold, features, metrics)│
│   - {SYMBOL}_rf_model.pkl (trained scikit-learn model)         │
│                                                                 │
│ Server Location: src/models/{SYMBOL}_best.json                 │
│ Git Location: ✅ src/models/{SYMBOL}_best.json (COMMITTED)     │
│                                                                 │
│ In Git: ✅ YES (models are committed to git)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Portfolio Greedy Search Optimization                   │
├─────────────────────────────────────────────────────────────────┤
│ Script: research/portfolio_optimizer_greedy_train_test.py      │
│ Inputs: Trade results from ALL symbols (results/)              │
│ Process: Greedy search to find optimal symbol combination      │
│ Outputs:                                                        │
│   - greedy_optimization_TIMESTAMP.json (detailed results)      │
│   - Updates config.yml with optimal settings                   │
│                                                                 │
│ Server Location: results/greedy_optimization_*.json            │
│ Git Location: ❌ NOT in git (results/ is gitignored)           │
│                                                                 │
│ Final Settings Location: config.yml (✅ IN GIT)                │
│   - portfolio.instruments: [optimal symbol list]               │
│   - portfolio.max_positions: N                                 │
│                                                                 │
│ In Git:                                                         │
│   - Results JSON: ❌ No (in results/)                          │
│   - config.yml: ✅ Yes (final settings committed)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ What I Can and Cannot See

### In My Git Repo Environment (`/home/user/rooney-capital-v1`)

**✅ I CAN see:**
- All source code (`src/`)
- All config files (`config.yml`, `config.multi_alpha.example.yml`)
- All ML models (`src/models/*.json`, `src/models/*.pkl`)
- Documentation files

**❌ I CANNOT see:**
- `results/` directory (doesn't exist in git)
- `results/greedy_optimization_20251105_053115.json` (only on your server)
- Strategy trade CSVs (gitignored)
- Original market data (not in git)

### On Your Server (`/opt/pine/rooney-capital-v1`)

**You CAN see:**
- Everything from git
- PLUS `results/` directory with optimization results
- PLUS original market data (wherever it's stored)
- PLUS any local files not committed

---

## 🔍 Verification of My References

Let me verify every file I referenced:

### Files I Referenced That ARE in Git ✅

1. **`src/models/{SYMBOL}_best.json`** ✅
   ```bash
   $ git ls-files src/models/*.json
   src/models/6A_best.json
   src/models/6B_best.json
   ... (12 files)
   ```
   **Status**: Correct reference

2. **`src/models/{SYMBOL}_rf_model.pkl`** ✅
   ```bash
   $ git ls-files src/models/*.pkl
   src/models/6A_rf_model.pkl
   ... (12 files)
   ```
   **Status**: Correct reference

3. **`config.yml`** ✅
   **Status**: In git, correct reference

4. **`config.multi_alpha.example.yml`** ✅
   **Status**: In git, I modified this file correctly

### Files I Referenced That Are ONLY on Your Server ⚠️

1. **`results/greedy_optimization_20251105_053115.json`** ⚠️
   - **My action**: Referenced it in `config/portfolio_optimization_ibs.json`
   - **Status**: I cannot access this, but reference is CORRECT
   - **Reason**: This file exists on your server (you showed me the path)
   - **Usage**: I left Sharpe/CAGR as `null` because I can't read it

2. **`results/{SYMBOL}_optimization/{SYMBOL}_trades.csv`** ⚠️
   - **My action**: Mentioned in documentation
   - **Status**: I cannot access these, but reference is CORRECT
   - **Reason**: These are generated by backtesting on your server

### Files I Created That Reference Server Files ✅

1. **`config/portfolio_optimization_ibs.json`**
   ```json
   {
     "source_file": "results/greedy_optimization_20251105_053115.json",
     "expected_performance": {
       "sharpe_ratio": null,  // ← I left null because I can't access results file
       "cagr": null
     }
   }
   ```
   **Status**: ✅ Correct - references server file I can't access

---

## 🚀 Future Strategy Generation Process

### Example: Creating "Breakout" Strategy

#### On Your Server (Steps 1-4):

```bash
# STEP 1: Generate Strategy Trades (ON SERVER)
cd /opt/pine/rooney-capital-v1
python research/backtest_runner.py --symbol ES --strategy breakout
# Generates: results/ES_optimization/ES_trades.csv

# Do for all symbols...
# Generates: results/{SYMBOL}_optimization/{SYMBOL}_trades.csv

# STEP 2: ML Optimization Per Instrument (ON SERVER)
python research/train_rf_three_way_split.py --symbol ES
# Generates: src/models/breakout/ES_best.json + ES_rf_model.pkl

# Do for all symbols...

# STEP 3: Portfolio Greedy Search (ON SERVER)
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results/breakout_optimization/ \
    --update-config
# Generates: results/greedy_optimization_TIMESTAMP.json
# Updates: config.yml with optimal symbols and max_positions

# STEP 4: Export Portfolio Config (ON SERVER)
python research/export_portfolio_config.py \
    --results results/greedy_optimization_TIMESTAMP.json \
    --output config/portfolio_optimization_breakout.json
# Generates: config/portfolio_optimization_breakout.json
```

#### Then Commit to Git (Steps 5-6):

```bash
# STEP 5: Commit ML Models to Git
git add src/models/breakout/*.json
git add src/models/breakout/*.pkl
git commit -m "Add breakout ML models"

# STEP 6: Commit Config Files to Git
git add config/portfolio_optimization_breakout.json
git add config.multi_alpha.yml  # (after updating)
git commit -m "Add breakout portfolio optimization config"

# STEP 7: Push to Remote
git push
```

#### What Gets Committed vs Ignored:

**Committed to Git** ✅:
- `src/models/breakout/*.json` (ML metadata)
- `src/models/breakout/*.pkl` (ML models)
- `config/portfolio_optimization_breakout.json` (optimization settings)
- `config.multi_alpha.yml` (updated with breakout config)

**NOT Committed (Stays on Server)** ❌:
- `results/` directory (gitignored)
- `results/greedy_optimization_*.json` (detailed results)
- `results/{SYMBOL}_optimization/*.csv` (trade data)
- Original market data

---

## 🎯 Multi-Alpha File References

### strategy_worker.py Will Load:

```python
# FROM GIT (these files are committed):
ml_bundle = load_model_bundle(
    symbol,
    base_dir="src/models/"  # ← Models ARE in git
)
# Returns: model, features, threshold from {SYMBOL}_best.json

# FROM CONFIG (in git):
config = load_config("config.multi_alpha.yml")
portfolio_optimization = load_json("config/portfolio_optimization_ibs.json")

# Uses:
- config.strategies.ibs.instruments  # 9 symbols from greedy search
- config.strategies.ibs.max_positions  # 2 from greedy search
- ml_bundle.threshold  # Per-instrument from {SYMBOL}_best.json
```

### What strategy_worker.py Will NOT Need:

```python
# NEVER needs to access:
- results/ directory (optimization already done)
- Trade CSV files (not needed for live trading)
- greedy_optimization_*.json (results already in config files)
- Original market data (not needed for live trading)
```

---

## ✅ Verification Summary

### My File References Are Correct ✅

1. **ML Models**: ✅ Correctly referenced `src/models/` (in git)
2. **Config Files**: ✅ Correctly referenced `config.yml` (in git)
3. **Portfolio Config**: ✅ Correctly created `config/portfolio_optimization_ibs.json`
4. **Multi-Alpha Config**: ✅ Correctly updated `config.multi_alpha.example.yml`

### What I Cannot Access (But Referenced Correctly) ⚠️

1. **Greedy Results**: ⚠️ `results/greedy_optimization_20251105_053115.json`
   - **Action**: Referenced it, left metrics as `null`
   - **Correct?**: ✅ Yes, because I can't access server files

2. **Trade Data**: ⚠️ `results/{SYMBOL}_optimization/*.csv`
   - **Action**: Mentioned in documentation only
   - **Correct?**: ✅ Yes, strategy_worker doesn't need these

### Questions for You 🤔

1. **Where is original market data stored?**
   - I haven't found references to where Databento historical data lives
   - Is it in a database? Parquet files? Where?

2. **Can you confirm the results/ directory structure?**
   ```
   results/
   ├── greedy_optimization_TIMESTAMP.json
   ├── ES_optimization/
   │   └── ES_trades.csv
   ├── NQ_optimization/
   │   └── NQ_trades.csv
   └── ...
   ```
   Is this the structure?

3. **For future strategies, where should ML models go?**
   - `src/models/breakout/` for breakout strategy?
   - `src/models/meanreversion/` for mean reversion?
   - Or all in `src/models/` with prefixes?

---

## 📋 Action Items

### What I Need to Update (If Issues Found):

1. ❓ **Verify original data references** - Where is market data stored?
2. ❓ **Confirm results/ structure** - Is my understanding correct?
3. ❓ **Update documentation** if I referenced anything incorrectly

### What Looks Good ✅:

1. ✅ ML model loading (src/models/)
2. ✅ Config file references (config.yml, config.multi_alpha.yml)
3. ✅ Portfolio optimization file creation
4. ✅ Understanding that results/ is server-only

---

**Status**: Awaiting your feedback on:
1. Original data storage location
2. Confirmation of results/ structure
3. Any file references that look wrong

Then I can update documentation accordingly!
