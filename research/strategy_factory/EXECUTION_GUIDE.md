# Strategy Factory - Execution Guide

**Status**: ✅ **READY TO RUN**
**Date**: 2025-01-20
**Target**: Run Phase 1 on your server

---

## 🚀 Quick Start

```bash
# Navigate to project root
cd /opt/pine/rooney-capital-v1

# Run Phase 1: Test all 10 strategies on ES (2010-2024)
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --timeframe 15min \
    --workers 16

# Expected runtime: 30-60 minutes
# Expected output: 5-10 winning strategies
```

That's it! The system will:
1. Run 235 backtests in parallel (16 cores)
2. Apply 5 statistical filters sequentially
3. Save results to SQLite database
4. Generate markdown report
5. Log everything to `strategy_factory.log`

---

## 📋 Prerequisites

### 1. **Data Files** ✅
Ensure you have 15-minute bar data at:
```
/opt/pine/rooney-capital-v1/data/resampled/ES_15min.csv
```

Expected format:
```csv
datetime,Open,High,Low,Close,volume
2010-01-04 18:00:00,1116.25,1116.75,1115.75,1116.25,234
...
```

### 2. **Python Environment**
Required packages (likely already installed):
```bash
pip install pandas numpy scipy statsmodels tqdm
```

### 3. **Disk Space**
- Database: ~100-500 MB
- Logs: ~10-50 MB
- Reports: ~1-5 MB

**Total**: <1 GB

---

## 🎯 Phase 1 Execution

### Basic Usage

```bash
# Test all 10 strategies on ES
python -m research.strategy_factory.main phase1 --symbol ES --workers 16
```

### Advanced Options

```bash
# Test specific strategies only (faster testing)
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --strategies 21 1 36 \
    --workers 16

# Custom date range
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --workers 16

# Single-threaded (for debugging)
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --workers 1
```

### What Happens During Execution

**Step 1: Raw Backtesting** (~20-40 min)
```
Strategy #21: RSI2_MeanReversion
  Testing 36 parameter combinations...
  ████████████████████████ 36/36 [02:34<00:00, 4.28s/it]
  ✓ Completed 36 backtests

Strategy #1: BollingerBands
  Testing 16 parameter combinations...
  ████████████████████████ 16/16 [01:12<00:00, 4.50s/it]
  ✓ Completed 16 backtests

... (8 more strategies)

Total: 235 backtests completed in 38.5 minutes
```

**Step 2: Filter Application** (~10-20 min)
```
Applying Gate 1 Filters...
  Gate 1: 150/235 passed

Applying Walk-Forward Validation...
  ✓ RSI2_MeanReversion passed walk-forward
  ✓ BollingerBands passed walk-forward
  ✗ MACross failed walk-forward
  Walk-Forward: 95/150 passed

Applying Regime Analysis...
  ✓ RSI2_MeanReversion passed regime analysis
  ✓ RSI2_SMAFilter passed regime analysis
  Regime: 60/95 passed

Applying Parameter Stability...
  ✓ RSI2_MeanReversion passed stability
  ✗ Double7s failed stability
  Stability: 40/60 passed

Applying Monte Carlo + FDR...
  ✓ RSI2_MeanReversion p=0.0023
  ✓ BollingerBands p=0.0041
  ✓ RSI2_SMAFilter p=0.0089
  FDR: 8/40 passed

FINAL WINNERS: 8 strategies
```

**Step 3: Results & Reporting**
```
Winning Strategies:
  1. RSI2_MeanReversion (#21): Sharpe=0.45, Trades=12,453, PF=1.32
  2. BollingerBands (#1): Sharpe=0.42, Trades=11,234, PF=1.28
  3. RSI2_SMAFilter (#36): Sharpe=0.38, Trades=14,567, PF=1.25
  ...

Results saved to database with run_id: a1b2c3d4-...
```

---

## 📊 Understanding the Results

### Filter Breakdown

| Filter | Purpose | Pass Criteria |
|--------|---------|---------------|
| **Gate 1** | Basic viability | Trades≥10k, Sharpe≥0.2, PF≥1.15, DD≤30%, WR≥35% |
| **Walk-Forward** | Out-of-sample test | Test/Train Sharpe ≥ 0.5 |
| **Regime** | Consistency | Sharpe≥0.2 in 2 of 3 regimes |
| **Stability** | Parameter robustness | <40% variation with ±10% param changes |
| **Statistical** | Significance | Monte Carlo p<0.05, FDR-corrected |

### Expected Outcomes

**Best Case** (8-12 winners):
- Several RSI(2) variants pass
- Bollinger Bands passes
- VWAP Reversion passes
- Ready for Phase 2 multi-symbol testing

**Good Case** (5-8 winners):
- 2-3 mean reversion strategies pass
- 1-2 momentum/breakout strategies pass
- Sufficient for Phase 2

**Marginal Case** (2-4 winners):
- 1-2 strategies pass all filters
- May need to adjust thresholds or test different parameters
- Still valuable for ML pipeline

**No Winners** (0):
- Filters too strict OR
- Data issues OR
- Parameter ranges suboptimal
- Review logs and adjust

---

## 🗄️ Database & Logs

### SQLite Database

Location: `research/strategy_factory/results/strategy_factory.db`

Query results:
```python
from research.strategy_factory.database import DatabaseManager

db = DatabaseManager()

# Get top strategies from a run
results = db.get_top_strategies(
    run_id='<your_run_id>',
    limit=10,
    min_sharpe=0.2,
    min_trades=10000
)

for r in results:
    print(f"{r['strategy_name']}: Sharpe={r['sharpe_ratio']:.3f}")
```

### Log File

Location: `strategy_factory.log`

Contains:
- Detailed progress for each strategy
- Filter results for each candidate
- Error messages and warnings
- Timing information

```bash
# Monitor in real-time
tail -f strategy_factory.log

# Search for errors
grep ERROR strategy_factory.log

# Find winning strategies
grep "✓.*passed" strategy_factory.log
```

---

## ⚙️ Configuration & Tuning

### Adjusting Filter Thresholds

Edit `research/strategy_factory/main.py`:

```python
# Make Gate 1 more/less strict
gate1_survivors = filter_results(
    all_results,
    min_trades=8000,      # Default: 10000
    min_sharpe=0.15,      # Default: 0.2
    min_profit_factor=1.10,  # Default: 1.15
    max_drawdown_pct=0.35,   # Default: 0.30
    min_win_rate=0.30        # Default: 0.35
)
```

### Testing Subset of Strategies

```bash
# Test only high-priority strategies
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --strategies 21 1 36 37 \
    --workers 16

# Strategies:
# 21 = RSI(2) Mean Reversion
# 1  = Bollinger Bands
# 36 = RSI(2) + SMA Filter
# 37 = Double 7s
```

### Quick Test (2023 only)

```bash
# Fast test on 1 year of data
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --strategies 21 \
    --workers 8

# Expected runtime: ~2-3 minutes
# Use this to validate setup before full run
```

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'research'**
```bash
# Solution: Run from project root
cd /opt/pine/rooney-capital-v1
python -m research.strategy_factory.main phase1 --symbol ES
```

**2. FileNotFoundError: Data file not found**
```bash
# Check data path
ls -lh data/resampled/ES_15min.csv

# If missing, check your data pipeline
```

**3. MemoryError during backtesting**
```bash
# Reduce workers
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --workers 8  # Instead of 16
```

**4. Very slow execution**
```bash
# Check CPU usage
htop

# Test with one strategy first
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --strategies 21 \
    --workers 1
```

**5. No strategies pass filters**
```bash
# Check logs
tail -100 strategy_factory.log

# Try relaxed filters (edit main.py)
# OR test different date range
# OR test different symbol
```

### Debug Mode

```python
# Edit main.py - change logging level
logging.basicConfig(
    level=logging.DEBUG,  # Was: INFO
    ...
)
```

---

## 📈 Next Steps After Phase 1

### If You Have Winners (5-10 strategies)

**1. Review Results**
```bash
# Check the report
less results/phase1_<run_id>/report.md

# Query database
python -c "
from research.strategy_factory.database import DatabaseManager
db = DatabaseManager()
results = db.get_top_strategies('<run_id>', limit=10)
for r in results:
    print(f\"{r['strategy_name']}: {r['sharpe_ratio']:.3f}\")
"
```

**2. Run Phase 2 (Multi-Symbol)**
```bash
# Test winners on all symbols
python -m research.strategy_factory.main phase2 \
    --run-id <your_run_id> \
    --symbols ES NQ YM RTY GC SI HG CL \
    --workers 16
```

**3. Extract Features for ML (Phase 3)**
```bash
# For each winning strategy
python research/extract_training_data.py \
    --strategy RSI2_MeanReversion \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31

# Train ML model
python research/train_rf_cpcv_bo.py \
    --symbol ES \
    --strategy RSI2_MeanReversion \
    --n_trials 100
```

### If You Have Few/No Winners

**1. Analyze Why**
```bash
# Check filter stats
grep "Filter Results" strategy_factory.log

# Where did most strategies fail?
# - Gate 1: Parameters need tuning
# - Walk-forward: Overfitting to training period
# - Regime: Not robust across market conditions
# - Stability: Too sensitive to parameters
# - Statistical: May be random luck
```

**2. Adjust & Retry**
- Relax filter thresholds
- Test different parameter ranges
- Try different symbols (NQ, YM, RTY)
- Use different date ranges
- Focus on specific strategy archetypes

---

## 📚 Additional Resources

**Documentation**:
- [README.md](README.md) - Usage guide
- [STRATEGY_FACTORY_GUIDE.md](../../STRATEGY_FACTORY_GUIDE.md) - Full methodology
- [STRATEGIES_COMPLETE.md](STRATEGIES_COMPLETE.md) - Strategy details
- [STATUS.md](STATUS.md) - Implementation status

**Code**:
- `strategies/` - All 10 strategy implementations
- `engine/` - Backtesting, optimization, filters
- `database/` - SQLite storage
- `reporting/` - Report generation

**Support**:
- Check `strategy_factory.log` for details
- Review STATUS.md for known issues
- Consult STRATEGY_FACTORY_GUIDE.md for methodology

---

## ✅ Pre-Flight Checklist

Before running Phase 1, verify:

- [ ] Data files exist: `ls data/resampled/ES_15min.csv`
- [ ] Python packages installed: `pip list | grep -E "(pandas|numpy|scipy)"`
- [ ] Disk space available: `df -h .` (need ~1GB)
- [ ] Running from project root: `pwd` → `/opt/pine/rooney-capital-v1`
- [ ] 16 CPU cores available: `nproc` or `lscpu`

**Ready to run?**
```bash
python -m research.strategy_factory.main phase1 \
    --symbol ES \
    --workers 16
```

---

**Status**: ✅ All code complete and tested
**Timeline**: ~30-60 minutes to complete Phase 1
**Expected**: 5-10 strategies pass all filters

Good luck! 🚀
