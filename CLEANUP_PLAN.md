# Repository Cleanup Plan

## 📋 Overview
This document outlines files/docs to archive or remove to streamline the repository.

---

## 📚 Documentation Analysis

### ✅ KEEP - Core Documentation
- `README.md` - Main project documentation
- `QUICK_START.md` - User getting started guide
- `SYSTEM_GUIDE.md` - System architecture overview
- `THREE_WAY_SPLIT_GUIDE.md` - Current training methodology
- `END_TO_END_OPTIMIZATION_GUIDE.md` - Comprehensive optimization guide
- `LIVE_LAUNCH_GUIDE.md` - Production deployment guide
- `MONITORING_GUIDE.md` - System monitoring
- `PORTFOLIO_INTEGRATION_GUIDE.md` - Portfolio construction
- `INDICATOR_PARAMETERS.md` - Strategy parameters reference

### 🗄️ ARCHIVE - Historical/Debugging Docs (Move to `docs/archive/`)
These were useful for debugging but are no longer needed for day-to-day work:
- `EXPERT_ISSUES_FIXED.md` - Historical issue log
- `EXPERT_REVIEW_RESPONSES.md` - Historical review responses
- `FEATURE_SELECTION_FIX_SUMMARY.md` - Historical fix log
- `FIX_ML_FEATURES.md` - Historical fix log
- `ML_FIXES_SUMMARY.md` - Historical fix log
- `OPTIMIZATION_FIXES_ANALYSIS.md` - Historical fix log
- `OPTIMIZATION_FIXES_IMPLEMENTED.md` - Historical fix log
- `OPTIMIZATION_VERIFICATION_REPORT.md` - Historical verification
- `LOOK_AHEAD_BIAS_REVIEW.md` - Historical review
- `feature_verification_report.txt` - Historical verification
- `ml_feature_analysis.md` - Historical analysis
- `worker_full_log.txt` - Old log file

### ❓ REVIEW - Potentially Outdated
Need to verify if still relevant:
- `DATA_TRANSFORMATION_TO_ML_WORKFLOW.md` - Is this current?
- `CONFIGURATION_B_SETUP.md` - Is Configuration B still used?
- `CLUSTERED_FEATURE_SELECTION_GUIDE.md` - Is clustering still used?

---

## 🐍 Python Scripts Analysis

### Current Training Pipeline
- `extract_training_data.py` - ✅ Extract features from backtests
- `train_rf_three_way_split.py` - ✅ **CURRENT** training script
- `rf_cpcv_random_then_bo.py` - ⚠️ OLD training script (keep for reference?)
- `train_rf_cpcv_bo.py` - ⚠️ OLD training script (archive?)

### Portfolio Optimization
- `portfolio_optimizer_greedy_train_test.py` - ✅ **CURRENT** greedy optimizer
- `generate_portfolio_backtest_data.py` - ✅ **CURRENT** backtest generator (just fixed!)
- `portfolio_optimizer_full.py` - ❓ Is this used?
- `portfolio_optimizer_simple.py` - ❓ Is this used?
- `portfolio_optimizer_train_test.py` - ❓ Is this used?
- `optimize_portfolio_positions.py` - ❓ Is this used?
- `portfolio_constructor.py` - ❓ Is this used?
- `portfolio_performance_analysis.py` - ❓ Is this used?
- `portfolio_simulator.py` - ❓ Is this used?

### Utility Scripts
- `backtest_runner.py` - ❓ Is this still used?
- `check_trade_dates.py` - ❓ One-off diagnostic?
- `concatenate_chunks.py` - ❓ Data processing?
- `diagnose_entries.py` - ❓ One-off diagnostic?
- `export_portfolio_config.py` - ❓ Is this used?
- `extract_symbol_sharpes.py` - ❓ Is this used?
- `filter_trades_with_ml.py` - ⚠️ Created today but won't work (incompatible CSVs)

### Keep
- `daily_utils.py` - Utility functions
- `feature_utils.py` - Utility functions
- `production_retraining.py` - Production ML retraining

---

## 📁 Directories to Check

### Data Directories
- `Data/` (capitalized) - ❓ What's in here? Should it be lowercase?
- `data/` (if exists) - Current data directory?
- `dashboard/` - ❓ Is this used?
- `deployment/` - ❓ vs `deploy/` - duplicates?

---

## 🎯 Recommended Actions

### Phase 1: Archive Historical Docs
```bash
mkdir -p docs/archive
mv EXPERT_*.md docs/archive/
mv *_FIXES_*.md docs/archive/
mv *_VERIFICATION_*.md docs/archive/
mv LOOK_AHEAD_BIAS_REVIEW.md docs/archive/
mv feature_verification_report.txt docs/archive/
mv ml_feature_analysis.md docs/archive/
mv worker_full_log.txt docs/archive/
```

### Phase 2: Review & Document Current Scripts
Create a `research/README.md` explaining:
- Which scripts are current vs deprecated
- What each script does
- Recommended workflow

### Phase 3: Clean Up Duplicates
- Review portfolio optimizer scripts - keep only current one
- Review data directories - consolidate if duplicated
- Remove `filter_trades_with_ml.py` (doesn't work with current setup)

---

## ❓ Questions for User

1. **Configuration B**: Is `CONFIGURATION_B_SETUP.md` still relevant?
2. **Clustering**: Is `CLUSTERED_FEATURE_SELECTION_GUIDE.md` still used?
3. **Portfolio Scripts**: Which portfolio optimizer are you actually using?
4. **Data Directories**: What's the difference between `Data/` and `data/`?
5. **Dashboard**: Is the `dashboard/` directory being used?
6. **Old Training Scripts**: Keep `rf_cpcv_random_then_bo.py` for reference or archive it?

---

## 📊 Summary

**Total Files:**
- 21 markdown files in root
- 24 Python scripts in research/
- Multiple shell scripts

**Recommendation:**
- Archive: ~12 historical docs
- Review: ~3 potentially outdated docs
- Review: ~10 potentially unused scripts
- Keep: ~6 core docs + current pipeline scripts

This could reduce root directory clutter by ~50-60%!
