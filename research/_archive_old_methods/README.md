# Archived Training Methods

This directory contains deprecated training scripts that have been superseded by better approaches.

## Deprecated: rf_cpcv_random_then_bo.py

**Status:** ❌ DEPRECATED - Do not use
**Replaced by:** `train_rf_three_way_split.py`
**Date archived:** 2025-11-17

### Why Deprecated?

The `rf_cpcv_random_then_bo.py` script had a fundamental flaw: **threshold optimization bias**.

**Problem:**
- Used same CPCV folds for BOTH hyperparameter tuning AND threshold optimization
- This "double-dipping" caused 5-15% optimistic bias in reported Sharpe ratios
- Live performance consistently underperformed backtest estimates

**Solution:**
Use `train_rf_three_way_split.py` instead, which uses:
- 2010-2018: Hyperparameter tuning (Random Search + Bayesian Opt)
- 2019-2020: Threshold optimization (separate data)
- 2021-2024: Final evaluation (completely untouched)

### Migration Instructions

**Old command:**
```bash
python research/rf_cpcv_random_then_bo.py \
    --symbol ES \
    --data data/training/ES_transformed_features.csv \
    --output src/models
```

**New command:**
```bash
python research/train_rf_three_way_split.py \
    --symbol ES \
    --train-end 2018-12-31 \
    --threshold-end 2020-12-31 \
    --rs-trials 120 \
    --bo-trials 300
```

### Documentation

See `THREE_WAY_SPLIT_GUIDE.md` for full documentation of the new approach.

### Old Model Archives

Old models trained with `rf_cpcv_random_then_bo.py` have been archived to:
```
src/models/_archive_old_cpcv_20251117/
```

These models should NOT be used in production due to optimistic bias.

---

**Important:** If you see references to `rf_cpcv_random_then_bo.py` in documentation, those guides are outdated. Always use `train_rf_three_way_split.py` for new model training.
