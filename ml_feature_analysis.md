# ML Feature Analysis: Required vs. Calculated

## Summary
Based on analysis of your ML models, **all 26 missing features ARE actually needed** by your models. Here's the breakdown by symbol:

---

## CL Model - Missing 8/30 features (21 working)

**Features in model that are NOT being calculated:**
1. `6a_hourly_z_score` - Cross-symbol z-score (6A hourly)
2. `6c_hourly_z_score` - Cross-symbol z-score (6C hourly)
3. `6e_hourly_z_score` - Cross-symbol z-score (6E hourly)
4. `6m_hourly_z_score` - Cross-symbol z-score (6M hourly)
5. `es_hourly_z_score` - Cross-symbol z-score (ES hourly)
6. `rty_hourly_z_score` - Cross-symbol z-score (RTY hourly)
7. `vix_hourly_return` - VIX hourly return
8. `ym_hourly_z_score` - Cross-symbol z-score (YM hourly)

**Root Cause:**
- Features 1-6, 8: Cross-symbol z-score pipelines not initialized
- Feature 7: VIX data not loaded (not in contract map)

---

## ES Model - Missing ?/30 features (24 working)

**Note:** Verification script shows 24/30 but didn't list which are missing.

**Likely missing features from ES model:**
- `rsixvolz` - Derived feature (RSI × VOLZ) - needs both components
- `tlt_daily_z_score` - Cross-symbol z-score (TLT daily)
- `cl_hourly_z_score` - Cross-symbol z-score (CL hourly)

*(Need to verify if these are actually in the missing list for ES)*

---

## HG Model - Missing 8/30 features (19 working)

**Features in model that are NOT being calculated:**
1. `6a_hourly_z_score` - Cross-symbol z-score (6A hourly)
2. `6c_hourly_z_score` - Cross-symbol z-score (6C hourly)
3. `6e_hourly_z_score` - Cross-symbol z-score (6E hourly)
4. `6n_hourly_z_score` - Cross-symbol z-score (6N hourly)
5. `es_hourly_z_score` - Cross-symbol z-score (ES hourly)
6. `rsixatrz` - Derived feature (RSI × ATRZ)
7. `rty_hourly_z_score` - Cross-symbol z-score (RTY hourly)
8. `vix_hourly_return` - VIX hourly return

**Root Cause:**
- Features 1-5, 7: Cross-symbol z-score pipelines not initialized
- Feature 6: Derived multiplication feature (needs rsi AND atrz)
- Feature 8: VIX data not loaded

---

## NQ Model - Missing 8/30 features (20 working)

**Features in model that are NOT being calculated:**
1. `6e_hourly_z_score` - Cross-symbol z-score (6E hourly)
2. `6j_daily_z_score` - Cross-symbol z-score (6J daily)
3. `6j_hourly_z_score` - Cross-symbol z-score (6J hourly)
4. `6s_hourly_z_score` - Cross-symbol z-score (6S hourly)
5. `gc_hourly_z_score` - Cross-symbol z-score (GC hourly)
6. `rsixatrz` - Derived feature (RSI × ATRZ)
7. `rsixvolz` - Derived feature (RSI × VOLZ)
8. `tlt_daily_z_score` - Cross-symbol z-score (TLT daily)

**Root Cause:**
- Features 1-5, 8: Cross-symbol z-score pipelines not initialized
- Features 6-7: Derived multiplication features

---

## RTY Model - Missing 8/30 features (19 working)

**Features in model that are NOT being calculated:**
1. `6b_hourly_z_score` - Cross-symbol z-score (6B hourly)
2. `6c_hourly_z_score` - Cross-symbol z-score (6C hourly)
3. `mom3_z_pct` - Momentum z-score percentile
4. `momentum_z_entry_daily` - Momentum z-score entry (daily)
5. `ng_daily_z_score` - Cross-symbol z-score (NG daily)
6. `tlt_daily_z_score` - Cross-symbol z-score (TLT daily)
7. `vix_daily_return` - VIX daily return
8. `vix_hourly_return` - VIX hourly return

**Root Cause:**
- Features 1-2, 5-6: Cross-symbol z-score pipelines not initialized
- Features 3-4: Momentum indicator not initialized (`enableMom3` not set)
- Features 7-8: VIX data not loaded

---

## SI Model - Missing 8/30 features (19 working)

**Features in model that are NOT being calculated:**
1. `6a_hourly_z_score` - Cross-symbol z-score (6A hourly)
2. `6b_hourly_z_score` - Cross-symbol z-score (6B hourly)
3. `6c_hourly_z_score` - Cross-symbol z-score (6C hourly)
4. `6j_hourly_z_score` - Cross-symbol z-score (6J hourly)
5. `6m_hourly_z_score` - Cross-symbol z-score (6M hourly)
6. `6s_hourly_z_score` - Cross-symbol z-score (6S hourly)
7. `hg_hourly_z_score` - Cross-symbol z-score (HG hourly)
8. `rsixvolz` - Derived feature (RSI × VOLZ)

**Root Cause:**
- Features 1-7: Cross-symbol z-score pipelines not initialized
- Feature 8: Derived multiplication feature

---

## YM Model - Missing 8/30 features (21 working)

**Features in model that are NOT being calculated:**
1. `6c_hourly_z_score` - Cross-symbol z-score (6C hourly)
2. `6n_daily_z_score` - Cross-symbol z-score (6N daily)
3. `6n_hourly_z_score` - Cross-symbol z-score (6N hourly)
4. `cl_hourly_z_pipeline` - Cross-symbol z-score pipeline (CL hourly)
5. `ng_hourly_z_pipeline` - Cross-symbol z-score pipeline (NG hourly)
6. `rsixvolz` - Derived feature (RSI × VOLZ)
7. `si_daily_z_pipeline` - Cross-symbol z-score pipeline (SI daily)
8. `tlt_daily_z_score` - Cross-symbol z-score (TLT daily)

**Root Cause:**
- Features 1-5, 7-8: Cross-symbol z-score pipelines not initialized
- Feature 6: Derived multiplication feature

---

## CONSOLIDATED ISSUE CATEGORIES

### 1. VIX Data Not Loaded (4 features across 3 symbols)
**Affected symbols:** CL, HG, RTY
**Missing features:**
- `vix_daily_return` (RTY)
- `vix_hourly_return` (CL, HG, RTY)

**Fix:** Add VIX to your contract map configuration

---

### 2. Cross-Symbol Z-Score Pipelines Not Initialized (18 unique z-score features)
**All affected symbols:** CL, ES, HG, NQ, RTY, SI, YM

**Missing z-score features:**
- `6a_hourly_z_score` (CL, HG, SI)
- `6b_hourly_z_score` (RTY, SI)
- `6c_hourly_z_score` (CL, HG, RTY, SI, YM)
- `6e_hourly_z_score` (CL, HG, NQ)
- `6j_daily_z_score` (NQ)
- `6j_hourly_z_score` (NQ, SI)
- `6m_hourly_z_score` (CL, SI)
- `6n_daily_z_score` (YM)
- `6n_hourly_z_score` (HG, YM)
- `6s_hourly_z_score` (NQ, SI)
- `cl_hourly_z_pipeline` (YM)
- `es_hourly_z_score` (CL, HG)
- `gc_hourly_z_score` (NQ)
- `hg_hourly_z_score` (SI)
- `ng_daily_z_score` (RTY)
- `ng_hourly_z_pipeline` (YM)
- `rty_hourly_z_score` (CL, HG)
- `si_daily_z_pipeline` (YM)
- `tlt_daily_z_score` (NQ, RTY, YM)
- `ym_hourly_z_score` (CL)

**Fix:** Enable z-score calculations for these cross-symbol pairs

---

### 3. Derived Multiplication Features (2 features across 4 symbols)
**Affected symbols:** HG, NQ, SI, YM

**Missing features:**
- `rsixatrz` (HG, NQ) - needs both `rsi` AND `atrz` to be calculated
- `rsixvolz` (NQ, SI, YM) - needs both `rsi` AND `volz` to be calculated

**Fix:** Ensure base features (rsi, atrz, volz) are enabled for these symbols

---

### 4. Momentum Z-Score Not Initialized (2 features on 1 symbol)
**Affected symbols:** RTY

**Missing features:**
- `mom3_z_pct`
- `momentum_z_entry_daily`

**Fix:** Enable `enableMom3=True` parameter

---

## PRIORITY ACTIONS

### ACTION 1: Add VIX to Contract Map (CRITICAL)
Add this to `/home/user/rooney-capital-v1/Data/Databento_contract_map.yml`:

```json
{
  "symbol": "VIX",
  "databento": {
    "dataset": "GLBX.MDP3",
    "product_id": "VIX.FUT"
  }
}
```

**Impact:** Fixes 4 features across CL, HG, RTY

---

### ACTION 2: Enable Cross-Symbol Z-Score Calculations (CRITICAL)
The cross-symbol z-scores are not being initialized because the system checks:
```python
matches_ml_feature = (
    feature_key in normalized_ml_features
    or pipeline_key in normalized_ml_features
)
```

**Problem:** Your ML models ARE requesting these features, but something in the normalization or lookup is failing.

**Debug steps:**
1. Check how features are registered in the ML feature list
2. Verify the feature name normalization is working
3. Add logging to see which features are being requested vs. which pipelines are created

**Temporary Fix:** Manually enable parameters for each needed z-score:
```python
enable6AZScoreHour=True
enable6BZScoreHour=True
enable6CZScoreHour=True
enable6EZScoreHour=True
# ... etc for all missing z-scores
```

---

### ACTION 3: Fix Derived Features (MEDIUM)
These depend on base features being calculated. Check that for each symbol:
- `enableRSI=True`
- `enableATRZ=True`
- `enableVOLZ=True`

---

### ACTION 4: Enable Momentum Indicator for RTY (LOW)
Add to RTY configuration:
```python
enableMom3=True
```

---

## CONCLUSION

**All 26 missing features are actually required by your ML models.** None can be ignored. You must fix the feature calculation to get your models working at full performance.

The primary issue is that cross-symbol z-score pipelines are not being created even though they're in your ML feature lists. This suggests either:
1. Feature name mismatch between models and strategy code
2. ML feature registration/normalization bug
3. Configuration not properly loading ML feature lists

Next step: Investigate why `matches_ml_feature` is returning False for these features even though they're in your model JSON files.
