# Extract Training Data Audit

**Date:** 2025-11-11
**Script:** `research/extract_training_data.py`
**Purpose:** Identify lookahead bias, data leakage, and unrealistic assumptions

---

## ✅ Issues Fixed

### 1. ✅ Feature Capture Timing (CRITICAL - FIXED)
**Line 273:** `features = self.collect_filter_values(intraday_ago=-1)`

**Timeline:**
- Bar N (10:00 AM): IBS < 0.2, call `self.buy()`
- Bar N+1 (11:00 AM): Order executes at open, `notify_order()` called
- **Fix:** `intraday_ago=-1` captures features from Bar N (decision time)

**Status:** ✅ FIXED - Features now captured from correct bar

---

## ⚠️ Issues Identified

### 2. ⚠️ Entry Timestamp Mismatch (MINOR)
**Lines 274, 328:**
```python
entry_time = bt.num2date(self.hourly.datetime[0])  # Bar N+1 timestamp
'Date/Time': entry_time  # Records 11:00 AM
```

**Issue:**
- Features captured from Bar N (10:00 AM)
- Timestamp recorded as Bar N+1 (11:00 AM)
- Mismatch: timestamp is 1 hour after feature calculation time

**Impact:** Minor - doesn't affect ML training (features are correct), but timestamp is inconsistent

**Recommendation:**
```python
# Capture timestamp from same bar as features
entry_time = bt.num2date(self.hourly.datetime[-1])  # Bar N timestamp
```

**Priority:** LOW (cosmetic issue, doesn't affect training results)

---

### 3. 🚨 Missing Warmup Period for Reference Symbols (CRITICAL)
**Lines 428-435:**
```python
hourly_df, daily_df = load_symbol_data(
    ref_symbol,
    data_dir=data_dir,
    start_date=start_date,  # ← NO WARMUP!
    end_date=end_date
)
```

**Issue:**
- Reference symbols loaded from `start_date` (same as primary symbol)
- Cross-asset features (ES_z_score_hour, NQ_daily_return) will be NaN at start
- No historical data for percentile calculations
- Same bug we found in `generate_portfolio_backtest_data.py`!

**Impact:** HIGH - Early trades in training data have incomplete cross-asset features

**Example:**
- If extracting 2010-2024, reference symbols load from 2010-01-01
- But cross-asset z-scores need 252+ bars of history
- First ~252 bars (1 year) will have NaN/incomplete cross-asset features
- ML model trains on bad data for ~5-10% of dataset

**Fix Required:**
```python
# Calculate warmup start (252 trading days = ~1 year)
from datetime import datetime, timedelta
start_dt = datetime.strptime(start_date, '%Y-%m-%d')
warmup_start_dt = start_dt - timedelta(days=252 * 7 // 5)  # Account for weekends
warmup_start_date = warmup_start_dt.strftime('%Y-%m-%d')

# Load reference symbols from warmup start
hourly_df, daily_df = load_symbol_data(
    ref_symbol,
    data_dir=data_dir,
    start_date=warmup_start_date,  # ← FIXED! Includes warmup
    end_date=end_date
)
```

**Priority:** HIGH (affects training data quality)

---

### 4. ✅ Commission Calculation (CORRECT)
**Lines 318-321:**
```python
commission_total = 2.00  # $1.00 entry + $1.00 exit
pnl_usd = gross_pnl - commission_total
```

**Status:** ✅ CORRECT
- $1.00 per side = $2.00 total per round trip
- Matches production trading costs

---

### 5. ✅ Slippage Handling (CORRECT)
**Line 404:**
```python
cerebro.broker.set_slippage_fixed(tick_size)  # 1 tick per side
```

**Status:** ✅ CORRECT
- Backtrader applies slippage to executed prices automatically
- 1 tick per side matches realistic execution
- Slippage already included in `order.executed.price`
- No additional deduction needed

---

### 6. ✅ Binary Label Calculation (CORRECT)
**Lines 323-324:**
```python
binary = 1 if pnl_usd > 0 else 0
```

**Status:** ✅ CORRECT
- Based on NET PnL (after commissions and slippage)
- Realistic: only profitable after costs = winner
- 0% PnL (breakeven) = loser (conservative)

---

### 7. ✅ Entry Filtering Logic (CORRECT)
**Lines 233-257:**
```python
def entry_allowed(self, dt, ibs_val: float) -> bool:
    # Check session time only
    if not self.in_session(dt):
        return False

    # Check base IBS entry range only
    if not (self.p.ibs_entry_low <= ibs_val <= self.p.ibs_entry_high):
        return False

    # All other filters bypassed!
    return True
```

**Status:** ✅ CORRECT
- Captures ALL base IBS trades (winners and losers)
- Necessary for ML training (need both classes)
- Filters are still CALCULATED (via `collect_filter_values()`)
- Filters just don't BLOCK entries during extraction

---

### 8. ✅ Cheat-on-Close Disabled (CORRECT)
**Lines 406-409:**
```python
# NOTE: Cheat-on-close is DISABLED (default Backtrader behavior)
# Orders fill at NEXT bar's open price, matching live trading execution
```

**Status:** ✅ CORRECT
- Realistic execution: orders fill at next bar open
- Matches live trading (can't fill at current bar close)
- No lookahead bias from execution timing

---

### 9. ✅ Unlimited Capital (CORRECT FOR EXTRACTION)
**Lines 390-393:**
```python
cerebro.broker.setcash(1_000_000_000.0)  # $1 billion - no margin issues!
```

**Status:** ✅ CORRECT FOR EXTRACTION
- Purpose: Capture ALL base IBS signals without capital constraints
- Realistic for training data (we want all potential trades)
- Portfolio constraints applied later during optimization
- ML model learns "is this trade good?" not "do I have enough capital?"

---

## 🔍 Filter Calculation Review

Now let me check if `collect_filter_values()` itself has any lookahead bias issues...

### Cross-Asset Features
**Lines 153-167:** Cross-asset symbols defined
- ES, NQ, RTY, YM, GC, SI, HG, CL, NG, PL, currencies, TLT
- Z-scores: `{symbol}_z_score_hour`, `{symbol}_z_score_day`
- Returns: `{symbol}_hourly_return`, `{symbol}_daily_return`

**Question:** Are cross-asset returns calculated correctly?
- Need to verify `collect_filter_values()` in `ibs_strategy.py` uses proper historical data
- Returns should be Bar N-1 → Bar N (not Bar N → Bar N+1)

**Action:** Review `src/strategy/ibs_strategy.py::collect_filter_values()` separately

---

## Summary of Required Fixes

### Priority 1 (CRITICAL)
1. ✅ **Feature capture timing** - FIXED (line 273)
2. ⚠️ **Missing warmup for reference symbols** - NEEDS FIX (lines 428-435)

### Priority 2 (MINOR)
3. ⚠️ **Entry timestamp mismatch** - OPTIONAL FIX (line 274)

### Priority 3 (VERIFICATION)
4. 🔍 **Review collect_filter_values()** - Need to audit main strategy file

---

## Recommendations

### Immediate Actions (Before Retraining)
1. ✅ Apply feature capture timing fix (already done)
2. ⚠️ Add 252-day warmup period for reference symbols
3. ⚠️ Fix entry timestamp to match feature calculation time (optional)

### Verification Actions
1. 🔍 Audit `src/strategy/ibs_strategy.py::collect_filter_values()` for lookahead bias
2. 🔍 Verify cross-asset return calculations use correct historical bars
3. 🔍 Check percentile calculations don't peek into future

### Testing Protocol
After applying fixes:
1. Extract training data for NG (2010-2024)
2. Check first 500 trades for NaN cross-asset features
3. Verify timestamp consistency (entry_time should be decision bar)
4. Compare PnL from training CSV vs backtest (should match)

---

**Last Updated:** 2025-11-11
**Status:** 1 critical fix applied, 1 critical fix pending, 1 minor fix optional
