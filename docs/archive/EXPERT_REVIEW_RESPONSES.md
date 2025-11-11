# Expert Review: Optimization Questions & Responses

**Date**: 2025-10-30
**Context**: External expert review of optimization methodology

---

## Question 1: Label Generation Timing

### The Question

> When you create your binary labels (1=profitable, 0=loss) for the Random Forest training data in `data/training/{SYMBOL}_transformed_features.csv`, are these labels generated using the triple-barrier method during backtesting and then saved? Or are they computed separately? Most importantly: is there any possibility that label calculation uses information from bars that would be in "purged" regions during CPCV?

### Our Current Implementation

**Label Generation Method**: **Live backtest outcomes** (NOT triple-barrier)

**Process** (`research/extract_training_data.py`, lines 268-336):

1. **Entry Time** (line 269): Features captured when entry order fills
   ```python
   features = self.collect_filter_values(intraday_ago=0)
   ```

2. **Exit Time** (lines 287-320): Label calculated when exit order fills
   ```python
   # Calculate outcomes when position closes
   price_return = (exit_price - entry_price) / entry_price
   pnl_usd = gross_pnl - commission_total  # Net PnL after costs
   binary = 1 if pnl_usd > 0 else 0  # Binary label
   ```

3. **Timeline**:
   - Entry: Day 1 (features captured)
   - Exit: Day 1-3 (max hold = 2 days, so exit by Day 3)
   - Label finalized: Day 3

**Not Triple-Barrier**: We use actual trade outcomes from strategy exits (IBS exit, trailing stop, or max hold), not synthetic barriers.

### Critical Issue Identified: ⚠️ LABEL LEAKAGE RISK

**Problem**: Our CPCV purge uses **ENTRY dates**, but labels aren't finalized until **EXIT dates**.

**Code Evidence** (`rf_cpcv_random_then_bo.py:443`):
```python
embargoed_cpcv_splits(Xy["Date"], folds, k_test, embargo_days)
#                     ↑ ENTRY dates only!
```

**Scenario Where This Causes Leakage**:
```
Test fold: Days 100-110
Purge window: Days 97-99 (3-day embargo from entry)
Problematic trade: Entry on Day 96, Exit on Day 98

Current behavior:
- Day 96 entry is INCLUDED in training (> 3 days before test)
- But exit on Day 98 is WITHIN the purge window
- Label (finalized Day 98) contaminates the purge zone
```

**Impact**: Positions that EXIT near test fold boundaries leak information into the purge window.

### Our Status

✅ **No look-ahead bias in label calculation itself** (labels use only data up to exit time)
⚠️ **Potential label leakage** due to entry-date-based purge with delayed label finalization

### Recommended Fix

**Option A: Use Exit Dates for Purge** (Preferred)
```python
# Instead of:
embargoed_cpcv_splits(Xy["Date"], ...)  # Entry dates

# Use:
embargoed_cpcv_splits(Xy["Exit Date/Time"], ...)  # Exit dates
```

**Option B: Increase Embargo by Max Hold**
```python
MAX_HOLD_DAYS = 2  # From strategy
embargoed_cpcv_splits(Xy["Date"], embargo_days=3 + MAX_HOLD_DAYS)  # 5 days total
```

**Current Mitigation**: With 3-day embargo and typical 1-2 day holds, we have ~1 day buffer after most exits. This is **marginally safe** but should be made **explicitly robust**.

---

## Question 2: Hyperparameter Search Space Boundaries

### The Question

> In your Bayesian optimization phase that "narrows the search space to high-performing regions from random search" - how exactly are you constraining it? Are you setting hard bounds (e.g., only test n_estimators between 600-900), or are you using the random search results to set priors/weights in the Bayesian sampler? The former could cause you to miss the global optimum if your random search happened to find a local maximum.

### Our Current Implementation

**Answer**: We are setting **HARD BOUNDS** based on Random Search results.

**Code Evidence** (`rf_cpcv_random_then_bo.py`, lines 1099-1142):

```python
# Take top 10% of random search results (minimum 5)
top = rs_df.head(max(5, max(1, rs_trials // 10)))  # Line 1099

# Extract ONLY hyperparameters that appeared in top results
est_range = sorted(set(int(x) for x in top["n_estimators"]))  # Line 1100
depth_opts = sorted(depth_set)  # Line 1109
leaf_range = sorted(set(int(x) for x in top["min_samples_leaf"]))  # Line 1110
mf_opts = ...  # Lines 1112-1122
boot_opts = sorted(set(bool(x) for x in top["bootstrap"]))  # Line 1123
cs_opts = ...  # Lines 1124-1131

# Bayesian optimization objective ONLY samples from these values
def objective(trial: optuna.Trial):
    n_estimators = trial.suggest_categorical("n_estimators", est_range)  # HARD BOUND
    max_depth = trial.suggest_categorical("max_depth", depth_opts)  # HARD BOUND
    min_leaf = trial.suggest_categorical("min_samples_leaf", leaf_range)  # HARD BOUND
    # ...
```

**Example**:
- Random Search trials: 120
- Top 10%: 12 trials
- If those 12 trials only tested `n_estimators = [600, 900, 1200]`
- Bayesian optimization will **NEVER** try `n_estimators = 300` or `1500`
- Even if 300 or 1500 might be optimal!

### Critical Issue: 🚨 POTENTIAL GLOBAL OPTIMUM MISSED

**Problem**: Hard bounds mean Bayesian optimization **cannot explore** regions not sampled by top Random Search results.

**Risk**: If Random Search finds a **local maximum**, Bayesian optimization will refine that local maximum instead of finding the global maximum.

**Likelihood**:
- With 120 Random Search trials and 8 hyperparameters, coverage is sparse
- Random Search may miss the best region entirely
- Example: If best `n_estimators=500` but Random Search only tried [600, 900, 1200], Bayesian optimization will never find 500

### Why We Did This (Original Intent)

**Goal**: Focus Bayesian optimization on "promising regions" from Random Search to be more efficient.

**Problem**: This assumes Random Search found the RIGHT promising region, which is not guaranteed with only 120 trials.

### Recommended Fixes

**Option 1: Remove Hard Bounds** (Safest)
```python
# Use FULL hyperparameter space for Bayesian optimization
def objective(trial: optuna.Trial):
    # Use original ranges, not top results
    n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500, 700, 900, 1200])
    max_depth = trial.suggest_categorical("max_depth", [3, 5, 7, 10, 15, None])
    # ... full space
```
- Pro: Guaranteed not to miss global optimum
- Con: Bayesian optimization less focused, may be slower to converge

**Option 2: Use Priors/Weights Instead of Hard Bounds**
```python
# Give higher probability to top Random Search regions, but don't exclude others
# Requires custom Optuna sampler with weighted priors
# More complex to implement but preserves exploration
```
- Pro: Focuses on promising regions while keeping global search possible
- Con: More complex implementation, not standard Optuna

**Option 3: Expand Bounds from Top Results**
```python
# Instead of ONLY top values, include neighbors
top = rs_df.head(max(5, rs_trials // 10))
est_range = sorted(set(int(x) for x in top["n_estimators"]))

# EXPAND to include neighbors
est_min, est_max = min(est_range), max(est_range)
est_expanded = [e for e in FULL_EST_RANGE if est_min * 0.5 <= e <= est_max * 2]
```
- Pro: Maintains some focus while allowing nearby exploration
- Con: Still arbitrary bounds, could still miss global optimum

**Option 4: Two-Stage Bayesian** (Hybrid)
```python
# Stage 1: Bayesian with hard bounds (100 trials) - fast convergence
# Stage 2: Bayesian with full space (200 trials) - global search
```
- Pro: Gets both focused refinement AND global exploration
- Con: More complex, longer runtime

### Current Status

🚨 **HIGH RISK**: Hard bounds from top 10% of 120 Random Search trials
- With 8 hyperparameters and sparse sampling, we may be trapped in local optimum
- Bayesian optimization cannot recover from bad Random Search luck

### Priority

**HIGH** - This could significantly impact model quality. If Random Search misses the best region, we're stuck.

---

## Question 3: Production Retraining Cadence

### The Question

> You mentioned quarterly retraining in the context of threshold updates - but when you retrain in production, are you re-running the full 420-trial optimization (120 random + 300 Bayesian), or are you doing something lighter? If you're re-optimizing hyperparameters each quarter on expanding windows, you may be introducing forward-looking bias through parameter selection that "knows" about recent market regimes.

### Our Current Implementation

**Answer**: **No production retraining cadence implemented yet.**

**Status**: Production retraining is documented in the roadmap (`docs/ml_pipeline_roadmap.md`, Phase 7) but **NOT YET IMPLEMENTED**.

**Roadmap Documentation**:
```
Phase 7: Production Monitoring 📡 (Ongoing)
Status: 🔲 Not Started

Retraining triggers:
- [ ] Calendar-based (every 6 months)
- [ ] Performance-based (accuracy drops 10%)
- [ ] Market regime change detection
```

**Key Observations**:
1. Roadmap suggests **6 months** (not quarterly)
2. No code currently exists for automated retraining
3. No specification of whether retraining would be full 420-trial optimization or lighter

### The Expert's Concern: Forward-Looking Bias

**Scenario**: If we retrain quarterly with full optimization on expanding windows

```
Timeline:
Q1 2024: Optimize on 2010-2024 Q1 → Deploy model A
Q2 2024: Optimize on 2010-2024 Q2 → Deploy model B (knows Q2 data!)
Q3 2024: Optimize on 2010-2024 Q3 → Deploy model C (knows Q3 data!)

Problem: Model C's hyperparameters were selected KNOWING about Q2-Q3 data,
even though it's supposedly trading "live" during Q3.
```

**This is a form of data snooping**: Hyperparameters chosen with knowledge of "future" market regimes.

### Recommended Approach: Walk-Forward Optimization

To avoid forward-looking bias in production retraining:

#### Option A: Fixed Hyperparameters, Retrain Weights Only

```python
# Every 6 months:
# 1. Keep hyperparameters FIXED (from original optimization)
# 2. Only retrain Random Forest weights on expanding window
# 3. Optionally re-optimize threshold on most recent year

Pros:
- No forward-looking bias in hyperparameter selection
- Fast (no 420-trial optimization)
- Model adapts to new data patterns

Cons:
- Hyperparameters may become stale over many years
- Doesn't adapt to regime changes in optimal hyperparameters
```

#### Option B: Walk-Forward Re-Optimization (Anchored)

```python
# Every 6 months:
# 1. Re-run full 420-trial optimization on data UP TO retraining date
# 2. Deploy new model for NEXT period

Example:
- Jan 2024: Optimize on 2010-2023 → Deploy for 2024 Q1-Q2
- Jul 2024: Optimize on 2010-2024 Jun → Deploy for 2024 Q3-Q4
- Jan 2025: Optimize on 2010-2024 → Deploy for 2025 Q1-Q2

Pros:
- Hyperparameters adapt to market evolution
- No forward-looking bias (optimization window ends BEFORE deployment)

Cons:
- Computationally expensive (420 trials every 6 months)
- Potential overfitting to recent data if optimization window gets very long
```

#### Option C: Walk-Forward with Rolling Window

```python
# Every 6 months:
# 1. Re-run optimization on ROLLING 10-year window ending at retraining date
# 2. Deploy for next period

Example:
- Jan 2024: Optimize on 2014-2023 → Deploy for 2024 Q1-Q2
- Jul 2024: Optimize on 2014.5-2024 Jun → Deploy for 2024 Q3-Q4

Pros:
- Adapts to regime changes (old data drops off)
- Prevents overfitting to very long histories
- No forward-looking bias

Cons:
- Loses very old data (may be useful for rare events)
- Computationally expensive
```

#### Option D: Lightweight Retraining + Periodic Re-Optimization

```python
# Every 1 month: Retrain weights only (fast)
# Every 12 months: Full 420-trial re-optimization (anchored, no look-ahead)

Pros:
- Frequent adaptation via weight retraining
- Infrequent but thorough hyperparameter refresh
- Practical balance of computational cost vs adaptiveness

Cons:
- More complex to implement
- Annual re-optimization may be too infrequent for fast-changing markets
```

### Our Recommendation

**For Your System**: **Option D (Hybrid)** seems most practical:

1. **Monthly**: Retrain Random Forest weights + re-optimize threshold
   - Use existing hyperparameters
   - Retrain on expanding window (or rolling 10-year window)
   - Re-optimize threshold on most recent 2 years
   - Fast (~10 minutes per symbol)

2. **Annually**: Full hyperparameter re-optimization
   - Run full 420-trial optimization on anchored window (ending at retraining date)
   - Deploy for NEXT year
   - No forward-looking bias
   - Slow (~8 hours per symbol)

3. **Ad-hoc**: Performance-triggered re-optimization
   - If Sharpe ratio drops below threshold (e.g., < 0.3 for 3 months)
   - Run full optimization to check if new hyperparameters help
   - Only deploy if OOS validation shows improvement

### Critical Implementation Detail

**MUST use anchored or rolling windows** - Never optimize on data that includes the deployment period!

```python
# CORRECT (Anchored):
optimize_on_data(start="2010-01-01", end="2023-12-31")
deploy_for_period(start="2024-01-01")

# WRONG (Look-ahead bias):
optimize_on_data(start="2010-01-01", end="2024-06-30")
deploy_for_period(start="2024-01-01")  # Optimization "knew" about Jan-Jun 2024!
```

### Current Status

✅ **No forward-looking bias currently** (no retraining implemented)
⚠️ **Must implement carefully** when productionizing

---

## Summary of Issues & Priorities

| Issue | Severity | Impact | Recommended Action |
|-------|----------|--------|-------------------|
| **1. Label leakage (entry vs exit dates)** | 🟡 Medium | Marginal bias (~1-3%) | Increase embargo to 5 days OR use exit dates |
| **2. Hard bounds in Bayesian optimization** | 🔴 High | May miss global optimum | Remove hard bounds, use full space |
| **3. Forward-looking bias in retraining** | 🟢 Low | Not implemented yet | Design anchored walk-forward when implementing |

## Next Steps

### Immediate (This Week)
1. **Fix Bayesian optimization bounds** (Question 2)
   - Remove hard bounds derived from top Random Search results
   - Use full hyperparameter space for Bayesian optimization
   - Re-run optimization on key symbols to validate

2. **Increase embargo period** (Question 1)
   - Change `embargo_days=3` to `embargo_days=5` (max_hold + buffer)
   - OR implement exit-date-based purge
   - Re-validate that OOS Sharpe remains stable

### Short-Term (Next Month)
3. **Document production retraining policy** (Question 3)
   - Specify: monthly weight retraining + annual hyperparameter re-optimization
   - Implement anchored walk-forward framework
   - Add validation that no future data leaks into optimization

### Validation
4. **Test impact of fixes**
   - Re-run optimization with unbounded Bayesian search
   - Compare OOS Sharpe with 3-day vs 5-day embargo
   - Verify no suspicious improvements (would indicate previous leakage)

---

## Expert Feedback Welcome

This analysis reveals legitimate concerns. We appreciate the expert's thorough review and recommend addressing these issues before production deployment.

**Questions for Expert**:
1. Do you agree with priority ranking (Bayesian bounds = High, Label leakage = Medium)?
2. For Question 2, do you prefer Option 1 (remove bounds) or Option 4 (two-stage)?
3. For Question 3, does the hybrid approach (monthly retrain weights + annual re-optimize hyperparameters) sound reasonable?

---

**Document Version**: 1.0
**Author**: Claude Code + Rooney Capital
**Date**: 2025-10-30
