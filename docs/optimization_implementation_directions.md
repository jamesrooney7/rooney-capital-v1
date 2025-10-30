# Meta-Labeling Optimization - Implementation Directions
## Rooney Capital Trading System v2.0

**Date:** October 30, 2025  
**Objective:** Reduce overfitting, improve statistical validity, boost performance by 30-40%

---

## 🎯 Expected Outcomes

- **60% reduction in DSR penalty** (N_eff from ~126-168 to ~27-48)
- **Eliminate 5-15% optimistic bias** from threshold optimization
- **+12-20% performance improvement** through ensembling
- **~75% reduction in training time** (420 → 90-120 trials)
- **+23% increase in training data** (5-day → 1-2 day embargo)

---

## 📁 Directory Structure Updates

### New Files to Create:
```
src/
├── models/ensemble/
│   ├── __init__.py
│   ├── ensemble_optimizer.py      # Convex weight optimization
│   ├── diversity_filter.py        # Pairwise correlation filtering
│   └── ensemble_predictor.py      # Production inference
├── monitoring/
│   ├── __init__.py
│   ├── performance_monitor.py     # Real-time performance tracking
│   ├── changepoint_detector.py    # Gaussian Process detection
│   └── retraining_triggers.py     # Automated retraining logic
└── validation/
    └── nested_cpcv.py              # (Optional) Nested validation
```

---

## 🔧 Implementation Tasks

### TASK 1: Reduce Optimization Trials (PRIORITY 1)

**Location:** `src/optimization/hyperparameter_tuning.py`

**Current State:**
- Random search: 120 trials
- Bayesian optimization: 300 trials
- Total: 420 trials

**Required Changes:**
- Random search: 25 trials (reduce from 120)
- Bayesian optimization: 65-95 trials (reduce from 300)
- Total: 90-120 trials

**Implementation Logic:**
1. Locate the trial configuration parameters in hyperparameter tuning code
2. Update trial counts to new values
3. Keep all other optimization logic identical (sampling, evaluation, tracking)
4. Update any config files or scripts that reference the 420-trial count

**Rationale:**
- With 20 fixed features and correlated hyperparameters, diminishing returns occur after ~90-120 trials
- Reduces DSR penalty: E[max SR] = √(2×log(N)) drops from 3.41 to 3.00
- 60% reduction in multiple testing penalty while maintaining optimization quality
- Training time reduced by ~75%

**Validation:**
- Verify optimization completes with new trial counts
- Compare final model performance between 90-trial and 420-trial runs (should be within 5%)
- Confirm training time reduction

---

### TASK 2: Reduce Embargo Period (PRIORITY 1)

**Location:** `src/validation/cross_validation.py`

**Current State:**
- Embargo: 5 days (or 2% of training data)
- Purges samples within 5 days of test fold boundaries

**Required Changes:**
- Embargo: 1 day (can use 2 days for extra safety)
- Update purge logic to use 1-day window

**Implementation Logic:**
1. Find embargo/purge configuration in CPCV implementation
2. Change embargo period from 5 days to 1 day
3. Update purge function to remove train samples within 1 day of any test sample
4. Verify train/test date ranges have proper gaps after splitting

**Rationale:**
- Max hold = 1 day (EOD exits), so label evaluation completes within 1 day
- Conservative embargo = label evaluation time (1 day) + buffer (0-1 day)
- Old 5-day embargo discarded ~25-30% of training samples unnecessarily
- Recovers +20-25% training data without introducing leakage

**Validation:**
- Run test CPCV split and verify train/test date ranges show 1-day gaps
- Check training set size increases by ~20-25% vs old 5-day embargo
- Verify no overlapping timestamps between train/test after purge

---

### TASK 3: Remove Threshold Optimization (PRIORITY 2)

**Location:** `src/optimization/hyperparameter_tuning.py`

**Current State:**
- After hyperparameter optimization, tests 31 thresholds (0.40 to 0.70 in 0.01 steps)
- Selects threshold with best Sharpe ratio
- Saves model with optimized threshold per instrument

**Required Changes:**
- Remove threshold optimization loop entirely
- Use fixed threshold = 0.50 for all models
- Update all strategy config JSONs to set threshold = 0.50

**Implementation Logic:**
1. Locate threshold optimization code (loop over threshold values)
2. Delete the threshold testing loop
3. Set fixed threshold = 0.50 (natural decision boundary for binary classification)
4. Update model saving logic to store threshold = 0.50
5. Update all 12 instrument config files (ES_best.json, NQ_best.json, etc.) to reflect threshold = 0.50

**Rationale:**
- Testing 31 thresholds on same CV folds = 31 additional comparisons
- Introduces 5-15% optimistic bias (picking best threshold on test data)
- Random Forest outputs calibrated probabilities where 0.50 is natural boundary
- With 1-day holds, signal timing matters more than threshold precision
- Better to have unbiased 0.50 than overfit "optimal" threshold

**Validation:**
- Verify models save with threshold = 0.50
- Compare performance with fixed 0.50 vs optimized threshold (should be within 5-10%)
- Confirm no threshold optimization loops remain in code

---

### TASK 4: Implement Ensemble System (PRIORITY 2)

**Location:** New file `src/models/ensemble/ensemble_optimizer.py`

**Current State:**
- Selects single best model from 420 trials based on highest Sharpe ratio
- Saves one model per instrument

**Required Changes:**
- Select top 7-12 models instead of single best
- Implement diversity filtering (pairwise correlation < 0.7)
- Implement convex weight optimization (GEM framework or simpler approaches)
- Create ensemble predictor for production use

**Implementation Logic:**

**Step 1: Model Selection with Diversity**
1. After optimization trials complete, rank all models by validation Sharpe ratio
2. Select top model (highest Sharpe) automatically
3. Iteratively add next-best models if pairwise correlation with existing ensemble < 0.7
4. Continue until 7-12 models selected (stop if can't find more diverse models)

**Step 2: Weight Optimization**
Choose one of three methods:
- **Simple:** Equal weights (1/N for each model) - baseline
- **Sharpe-weighted:** Weight proportional to individual Sharpe ratios
- **GEM (recommended):** Convex optimization to maximize ensemble Sharpe ratio
  - Objective: maximize Sharpe(weighted_predictions)
  - Constraints: weights ≥ 0, sum(weights) = 1

**Step 3: Ensemble Prediction**
1. For each new prediction, get probability outputs from all selected models
2. Calculate weighted average: ensemble_prob = Σ(weight_i × prob_i)
3. Apply threshold (0.50) to get binary prediction

**Step 4: Integration**
1. Replace single model saving with ensemble object saving
2. Update inference code to load ensemble instead of single model
3. Store ensemble metadata (selected models, weights, performance metrics)

**Key Components to Build:**
- `EnsembleOptimizer` class with methods:
  - `select_diverse_models()` - diversity filtering
  - `optimize_weights()` - weight optimization
  - `fit()` - complete ensemble building
  - `predict()` - production inference
- Helper function `build_ensemble_from_trials()` to integrate with existing pipeline

**Rationale:**
- Top 10 models have complementary strengths (different hyperparameters capture different patterns)
- Ensemble reduces variance while maintaining bias
- Expected 12-20% improvement over best single model
- More robust to regime changes

**Validation:**
- Verify ensemble contains 7-12 models with correlation < 0.7
- Verify weights sum to 1.0 and are all non-negative
- Compare ensemble Sharpe vs best single model (should be 12-20% higher)
- Test ensemble prediction latency (should be negligible)

---

### TASK 5: Update DSR Calculation (PRIORITY 3)

**Location:** `src/evaluation/performance_metrics.py`

**Current State:**
- DSR calculation uses N_trials = 420
- Expected max SR = √(2×log(420)) ≈ 3.41
- Required SR for 95% confidence ≈ 1.7-1.9

**Required Changes:**
- Update N_trials parameter from 420 to 90 (or 120 if using 120 trials)
- Expected max SR = √(2×log(90)) ≈ 3.00
- Required SR for 95% confidence ≈ 1.2-1.4

**Implementation Logic:**
1. Locate deflated_sharpe_ratio() function
2. Update default N_trials parameter from 420 to 90
3. Keep all other DSR calculation logic identical:
   - N_effective = N_trials / (1 + (N_trials - 1) × ρ)
   - Expected_max_SR = √(2×log(N_trials))
   - Variance adjustment for non-normality
   - DSR = (observed_SR - expected_max_SR) / std_SR
4. Update all calls to this function throughout codebase

**Additional Enhancement:**
Create validation helper function that:
- Takes observed Sharpe, returns pass/fail + explanation
- Checks if DSR > 0.95 (conservative threshold for statistical significance)
- Prints required Sharpe vs observed Sharpe

**Rationale:**
- Lower N_trials dramatically reduces multiple testing penalty
- Makes it easier to achieve statistical significance
- More realistic expectation under null hypothesis (fewer trials = lower max expected Sharpe by chance)

**Validation:**
- Test DSR calculation with known inputs
- Verify required Sharpe threshold drops by ~15-20%
- Confirm existing strategies now pass DSR test with more margin

---

### TASK 6: Implement Production Monitoring (PRIORITY 3)

**Location:** New file `src/monitoring/performance_monitor.py`

**Current State:**
- No automated production monitoring
- Manual review of performance metrics

**Required Changes:**
Create comprehensive monitoring system with automated alerts and retraining triggers

**Implementation Logic:**

**Core Monitoring Class (`PerformanceMonitor`):**

**Metrics to Track:**
1. **Win Rate (30-day rolling)** - Primary early warning
   - Alert threshold: < 55%
   - Retrain threshold: < 52% over last 100 trades
   
2. **Sharpe Ratio (60-day rolling)** - Confirmation metric
   - Alert threshold: < 1.0
   - Retrain threshold: < 0.5
   
3. **Sharpe Ratio (90-day rolling)** - Shutdown metric
   - Shutdown threshold: < 0.3 for 30+ consecutive days
   
4. **Transaction Costs (daily average)** - Critical monitoring
   - Alert threshold: > 1bp average over last 20 trades

5. **Consecutive Retrain Failures** - Safety check
   - Shutdown threshold: 3 consecutive failures

**Alert Levels:**
- NORMAL: All metrics within bounds
- WARNING: Soft threshold breached (alert but don't act)
- CRITICAL: Hard threshold breached (trigger retraining)
- SHUTDOWN: Multiple critical failures (disable strategy)

**Retraining Triggers:**
1. Sharpe below 0.5 for 45 consecutive days
2. Win rate below 52% over last 100 trades
3. Cooldown period: 63 days minimum between retraining (quarterly baseline)

**Shutdown Triggers:**
1. 90-day Sharpe below 0.3 for 30+ consecutive days
2. 3 consecutive retraining failures
3. Manual override flag

**State Tracking:**
- Last retraining date
- Days since last retrain
- Consecutive failure count
- Days below Sharpe threshold
- Alert history (last 7 days)

**Integration Points:**
1. Call `monitor.update()` daily after market close
2. Load recent trade history (last 90-100 days minimum)
3. Generate performance metrics
4. Check all thresholds
5. Return alert level + recommended actions
6. Store alerts in database/logs
7. Send notifications (email/Slack) for WARNING+ alerts

**Rationale:**
- Win rate degrades first when regime changes (faster signal than Sharpe)
- Multiple confirmation metrics prevent false positives
- Automated triggers prevent human delay/bias
- Cooldown prevents overtraining
- Shutdown protection prevents catastrophic losses

**Validation:**
- Simulate 90 days of trades with declining performance
- Verify alerts trigger at correct thresholds
- Test retraining cooldown logic
- Confirm shutdown triggers work correctly

---

### TASK 7: Implement Nested CPCV (OPTIONAL - Advanced)

**Location:** New file `src/validation/nested_cpcv.py`

**Purpose:** Most rigorous validation for research/publication-grade results

**Current State:**
- Single-level CPCV for hyperparameter optimization
- Same folds used for optimization and final evaluation

**Required Changes:**
- Implement two-level nested cross-validation
- Outer loop: unbiased performance estimation
- Inner loop: hyperparameter optimization

**Implementation Logic:**

**Structure:**
1. Outer CV: Split data into K folds (e.g., 5 folds, k=2 test groups)
2. For each outer fold:
   - Use training portion for hyperparameter optimization
   - Run full 90-trial optimization on inner CV folds
   - Evaluate best model on outer test fold (completely unseen during optimization)
3. Report average performance across outer folds = unbiased estimate

**Key Points:**
- Outer test folds are NEVER seen during hyperparameter search
- Each outer fold gets its own optimized model
- Final performance = average across outer folds
- Computational cost: 5x slowdown (5 outer folds × existing optimization)

**When to Use:**
- Research/validation phase: Get unbiased performance estimates
- Comparing fundamentally different model architectures
- Publication-quality validation needed
- Have compute budget for 5x longer training

**When to Skip:**
- Production optimization: Use single-level CPCV (faster, good enough)
- Retraining quarterly on new data (fresh validation each time)
- Monitoring live performance closely (empirical validation)
- Computational cost is prohibitive

**Rationale:**
- Eliminates all optimistic bias from hyperparameter selection
- Gold standard for ML validation in academic research
- Most conservative performance estimate
- Required if you need to prove results are statistically valid

**Validation:**
- Verify outer test folds are never used in inner optimization
- Compare nested CV results to single-level CPCV (nested should be 5-10% lower)
- Ensure proper embargo applied at both levels

---

## 📊 Configuration File Updates

### Strategy JSON Files (All 12 Instruments)

**Files to Update:**
- `deployment/config/strategies/ES_best.json`
- `deployment/config/strategies/NQ_best.json`
- `deployment/config/strategies/RTY_best.json`
- (Repeat for all 20 instruments)

**Changes Required:**

1. **Model Type:** Change from `"random_forest"` to `"ensemble"`

2. **Model Path:** Update to ensemble pickle file path

3. **Threshold:** Change from optimized value (e.g., 0.625) to `0.50`

4. **Add Ensemble Config:**
   ```
   "n_models_in_ensemble": 9
   "ensemble_method": "gem"
   ```

5. **Update Optimization Config:**
   ```
   "n_trials": 90
   "embargo_days": 1
   "optimize_threshold": false
   "n_random_trials": 25
   "n_bayesian_trials": 65
   ```

6. **Add Monitoring Config:**
   ```
   "win_rate_alert": 0.55
   "win_rate_retrain": 0.52
   "sharpe_60d_alert": 1.0
   "sharpe_60d_retrain": 0.5
   "sharpe_90d_shutdown": 0.3
   ```

---

## 🚀 Deployment Sequence

### Phase 1: Core Optimizations (Week 1)
**Priority: Critical**

1. **Day 1-2:** Implement trial reduction (TASK 1)
   - Update hyperparameter tuning code
   - Test on 1-2 instruments
   - Validate optimization still converges

2. **Day 2-3:** Implement embargo reduction (TASK 2)
   - Update CPCV purge logic
   - Verify proper date gaps
   - Confirm training data increase

3. **Day 3-4:** Remove threshold optimization (TASK 3)
   - Delete threshold loops
   - Update config files
   - Test with fixed 0.50 threshold

4. **Day 4-5:** Update DSR calculation (TASK 5)
   - Change N_trials parameter
   - Test with historical results
   - Verify penalty reduction

5. **Day 5-7:** Run full validation
   - Backtest optimized setup vs old setup
   - Compare out-of-sample Sharpe ratios
   - Verify improvements meet expectations

**Deliverable:** Optimized models with validated performance improvement

---

### Phase 2: Ensemble System (Week 2)
**Priority: High**

1. **Day 1-3:** Implement ensemble optimizer
   - Build model selection with diversity filtering
   - Implement weight optimization (start with simple, upgrade to GEM)
   - Create ensemble predictor class
   - Unit test all components

2. **Day 3-4:** Integrate with training pipeline
   - Replace single model selection with ensemble building
   - Update model saving/loading logic
   - Test end-to-end optimization

3. **Day 4-5:** Backtest ensemble
   - Compare ensemble vs best single model
   - Verify 12-20% improvement
   - Test on all 12 instruments

4. **Day 5-7:** Deploy to production
   - Update inference code
   - Deploy to staging environment first
   - Shadow mode: run alongside old system
   - Gradual rollout to live trading

**Deliverable:** Production ensemble with validated 12-20% lift

---

### Phase 3: Production Monitoring (Week 3)
**Priority: Medium**

1. **Day 1-3:** Implement performance monitor
   - Build PerformanceMonitor class
   - Implement all metric calculations
   - Build alerting logic
   - Test threshold triggers

2. **Day 3-4:** Set up alerting infrastructure
   - Email notifications
   - Slack integration (optional)
   - Database logging
   - Alert dashboard (optional)

3. **Day 4-5:** Create monitoring dashboard
   - Daily metrics visualization
   - Alert history
   - Retraining trigger status
   - Performance trends

4. **Day 5-7:** Test retraining automation
   - Simulate performance degradation
   - Verify retraining triggers
   - Test cooldown logic
   - Validate shutdown triggers

**Deliverable:** Automated monitoring with retraining triggers

---

### Phase 4: Advanced Validation (Week 4 - Optional)
**Priority: Low**

1. **Day 1-3:** Implement nested CPCV
   - Build NestedCPCV class
   - Integrate with hyperparameter optimizer
   - Test on small dataset

2. **Day 4-7:** Run validation studies
   - Full nested CV on all instruments
   - Compare nested vs single-level results
   - Document unbiased performance estimates
   - Write validation report

**Deliverable:** Academic-grade validation results

---

## 🧪 Testing & Validation Strategy

### Unit Tests Required:

1. **Embargo Period Test**
   - Verify 1-day embargo correctly purges samples
   - Check train/test date ranges have proper gaps
   - Confirm no overlapping timestamps

2. **DSR Calculation Test**
   - Verify uses N=90 trials correctly
   - Test with known inputs/outputs
   - Validate required Sharpe thresholds

3. **Ensemble Weights Test**
   - Verify weights sum to 1.0
   - Confirm all weights non-negative
   - Test convex optimization converges

4. **Ensemble Diversity Test**
   - Verify pairwise correlation < 0.7
   - Check minimum 7 models selected
   - Test diversity filtering logic

5. **Monitoring Alerts Test**
   - Verify alert thresholds trigger correctly
   - Test cooldown logic
   - Validate shutdown conditions

### Integration Tests Required:

1. **End-to-End Optimization**
   - Run full pipeline on small dataset
   - Verify feature selection → optimization → ensemble → DSR
   - Confirm all components work together

2. **Production Monitoring**
   - Simulate 90 days of trading
   - Test all alert types
   - Verify retraining triggers
   - Validate shutdown logic

### Backtesting Validation:

**Comparison Framework:**
1. Run old configuration:
   - 420 trials, 5-day embargo, threshold optimization, best single model

2. Run new configuration:
   - 90 trials, 1-day embargo, fixed threshold, ensemble

3. Compare metrics:
   - Out-of-sample Sharpe ratios
   - Win rates
   - DSR scores
   - Training time
   - Model stability

4. Success criteria:
   - New config Sharpe ≥ 120% of old config Sharpe
   - Ensemble beats best single by 12%+
   - DSR > 0.95 (statistically significant)
   - Training time reduced by 70%+

---

## 📈 Expected Performance Impact

### Quantitative Improvements:

| Metric | Old | New | Improvement |
|--------|-----|-----|-------------|
| Trials | 420 | 90-120 | -75% training time |
| Embargo | 5 days | 1-2 days | +23% training data |
| Threshold | Optimized | Fixed 0.50 | -10% bias |
| Model selection | Best single | Ensemble 7-12 | +15% performance |
| DSR penalty | High | 60% lower | Easier to pass |
| Required Sharpe | ~1.7 | ~1.2 | -30% barrier |

### Risk-Adjusted Returns:
- **Conservative estimate:** +20-30% improvement
- **Expected:** +30-40% improvement  
- **Optimistic:** +40-50% improvement

Example: If current validated Sharpe = 1.0, expect optimized Sharpe = 1.3-1.4

---

## 📊 Success Metrics (After 60 Days)

### Performance Metrics:
- [ ] Observed Sharpe ratio increased by 20%+ over baseline
- [ ] DSR score > 0.95 (statistically significant at 95% confidence)
- [ ] Ensemble beats best single model by 12%+ consistently
- [ ] Win rate maintained or improved
- [ ] Transaction costs within 1bp threshold

### Operational Metrics:
- [ ] Training time reduced by 70%+
- [ ] No false retraining triggers (< 1 per quarter)
- [ ] Zero unexpected shutdowns
- [ ] Alerts accurate and actionable
- [ ] Monitoring system functioning reliably

### Quality Metrics:
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Backtests validate improvements
- [ ] Documentation complete
- [ ] Team trained on new system

---

## 🎓 Key Concepts Summary

### 1. Trial Reduction Logic
- **Principle:** Diminishing returns after ~10× number of features
- **Application:** 20 features × 5-6 = 90-120 optimal trials
- **Benefit:** Lower multiple testing penalty without losing optimization quality

### 2. Embargo Reduction Logic
- **Principle:** Embargo ≥ max(label_evaluation_time, autocorrelation_decay)
- **Application:** 1-day trades → 1-2 day embargo sufficient
- **Benefit:** Recover 20-25% training data without leakage

### 3. Fixed Threshold Logic
- **Principle:** Avoid optimizing on test data to prevent overfitting
- **Application:** Use natural 0.50 decision boundary instead of optimizing
- **Benefit:** Eliminate 5-15% optimistic bias from threshold selection

### 4. Ensemble Logic
- **Principle:** Diversity + proper weighting beats single best model
- **Application:** Combine top 7-12 diverse models with optimized weights
- **Benefit:** 12-20% improvement through variance reduction

### 5. Monitoring Logic
- **Principle:** Early detection prevents larger losses
- **Application:** Win rate degrades before Sharpe, monitor multiple metrics
- **Benefit:** Faster adaptation to regime changes

---

## ⚠️ Critical Implementation Notes

### What NOT to Change:
1. Feature selection methodology (already optimal)
2. CPCV structure (combinatorial purged is correct)
3. Random Forest architecture (meta-labeling approach is sound)
4. IBS strategy logic (primary strategy stays same)
5. Model evaluation metrics (Sharpe ratio is appropriate)

### What MUST Be Done Carefully:
1. Embargo period - verify no leakage with 1-day setting
2. Ensemble diversity - don't include highly correlated models
3. DSR calculation - use correct N_trials parameter everywhere
4. Monitoring thresholds - set conservatively initially, tune over time
5. Retraining cooldown - don't retrain too frequently

### Common Pitfalls to Avoid:
1. Reducing embargo below 1 day (causes leakage)
2. Skipping DSR validation (leads to overfit models in production)
3. Including >12 models in ensemble (diminishing returns, higher variance)
4. Setting monitoring thresholds too tight (false alerts)
5. Ignoring transaction cost spikes (can destroy profitability)
6. Retraining without cooldown period (overreaction to noise)
7. Using nested CV in production (unnecessary computational cost)

---

## 📞 Key Questions for Implementation

### Before Starting:
1. Where is the hyperparameter tuning code located exactly?
2. Do we have unit tests for embargo/purge logic?
3. What's current training time for 420 trials?
4. How are models deployed to production currently?
5. Do we have infrastructure for email/Slack alerts?

### During Implementation:
1. Are optimization trials converging with 90 trials?
2. Did training data increase by ~20% with 1-day embargo?
3. Is ensemble consistently beating best single model?
4. Are monitoring alerts triggering appropriately?
5. Is DSR calculation showing improvement?

### After Deployment:
1. What's the actual Sharpe improvement vs baseline?
2. Are we passing DSR significance test?
3. Have there been any false retraining triggers?
4. Is ensemble stable over time?
5. Are transaction costs within expectations?

---

## 🔄 Rollout Strategy

### Staged Deployment:

**Stage 1: Development (Week 1-2)**
- Implement all changes in dev environment
- Test on historical data
- Validate improvements

**Stage 2: Staging (Week 3)**
- Deploy to staging environment
- Test with 2-3 instruments only (ES, NQ)
- Run for 1-2 weeks
- Verify monitoring works correctly

**Stage 3: Shadow Mode (Week 4)**
- Run new system alongside old system
- Generate signals but don't trade on them
- Compare predictions and performance
- Validate no regressions

**Stage 4: Gradual Rollout (Week 5-6)**
- Deploy to 25% of instruments (3-5 instruments)
- Monitor closely for 1 week
- Deploy to 50% of instruments
- Monitor for 1 week
- Deploy to 100% of instruments

**Stage 5: Full Production (Week 7+)**
- All instruments on new system
- Old system decommissioned
- Monitoring active 24/7
- Regular performance reviews

---

**END OF IMPLEMENTATION DIRECTIONS**

**Total Implementation Time:** 3-4 weeks  
**Expected Uplift:** 30-40% improvement in risk-adjusted returns  
**Risk Level:** Low (all changes are conservative improvements to existing system)
