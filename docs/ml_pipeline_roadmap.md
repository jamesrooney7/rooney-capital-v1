# ML Pipeline Roadmap: Research → Production Parity

**Goal:** Ensure what you backtest is exactly what trades live.

**Last Updated:** 2025-10-25

---

## System Configuration

### Data Setup
- **Historical Data:** `/home/user/Desktop/Rooney Capital/Futures Data/`
- **File Format:** CSV files (ES_bt.csv, 6A_bt.csv, etc.)
- **Time Period:** 2010-2024
- **Overlap Period:** December 2024 (Databento lookback = 1 year)

### ML Training
- **Method:** Random Forest + Bayesian Optimization
- **Workflow:**
  1. Per-instrument optimization (ES, NQ, YM, RTY, etc.)
  2. Portfolio-level optimization (position sizing, correlation, risk limits)

### Current Production Code
- **Strategy:** `src/strategy/ibs_strategy.py`
- **Features:** IBS, RSI-2, volume percentile, pair signals
- **Framework:** Backtrader

---

## Implementation Phases

### ~~Phase 0: Data Alignment & Validation~~ ✂️ SKIPPED
**Decision:** Not needed - code parity matters more than exact data parity

---

### Phase 1.5: Quick Feature Validation ⚡ (30 minutes)
**Status:** 🔲 Not Started

**Goal:** Sanity check that features calculated on historical data match expectations

**Tasks:**
- [ ] Load 1 month of historical data (Dec 2024)
- [ ] Calculate IBS, RSI-2 on historical data
- [ ] Visual inspection - do values look reasonable?
- [ ] If Databento overlap available, spot-check a few values
- [ ] ✅ Proceed if ±5% match OR ❌ Investigate if wildly different

**Deliverables:**
- Quick validation script (can be throwaway code)
- Confirmation that features are calculating correctly

---

### Phase 2: Unified Backtest Framework 📈 (Week 1)
**Status:** 🔲 Not Started

**Goal:** Run production `IbsStrategy` code on historical data

**Tasks:**
- [ ] Create `research/backtest_runner.py`
  - [ ] Load historical CSV data
  - [ ] Import production `IbsStrategy` (same class as live)
  - [ ] Configure Backtrader with same commission/slippage as live
  - [ ] Run backtest on 2023-2024 (recent data)
- [ ] Create `research/utils/data_loader.py`
  - [ ] Read ES_bt.csv, NQ_bt.csv, etc.
  - [ ] Convert to Backtrader data format
  - [ ] Handle timezone/timestamp parsing
- [ ] Run ES backtest (2023-2024)
- [ ] Compare results to notebook backtests
  - [ ] Are returns similar?
  - [ ] Are trade counts similar?
  - [ ] Debug any major discrepancies
- [ ] Extend to all symbols (NQ, YM, RTY, etc.)

**Key Principle:**
- ❌ Don't reimplement strategy logic
- ✅ Import and use `IbsStrategy` directly
- ✅ Same features, same logic, same position sizing

**Deliverables:**
- `research/backtest_runner.py` - Runs production code on historical data
- `research/utils/data_loader.py` - Loads CSV files into Backtrader
- Backtest results for all symbols (2023-2024)
- Validation that production code matches notebook results

---

### Phase 3: Model Training Pipeline 🤖 (Week 2)
**Status:** 🔲 Not Started

**Goal:** Integrate existing ML script into automated, repeatable pipeline

**Tasks:**
- [ ] Centralize feature calculations
  - [ ] Extract features from ML script into `src/features/feature_registry.py`
  - [ ] Ensure same feature code used in training AND live
  - [ ] Add unit tests for feature calculations
- [ ] Create training configuration system
  - [ ] `configs/training/ES.yml` - per-instrument config
  - [ ] `configs/training/portfolio.yml` - portfolio-level config
  - [ ] Define features, hyperparameter search space, validation approach
- [ ] Adapt existing ML script into `scripts/train_model.py`
  - [ ] Per-instrument optimization (Random Forest + Bayesian)
  - [ ] Walk-forward validation
  - [ ] Save model bundles with metadata
- [ ] Create model versioning system
  - [ ] `src/models/ES_v1.joblib`, `ES_v2.joblib`, etc.
  - [ ] `src/models/manifest.json` - track which version is "production"
  - [ ] Model metadata (training date, performance, features used)
- [ ] Training report generator
  - [ ] Out-of-sample performance
  - [ ] Feature importance
  - [ ] Hyperparameter optimization results

**Deliverables:**
- `src/features/feature_registry.py` - Centralized features
- `configs/training/*.yml` - Training configurations
- `scripts/train_model.py` - Automated training pipeline
- Model versioning system
- Training report template

---

### Phase 4: Portfolio-Level Optimization 💼 (Week 3)
**Status:** 🔲 Not Started

**Goal:** Optimize portfolio-level parameters using trained instrument models

**Tasks:**
- [ ] Create `scripts/optimize_portfolio.py`
  - [ ] Load all trained instrument models (ES_v1, NQ_v1, etc.)
  - [ ] Run portfolio backtest with production strategy
  - [ ] Optimize:
    - [ ] Position sizing per instrument
    - [ ] Correlation filters (don't trade both ES/NQ if correlated)
    - [ ] Portfolio-level risk limits (max drawdown, volatility target)
    - [ ] Cash allocation across instruments
- [ ] Portfolio validation
  - [ ] Out-of-sample portfolio metrics
  - [ ] Sharpe, Sortino, Max DD at portfolio level
  - [ ] Correlation matrix
  - [ ] Risk contribution per instrument
- [ ] Save portfolio configuration
  - [ ] `configs/portfolio_config.yml` - optimized parameters
  - [ ] Version control portfolio configs

**Deliverables:**
- `scripts/optimize_portfolio.py` - Portfolio optimization
- `configs/portfolio_config.yml` - Optimized portfolio parameters
- Portfolio backtest report

---

### Phase 5: Pre-Production Validation ✅ (Week 4)
**Status:** 🔲 Not Started

**Goal:** Prove new models work before going live

**Tasks:**
- [ ] Create `scripts/validate_model.py`
  - [ ] Feature availability check
  - [ ] Backtest performance check (min thresholds)
  - [ ] Out-of-sample validation
  - [ ] Feature importance sanity check
  - [ ] Prediction distribution check (not all 0s/1s)
  - [ ] Simulate live feature calculation
- [ ] Paper trading validation
  - [ ] Deploy with `POLICY_KILLSWITCH=true`
  - [ ] Run 1-2 weeks in paper mode
  - [ ] Compare paper results to backtest
  - [ ] Monitor for unexpected behavior
- [ ] Create validation checklist
  - [ ] All checks must pass before production deployment

**Deliverables:**
- `scripts/validate_model.py` - Automated validation
- Paper trading monitoring dashboard
- Validation report template
- Deployment checklist

---

### Phase 6: Production Deployment 🚀 (Week 5)
**Status:** 🔲 Not Started

**Goal:** Safe, auditable model deployment

**Tasks:**
- [ ] Create deployment workflow
  - [ ] `scripts/promote_model.py` - Update manifest to new version
  - [ ] `deploy/update_model.sh` - Deploy new models
  - [ ] `scripts/rollback_model.py` - Rollback if needed
- [ ] Update model manifest
  - [ ] `src/models/manifest.json` - Which version is live
  - [ ] Track deployment date, deployed by, previous version
- [ ] Deployment validation
  - [ ] Run automated validation
  - [ ] Paper trade for 1-2 weeks
  - [ ] Production deployment
  - [ ] Monitor for 1 week
- [ ] Rollback procedures
  - [ ] Document rollback process
  - [ ] Test rollback capability

**Deliverables:**
- Deployment scripts
- Model manifest system
- Rollback procedures
- Deployment checklist

---

### Phase 7: Production Monitoring 📡 (Ongoing)
**Status:** 🔲 Not Started

**Goal:** Detect when models degrade

**Tasks:**
- [ ] ML performance tracking
  - [ ] Log every prediction + outcome
  - [ ] Track prediction accuracy over time
  - [ ] Model calibration (do probabilities match outcomes?)
  - [ ] Feature drift detection
- [ ] Automated alerts
  - [ ] Accuracy drops below threshold → Discord alert
  - [ ] Feature drift detected → Warning
  - [ ] Prediction distribution changes → Investigation
- [ ] Retraining triggers
  - [ ] Calendar-based (every 6 months)
  - [ ] Performance-based (accuracy drops 10%)
  - [ ] Market regime change detection
- [ ] Create monitoring dashboard
  - [ ] Live accuracy vs backtest
  - [ ] Feature distributions over time
  - [ ] Model degradation signals

**Deliverables:**
- ML performance tracking in database
- Monitoring dashboard
- Automated degradation alerts
- Retraining schedule

---

## Key Principles for Parity

### 1. One Codebase
- ✅ Research imports from `src/`
- ❌ Don't copy-paste strategy logic into notebooks

### 2. Same Features Everywhere
- ✅ `FeatureRegistry.calculate_ibs()` used in training AND live
- ❌ Different feature calculations in notebook vs production

### 3. Validate Everything
- ✅ Backtest with production code before deploying
- ✅ Paper trade for 1-2 weeks
- ✅ Compare live performance to backtest

### 4. Version Everything
- Models, configs, feature definitions, data sources
- Can recreate any historical model

### 5. Monitor Continuously
- ML accuracy tracking
- Feature drift detection
- Performance degradation alerts

---

## Progress Tracking

### Sprint 1 (Current): Foundation
- [ ] Phase 1.5: Quick feature validation
- [ ] Phase 2: Backtest runner with production code

### Sprint 2: Training Pipeline
- [ ] Phase 3: Model training pipeline
- [ ] Phase 4: Portfolio-level optimization

### Sprint 3: Deployment
- [ ] Phase 5: Pre-production validation
- [ ] Phase 6: Production deployment

### Sprint 4: Monitoring
- [ ] Phase 7: Continuous monitoring

---

## Notes & Decisions

### 2025-10-25
- Decided to skip Phase 0 (data validation) - code parity more important
- Historical data: 2010-2024, CSV format
- Overlap period: Dec 2024 (Databento 1-year lookback)
- ML method: Random Forest + Bayesian optimization
- Workflow: Per-instrument → Portfolio-level optimization

---

## Questions / Blockers

(None currently)

---

## Resources

- [Backtrader Documentation](https://www.backtrader.com/docu/)
- Production strategy: `src/strategy/ibs_strategy.py`
- Current ML script: (To be shared in Phase 3)
