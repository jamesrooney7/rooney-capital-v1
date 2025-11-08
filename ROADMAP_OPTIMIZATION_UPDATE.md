# Roadmap Update: Optimization Integration

## Summary

This document contains updates to `MULTI_ALPHA_REFACTORING_ROADMAP.md` to incorporate ML optimization results and portfolio greedy search integration.

**Critical Addition**: The current roadmap does NOT include integration of optimization results. This update adds tasks to properly integrate:
1. Per-instrument ML bundles (`{SYMBOL}_best.json` + `{SYMBOL}_rf_model.pkl`)
2. Portfolio greedy search results (optimal symbol list, max_positions, daily_stop_loss)
3. Per-instrument parameter loading system

---

## Updated Week 2 - Phase 2.1

### **Phase 2.1: IBS Strategy Refactoring + Optimization Integration (Days 8-11)**

**Tasks:**

#### 1. Create Portfolio Optimization Configuration

**NEW TASK** - Create `config/portfolio_optimization_ibs.json`:

```json
{
  "strategy": "ibs",
  "portfolio_constraints": {
    "max_positions": 4,
    "daily_stop_loss": 2500.0,
    "symbols": ["6A", "6B", "6C", "6N", "6S", "CL", "ES", "PL", "RTY", "SI"]
  },
  "expected_performance": {
    "sharpe_ratio": 14.574,
    "cagr": 0.844,
    "max_drawdown_dollars": 5878.99,
    "breach_events": 0,
    "daily_stops_hit": 12
  },
  "optimization_date": "2025-10-31",
  "optimization_method": "greedy_search",
  "notes": "10-symbol portfolio optimized via greedy search. Expected Sharpe 14.57."
}
```

**Acceptance Criteria**:
- [ ] File created with greedy search results
- [ ] Validated against actual optimization output

#### 2. Update Multi-Alpha Configuration

**UPDATED TASK** - Update `config.multi_alpha.yml`:

```yaml
strategies:
  ibs:
    enabled: true
    broker_account: ${TRADERSPOST_IBS_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/  # ← FIXED: Was src/models/ibs/
    max_positions: 4           # ← FIXED: Was 2, now from greedy search
    daily_stop_loss: 2500      # ← From greedy search

    # OPTIMIZED: 10 symbols from portfolio greedy search
    instruments:
      - 6A
      - 6B
      - 6C
      - 6N
      - 6S
      - CL
      - ES
      - PL
      - RTY
      - SI

    strategy_params:
      # Session windows
      use_window1: true
      start_time1: "0000"
      end_time1: "1500"
      use_window2: true
      start_time2: "1700"
      end_time2: "2400"

      # Generic IBS thresholds (same for all instruments)
      enable_ibs_entry: true
      ibs_entry_high: 0.3
      ibs_entry_low: 0.0

      enable_ibs_exit: true
      ibs_exit_high: 0.7
      ibs_exit_low: 0.3

      # Generic risk management
      enable_stop: true
      stop_type: "percent"
      stop_perc: 2.0

      enable_tp: true
      tp_type: "percent"
      tp_perc: 3.0

      # NOTE: ml_threshold loaded per-instrument from {SYMBOL}_best.json
```

**Acceptance Criteria**:
- [ ] Instrument list matches portfolio optimization (10 symbols)
- [ ] max_positions matches greedy search (4)
- [ ] models_path points to correct directory (src/models/)

#### 3. Integrate ML Bundle Loading into Strategy Worker

**NEW TASK** - Update `src/runner/strategy_worker.py`:

Add ML bundle loading:

```python
def _setup_strategy(self):
    """Setup strategy with optimized ML parameters per instrument."""

    from src.models.loader import load_model_bundle

    # Load ML bundle for this symbol
    try:
        ml_bundle = load_model_bundle(
            self.primary_symbol,
            base_dir=self.strategy_config.models_path
        )
        logger.info(
            f"Loaded ML bundle for {self.primary_symbol}: "
            f"threshold={ml_bundle.threshold}, features={len(ml_bundle.features)}"
        )
    except FileNotFoundError:
        logger.warning(
            f"No ML model found for {self.primary_symbol}, running without ML filter"
        )
        ml_bundle = None

    # Create strategy parameters
    strategy_params = {
        'symbol': self.primary_symbol,
        'portfolio_coordinator': self.portfolio_coordinator,
        'size': self.instrument_config.size,
        **self.strategy_config.strategy_params,  # Generic params from YAML
    }

    # Add ML bundle if loaded
    if ml_bundle:
        strategy_params.update({
            'ml_model': ml_bundle.model,
            'ml_features': ml_bundle.features,
            'ml_threshold': ml_bundle.threshold,  # Per-instrument optimized!
        })

    # Load strategy class and add to Cerebro
    strategy_class = load_strategy(self.strategy_name)
    self.cerebro.addstrategy(strategy_class, **strategy_params)

    logger.info(
        f"Added strategy '{self.strategy_name}' for {self.primary_symbol} "
        f"with ML threshold {ml_bundle.threshold if ml_bundle else 'N/A'}"
    )
```

**Acceptance Criteria**:
- [ ] Strategy worker loads `{SYMBOL}_best.json` for each instrument
- [ ] ML threshold is per-instrument (not generic 0.65)
- [ ] Features are per-instrument
- [ ] Graceful fallback if ML model missing

#### 4. Verify ML Bundle Loading

**NEW TASK** - Create `tests/test_optimization_integration.py`:

```python
def test_ml_bundle_loading():
    """Test that ML bundles load correctly for all instruments."""

    from src.models.loader import load_model_bundle

    symbols = ["6A", "6B", "6C", "6N", "6S", "CL", "ES", "PL", "RTY", "SI"]

    for symbol in symbols:
        bundle = load_model_bundle(symbol, base_dir="src/models/")

        # Verify bundle structure
        assert bundle.symbol == symbol
        assert bundle.model is not None
        assert len(bundle.features) > 0
        assert bundle.threshold is not None
        assert 0 < bundle.threshold < 1

        # Verify metadata
        assert 'Sharpe' in bundle.metadata
        assert 'Profit_Factor' in bundle.metadata
        assert 'Features' in bundle.metadata

        print(f"✓ {symbol}: threshold={bundle.threshold:.3f}, features={len(bundle.features)}")

def test_portfolio_optimization_loading():
    """Test that portfolio optimization file loads correctly."""

    import json
    from pathlib import Path

    path = Path("config/portfolio_optimization_ibs.json")
    assert path.exists(), "Portfolio optimization file missing"

    with open(path) as f:
        config = json.load(f)

    # Verify structure
    assert config['strategy'] == 'ibs'
    assert config['portfolio_constraints']['max_positions'] == 4
    assert config['portfolio_constraints']['daily_stop_loss'] == 2500.0
    assert len(config['portfolio_constraints']['symbols']) == 10

    # Verify symbols match expected
    expected_symbols = {"6A", "6B", "6C", "6N", "6S", "CL", "ES", "PL", "RTY", "SI"}
    actual_symbols = set(config['portfolio_constraints']['symbols'])
    assert actual_symbols == expected_symbols

    print("✓ Portfolio optimization config valid")
```

**Acceptance Criteria**:
- [ ] All 10 instruments have ML bundles
- [ ] Each instrument has different threshold
- [ ] Portfolio optimization file validates correctly

#### 5. Document Optimization Integration

**NEW TASK** - Update `MULTI_ALPHA_SETUP.md`:

Add section:

```markdown
## Optimization Integration

The multi-alpha system automatically loads optimized parameters for each strategy:

### 1. ML Optimization Results (Per-Instrument)

Each instrument has optimized ML model parameters stored in:
- `src/models/{SYMBOL}_best.json` - Hyperparameters, features, threshold
- `src/models/{SYMBOL}_rf_model.pkl` - Trained RandomForest model

**Example (ES):**
```json
{
  "Symbol": "ES",
  "Prod_Threshold": 0.5,
  "Features": [...],  // 30 optimized features
  "Params": {...}     // RandomForest hyperparameters
}
```

The strategy worker automatically loads these on startup:
```python
ml_bundle = load_model_bundle("ES")  # Loads ES_best.json + ES_rf_model.pkl
# Uses ES-specific threshold (0.5), not generic (0.65)
```

### 2. Portfolio Greedy Search Results

Portfolio-level optimization determines:
- **Which symbols to trade**: Optimal 10-symbol portfolio
- **Max concurrent positions**: 4 (not 2)
- **Daily stop loss**: $2,500

Stored in: `config/portfolio_optimization_ibs.json`

The strategy worker uses these constraints automatically via config.multi_alpha.yml.

### 3. Per-Instrument Parameters (Optional)

If you have instrument-specific filter settings from optimization:
- Create `config/ibs_instrument_params.json`
- Strategy worker will load and override generic params

**Example:**
```json
{
  "ES": {
    "ibs_entry_high": 0.3,
    "stop_perc": 2.0
  },
  "NQ": {
    "ibs_entry_high": 0.35,
    "stop_perc": 2.5
  }
}
```
```

**Acceptance Criteria**:
- [ ] Documentation explains optimization integration
- [ ] Examples show how parameters are loaded
- [ ] Instructions for adding new strategies with optimization

---

## Updated Week 2 - Phase 2.2

### **Phase 2.2: Integration Testing + Optimization Verification (Days 12-13)**

Add to existing tasks:

#### 6. Verify Optimization Parameters in Integration Tests

**ADDED TASK** - Extend integration tests:

```python
def test_ml_threshold_per_instrument():
    """Verify each instrument uses its optimized ML threshold."""

    from src.runner.strategy_worker import StrategyWorker
    from src.config.config_loader import load_config

    config = load_config("config.test.yml")

    for symbol in config.strategies['ibs'].instruments:
        worker = StrategyWorker(strategy_name='ibs', config=config, symbol=symbol)

        # Get the ML threshold being used
        strategy_params = worker._get_strategy_params()
        ml_threshold = strategy_params['ml_threshold']

        # Load expected threshold from JSON
        from src.models.loader import load_model_bundle
        bundle = load_model_bundle(symbol)

        # Verify they match
        assert ml_threshold == bundle.threshold, \
            f"{symbol}: threshold mismatch! Got {ml_threshold}, expected {bundle.threshold}"

        print(f"✓ {symbol}: using optimized threshold {ml_threshold:.3f}")

def test_portfolio_constraints():
    """Verify portfolio coordinator uses greedy search constraints."""

    from src.runner.portfolio_coordinator import PortfolioCoordinator
    from src.config.config_loader import load_config

    config = load_config("config.multi_alpha.yml")
    ibs_config = config.strategies['ibs']

    coordinator = PortfolioCoordinator(
        max_positions=ibs_config.max_positions,
        daily_stop_loss=ibs_config.daily_stop_loss
    )

    # Verify constraints match greedy search
    assert coordinator.max_positions == 4, "Should use greedy-optimized max_positions"
    assert coordinator.daily_stop_loss == 2500, "Should use greedy-optimized stop loss"

    print("✓ Portfolio coordinator uses optimized constraints")
```

**Acceptance Criteria**:
- [ ] Each instrument confirmed using its optimized ML threshold
- [ ] Portfolio coordinator confirmed using greedy search constraints (4 positions, $2500 stop)
- [ ] Integration tests pass with optimization parameters

---

## Updated Week 4: Future Strategy Addition Process

### **Phase 4.3: Second Strategy Optimization Process**

When adding new strategies (e.g., "breakout"), follow this optimization workflow:

#### 1. ML Optimization (Per-Instrument)

**Tasks**:
- [ ] Run ML optimization for each instrument using your existing pipeline
- [ ] Generate `{SYMBOL}_best.json` + `{SYMBOL}_rf_model.pkl` for each instrument
- [ ] Store in `src/models/breakout/`

**Deliverables**:
```
src/models/breakout/
├── ES_best.json
├── ES_rf_model.pkl
├── NQ_best.json
├── NQ_rf_model.pkl
└── ... (all tested instruments)
```

#### 2. Portfolio Greedy Search

**Tasks**:
- [ ] Run `research/portfolio_simulator.py` with all optimized instruments
- [ ] Find optimal: symbol list, max_positions, daily_stop_loss
- [ ] Create `config/portfolio_optimization_breakout.json`

**Example Command**:
```bash
python research/portfolio_simulator.py \
    --results-dir results/breakout_optimization/ \
    --min-positions 1 \
    --max-positions 10 \
    --ranking-method sharpe \
    --output results/breakout_portfolio_optimization.csv
```

**Deliverables**:
```json
{
  "strategy": "breakout",
  "portfolio_constraints": {
    "max_positions": 3,
    "daily_stop_loss": 3000.0,
    "symbols": ["ES", "NQ", "CL", "GC", "SI"]
  },
  "expected_performance": {
    "sharpe_ratio": 12.3,
    "cagr": 0.65
  }
}
```

#### 3. Configuration Update

**Tasks**:
- [ ] Add strategy to `config.multi_alpha.yml`
- [ ] Use optimized symbol list from greedy search
- [ ] Use optimized max_positions and daily_stop_loss
- [ ] Set models_path to `src/models/breakout/`

**Example**:
```yaml
strategies:
  breakout:
    enabled: true
    broker_account: ${TRADERSPOST_BREAKOUT_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/breakout/  # ← Strategy-specific
    max_positions: 3                   # ← From greedy search
    daily_stop_loss: 3000              # ← From greedy search

    # Optimized symbols from greedy search
    instruments:
      - ES
      - NQ
      - CL
      - GC
      - SI

    strategy_params:
      # Breakout-specific parameters
      breakout_lookback: 20
      atr_multiplier: 2.0
```

#### 4. Verification

**Tasks**:
- [ ] Verify ML bundles load for all instruments
- [ ] Verify each instrument uses its optimized threshold
- [ ] Verify portfolio constraints match greedy search
- [ ] Run integration tests

**Acceptance Criteria**:
- ✅ All ML bundles present and loading correctly
- ✅ Per-instrument thresholds applied
- ✅ Portfolio constraints from greedy search
- ✅ Expected performance metrics documented

---

## Summary of Changes

### New Files Created:
1. `config/portfolio_optimization_ibs.json` - Greedy search results
2. `tests/test_optimization_integration.py` - Optimization verification tests
3. `OPTIMIZATION_INTEGRATION_ANALYSIS.md` - Gap analysis document (already created)

### Files Modified:
1. `config.multi_alpha.yml` - Fixed paths, instruments, max_positions
2. `src/runner/strategy_worker.py` - Added ML bundle loading
3. `MULTI_ALPHA_SETUP.md` - Added optimization integration docs
4. `tests/integration/test_multi_alpha_e2e.py` - Added optimization verification

### Key Changes:
- **Week 2 Phase 2.1**: +5 tasks for optimization integration
- **Week 2 Phase 2.2**: +2 tests for optimization verification
- **Week 4 Phase 4.3**: New section for future strategy optimization workflow

### Priority:
- **HIGH**: This is critical for achieving expected performance (Sharpe 14.57)
- **Must complete in Week 2**: Without optimization integration, system will underperform

---

## Action Items

### Immediate (Week 2):
1. [ ] Create `config/portfolio_optimization_ibs.json`
2. [ ] Update `config.multi_alpha.yml` with correct settings
3. [ ] Update `strategy_worker.py` to load ML bundles
4. [ ] Create `tests/test_optimization_integration.py`
5. [ ] Verify all 10 instruments have ML bundles
6. [ ] Run integration tests with optimization parameters

### Questions for User:
1. Do you have per-instrument filter settings (e.g., different ibs_entry_high per symbol)?
2. Confirm ML models are in `src/models/` not `src/models/ibs/`?
3. Should I create the portfolio optimization JSON from PORTFOLIO_INTEGRATION_GUIDE.md?

---

**Impact**: Without this integration, the multi-alpha system would use wrong parameters and significantly underperform the optimized results. This update ensures we achieve the expected Sharpe 14.57 and CAGR 84.4%.
