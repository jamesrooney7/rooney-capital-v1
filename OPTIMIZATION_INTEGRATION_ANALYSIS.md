# Optimization Integration Analysis

## Executive Summary

**Critical Gap Identified**: ⚠️ The multi-alpha configuration system does NOT yet integrate your ML optimization results and portfolio greedy search settings.

**Current State**:
- ✅ ML optimization JSON files exist (`src/models/{SYMBOL}_best.json`)
- ✅ Model loader exists (`src/models/loader.py`)
- ✅ Portfolio greedy search results documented (PORTFOLIO_INTEGRATION_GUIDE.md)
- ❌ Multi-alpha config does NOT load per-instrument optimized parameters
- ❌ Multi-alpha config does NOT use portfolio optimization results

**Action Required**: Integrate optimization results into the multi-alpha configuration and strategy worker system.

---

## Current Optimization Assets

### 1. ML Optimization Results (Per-Instrument)

**Location**: `/home/user/rooney-capital-v1/src/models/`

**Files Found**:
```
6A_best.json + 6A_rf_model.pkl
6B_best.json + 6B_rf_model.pkl
6E_best.json + 6E_rf_model.pkl
CL_best.json + CL_rf_model.pkl
ES_best.json + ES_rf_model.pkl
GC_best.json + GC_rf_model.pkl
HG_best.json + HG_rf_model.pkl
NG_best.json + NG_rf_model.pkl
NQ_best.json + NQ_rf_model.pkl
RTY_best.json + RTY_rf_model.pkl
SI_best.json + SI_rf_model.pkl
YM_best.json + YM_rf_model.pkl
```

**JSON Structure** (Example: ES_best.json):
```json
{
  "Symbol": "ES",
  "Sharpe": 0.9413,
  "Profit_Factor": 1.6469,
  "Trades": 875,
  "Prod_Threshold": 0.5,  ← ML probability threshold
  "Params": {              ← RandomForest hyperparameters
    "n_estimators": 900,
    "min_samples_leaf": 50,
    "max_depth": 5,
    "max_features": "log2",
    "bootstrap": true,
    "max_samples": 0.6407,
    "class_weight": "balanced_subsample"
  },
  "Features": [            ← Feature list (30 features for ES)
    "volume_z_percentile",
    "volz_pct",
    "rsixvolz",
    "ibsxvolz",
    "tlt_daily_z_score",
    ...
  ]
}
```

**What This Provides**:
1. **ML Model**: Trained RandomForestClassifier (`{SYMBOL}_rf_model.pkl`)
2. **Features**: Optimized feature set (varies by instrument)
3. **Threshold**: Production probability threshold (e.g., 0.5 for ES)
4. **Performance Metrics**: Sharpe, Profit Factor, Trade Count

### 2. Portfolio Greedy Search Results

**Source**: `PORTFOLIO_INTEGRATION_GUIDE.md`

**Optimal Portfolio**:
```json
{
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
  }
}
```

**What This Provides**:
1. **Symbol Selection**: Only trade these 10 symbols (from 18 total)
2. **Position Limits**: Max 4 concurrent positions
3. **Risk Management**: $2,500 daily stop loss
4. **Expected Performance**: Sharpe 14.57, CAGR 84.4%

### 3. Existing Model Loader

**File**: `src/models/loader.py`

**Key Functions**:
```python
load_model_bundle(symbol: str) -> ModelBundle
    # Loads {SYMBOL}_best.json + {SYMBOL}_rf_model.pkl
    # Returns: model, features, threshold, metadata

strategy_kwargs_from_bundle(bundle: ModelBundle) -> dict
    # Returns: {'ml_model': ..., 'ml_features': ..., 'ml_threshold': ...}
```

**Current Usage** (in existing system):
```python
# Old live_worker.py
bundle = load_model_bundle("ES")
strategy_kwargs = {
    'symbol': 'ES',
    **bundle.strategy_kwargs(),  # ml_model, ml_features, ml_threshold
    'size': 1,
    # ... other params
}
```

---

## Gap Analysis: What's Missing in Multi-Alpha Config

### Current Config Structure
**File**: `config.multi_alpha.example.yml`

```yaml
strategies:
  ibs:
    enabled: true
    broker_account: ${TRADERSPOST_IBS_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/ibs/  # ← Wrong path!
    max_positions: 2               # ← Should be 4!
    daily_stop_loss: 2500          # ✓ Correct

    instruments:  # ← Should be optimized 10 symbols!
      - ES
      - NQ
      - RTY
      ... (all 18 symbols)

    strategy_params:
      ibs_entry_high: 0.7   # ← Generic, not instrument-specific!
      ibs_exit_high: 0.3    # ← Generic, not instrument-specific!
      ml_threshold: 0.65    # ← Generic! Should load from {SYMBOL}_best.json
```

### Gaps Identified

#### Gap 1: No Per-Instrument ML Bundle Loading ❌
**Problem**:
- Config has `ml_threshold: 0.65` (generic)
- Should load from `ES_best.json` (Prod_Threshold: 0.5)
- Each instrument has different optimized threshold!

**Example**:
```
ES: Prod_Threshold = 0.5
NQ: Prod_Threshold = 0.6
CL: Prod_Threshold = 0.55
```

**Impact**: Using wrong ML threshold = suboptimal filtering

#### Gap 2: Wrong Instrument List ❌
**Problem**:
- Config lists all 18 instruments
- Portfolio greedy search optimized for 10 specific symbols
- Trading wrong instruments = worse performance

**Should Be**:
```yaml
instruments:
  - 6A   # ✓ From greedy search
  - 6B   # ✓ From greedy search
  - 6C   # ✓ From greedy search
  - 6N   # ✓ From greedy search
  - 6S   # ✓ From greedy search
  - CL   # ✓ From greedy search
  - ES   # ✓ From greedy search
  - PL   # ✓ From greedy search
  - RTY  # ✓ From greedy search
  - SI   # ✓ From greedy search
```

#### Gap 3: Wrong Max Positions ❌
**Problem**:
- Config has `max_positions: 2`
- Portfolio greedy search optimized for `max_positions: 4`

**Impact**: Trading fewer positions = leaving performance on table

#### Gap 4: Wrong Models Path ❌
**Problem**:
- Config has `models_path: src/models/ibs/`
- Actual ML models are in `src/models/` (not in ibs/ subdirectory)

**Impact**: Strategy worker won't find ML models

#### Gap 5: No Per-Instrument Parameters ❌
**Problem**:
- All instruments use same `ibs_entry_high: 0.7`
- Optimization may have found instrument-specific filter settings
- Need to load per-instrument filter params (if they exist)

**Question**: Do you have per-instrument filter settings from optimization?

---

## How Existing System Works

### Old Architecture (live_worker.py)
```python
# 1. Load ML bundle for each symbol
for symbol in SYMBOLS:
    bundle = load_model_bundle(symbol)  # Loads {SYMBOL}_best.json

    # 2. Create strategy with optimized params
    cerebro.addstrategy(
        IbsStrategy,
        symbol=symbol,
        ml_model=bundle.model,           # ← From optimization
        ml_features=bundle.features,     # ← From optimization
        ml_threshold=bundle.threshold,   # ← From optimization
        size=1,
        # Generic filter params (same for all instruments)
        ibs_entry_high=0.3,
        ibs_exit_high=0.7,
    )
```

### What's Correct ✅
- Loads per-instrument ML bundles
- Uses optimized threshold per instrument
- Uses optimized feature set per instrument

### What's Missing ❌
- Per-instrument filter settings (if they exist)
- Portfolio-level constraints (10 symbols, 4 max positions)

---

## Required Integration: Multi-Alpha Architecture

### Solution Design

#### 1. Portfolio Configuration File
**New File**: `config/portfolio_optimization.json`

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
    "max_drawdown_dollars": 5878.99
  },
  "optimization_date": "2025-10-31",
  "optimization_method": "greedy_search"
}
```

#### 2. Per-Instrument Settings File (Optional)
**New File**: `config/ibs_instrument_params.json`

```json
{
  "ES": {
    "ibs_entry_high": 0.3,
    "ibs_exit_high": 0.7,
    "enable_stop": true,
    "stop_perc": 2.0,
    "enable_tp": true,
    "tp_perc": 3.0
  },
  "NQ": {
    "ibs_entry_high": 0.35,
    "ibs_exit_high": 0.65,
    ...
  },
  ...
}
```

**Question for User**: Do you have per-instrument filter settings from optimization, or do all instruments use the same generic filter parameters?

#### 3. Updated Config Loader
**File**: `src/config/config_loader.py`

Add functions:
```python
def load_portfolio_optimization(strategy_name: str) -> Dict:
    """Load portfolio greedy search results."""
    path = Path(f"config/portfolio_optimization_{strategy_name}.json")
    with open(path) as f:
        return json.load(f)

def load_instrument_params(strategy_name: str, symbol: str) -> Dict:
    """Load per-instrument optimized parameters (if they exist)."""
    path = Path(f"config/{strategy_name}_instrument_params.json")
    if not path.exists():
        return {}  # Use defaults from YAML

    with open(path) as f:
        all_params = json.load(f)

    return all_params.get(symbol, {})
```

#### 4. Updated Strategy Worker
**File**: `src/runner/strategy_worker.py`

```python
def _setup_strategy(self):
    # 1. Load ML bundle for primary symbol
    ml_bundle = load_model_bundle(
        self.primary_symbol,
        base_dir=self.strategy_config.models_path
    )

    # 2. Load instrument-specific params (if they exist)
    instrument_params = load_instrument_params(
        self.strategy_name,
        self.primary_symbol
    )

    # 3. Merge params (priority: instrument > strategy_params > defaults)
    strategy_params = {
        'symbol': self.primary_symbol,
        'ml_model': ml_bundle.model,        # ← From optimization
        'ml_features': ml_bundle.features,  # ← From optimization
        'ml_threshold': ml_bundle.threshold, # ← From optimization (per-instrument!)
        'portfolio_coordinator': self.portfolio_coordinator,
        **self.strategy_config.strategy_params,  # Global defaults
        **instrument_params,  # Instrument-specific overrides
    }

    # 4. Add strategy to Cerebro
    strategy_class = load_strategy(self.strategy_name)
    self.cerebro.addstrategy(strategy_class, **strategy_params)
```

#### 5. Updated Multi-Alpha Config
**File**: `config.multi_alpha.yml`

```yaml
strategies:
  ibs:
    enabled: true
    broker_account: ${TRADERSPOST_IBS_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/  # ← Fixed path!
    max_positions: 4          # ← From portfolio optimization!
    daily_stop_loss: 2500     # ← From portfolio optimization!

    # Load optimized symbol list from portfolio optimization
    instruments:
      - 6A   # ← From greedy search
      - 6B
      - 6C
      - 6N
      - 6S
      - CL
      - ES
      - PL
      - RTY
      - SI

    # Generic parameters (per-instrument params loaded separately)
    strategy_params:
      # Session windows
      use_window1: true
      start_time1: "0000"
      end_time1: "1500"
      use_window2: true
      start_time2: "1700"
      end_time2: "2400"

      # Generic IBS thresholds (can be overridden per-instrument)
      enable_ibs_entry: true
      ibs_entry_high: 0.3
      ibs_entry_low: 0.0

      enable_ibs_exit: true
      ibs_exit_high: 0.7
      ibs_exit_low: 0.3

      # Generic risk management (can be overridden per-instrument)
      enable_stop: true
      stop_type: "percent"
      stop_perc: 2.0

      enable_tp: true
      tp_type: "percent"
      tp_perc: 3.0

      # Note: ml_threshold loaded from {SYMBOL}_best.json per instrument!
```

---

## Integration Workflow

### When Strategy Worker Starts

```
1. Load config.multi_alpha.yml
   ↓
2. For strategy "ibs":
   a. Read instruments list: [6A, 6B, 6C, 6N, 6S, CL, ES, PL, RTY, SI]
   b. Read max_positions: 4
   c. Read daily_stop_loss: 2500
   d. Read models_path: src/models/
   ↓
3. For each instrument (e.g., ES):
   a. Load ML bundle: src/models/ES_best.json + ES_rf_model.pkl
      - ml_model: RandomForestClassifier
      - ml_features: [30 features specific to ES]
      - ml_threshold: 0.5 (from ES_best.json)

   b. Load instrument params (optional): config/ibs_instrument_params.json
      - instrument_params['ES']: {ibs_entry_high: 0.3, ...}

   c. Merge parameters:
      - Base: strategy_params from YAML
      - Override: instrument_params (if exist)
      - Add: ML bundle (model, features, threshold)

   d. Create strategy instance with merged params
   ↓
4. Initialize portfolio coordinator:
   - max_positions: 4
   - daily_stop_loss: 2500
   - allowed_symbols: [6A, 6B, ...]
   ↓
5. Start trading
```

---

## Questions for User

### 1. Per-Instrument Filter Settings
Do you have per-instrument optimized filter settings (e.g., `ibs_entry_high`, `ibs_exit_high`, `stop_perc`) from your optimization process?

**Options**:
- **A**: Yes, each instrument has different filter settings → Need `config/ibs_instrument_params.json`
- **B**: No, all instruments use the same filter settings → Use strategy_params from YAML

### 2. Portfolio Optimization File Location
Do you have the portfolio greedy search results saved in a file, or should I create it from the information in `PORTFOLIO_INTEGRATION_GUIDE.md`?

### 3. ML Model Directory Structure
Confirm the ML models are in:
- `src/models/{SYMBOL}_best.json`
- `src/models/{SYMBOL}_rf_model.pkl`

Not in subdirectories like `src/models/ibs/{SYMBOL}_best.json`?

### 4. Future Strategies
For future strategies (e.g., "breakout"), will you run the same optimization pipeline?
1. ML optimization per instrument → `{SYMBOL}_best.json`
2. Portfolio greedy search → Optimal symbol list + max_positions

---

## Impact Assessment

### If We DON'T Integrate Optimization Results ❌

**Performance Impact**:
1. **Wrong ML Thresholds**: Using 0.65 instead of optimized 0.5-0.6 → Worse filtering
2. **Wrong Symbol List**: Trading all 18 instead of optimized 10 → Lower Sharpe
3. **Wrong Position Limits**: Trading 2 instead of optimized 4 → Leaving money on table

**Expected Loss**: Could reduce Sharpe from 14.57 to <10

### If We DO Integrate Optimization Results ✅

**Performance Match**:
1. ✅ Use optimized ML thresholds per instrument
2. ✅ Trade optimized 10-symbol portfolio
3. ✅ Use optimized 4 max positions
4. ✅ Use optimized $2,500 daily stop

**Expected Result**: Sharpe ~14.57, CAGR ~84.4% (as optimized)

---

## Recommendation

### Immediate Actions (Week 3)

1. **Create Portfolio Optimization File**
   ```bash
   config/portfolio_optimization_ibs.json
   ```
   With optimal symbols and constraints from greedy search

2. **Update config.multi_alpha.yml**
   - Fix `models_path: src/models/`
   - Fix `max_positions: 4`
   - Fix `instruments` list to 10 optimized symbols

3. **Update Strategy Worker**
   - Add `load_model_bundle()` integration
   - Load per-instrument ML params from `{SYMBOL}_best.json`
   - Optionally load per-instrument filter params (if they exist)

4. **Test Parameter Loading**
   - Verify ML bundles load correctly
   - Verify thresholds match optimization results
   - Verify features match optimization results

### Future Strategy Addition Process

**For each new strategy (e.g., "breakout")**:

1. **ML Optimization**
   - Run optimization per instrument
   - Generate `{SYMBOL}_best.json` + `{SYMBOL}_rf_model.pkl`
   - Store in `src/models/breakout/`

2. **Portfolio Greedy Search**
   - Simulate portfolio with different symbol combinations
   - Find optimal: symbol list, max_positions, daily_stop_loss
   - Save in `config/portfolio_optimization_breakout.json`

3. **Configuration**
   - Add strategy to `config.multi_alpha.yml`
   - Use optimized instruments list
   - Use optimized max_positions and daily_stop_loss
   - Optionally create per-instrument params file

4. **Deployment**
   - Strategy worker auto-loads ML bundles
   - Strategy worker auto-loads portfolio optimization
   - System uses optimized parameters automatically

---

## Next Steps

1. **User Feedback**: Answer questions above
2. **Create Portfolio Optimization File**: From greedy search results
3. **Update Config System**: Integrate optimization loading
4. **Update Strategy Worker**: Load ML bundles per instrument
5. **Test Integration**: Verify parameters load correctly
6. **Update Roadmap**: Add optimization integration tasks

**Priority**: HIGH - Critical for achieving expected performance
