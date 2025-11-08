# Rooney Capital Trading System - Comprehensive Architecture Analysis

## Executive Summary

The Rooney Capital system is a **single-strategy, multi-asset automated futures trading system** built on Backtrader, with live market data from Databento, execution via TradersPost, and ML-based signal filtering. Currently **monolithic but with clear layer separation** - perfect foundation for multi-alpha refactoring.

**Current State**: Single IBS (Internal Bar Strength) strategy applied across 18 futures contracts (equity indices, commodities, currencies) with instrument-specific ML veto models and portfolio-level risk controls.

---

## 1. Overall Project Structure

```
rooney-capital-v1/
├── src/                          # Core production code
│   ├── runner/                   # Live trading orchestration
│   │   ├── main.py              # Entry point
│   │   ├── live_worker.py       # Worker lifecycle (2,400+ LOC)
│   │   ├── databento_bridge.py  # Market data ingestion
│   │   ├── portfolio_coordinator.py  # Position & risk management
│   │   ├── traderspost_client.py    # Order execution webhook
│   │   ├── contract_map.py      # Contract metadata
│   │   ├── historical_loader.py # Warmup data loading
│   │   ├── ml_feature_tracker.py    # Feature readiness tracking
│   │   └── __init__.py
│   ├── strategy/                 # Trading logic (6,200+ LOC)
│   │   ├── ibs_strategy.py      # Main strategy implementation
│   │   ├── feature_utils.py     # Indicator calculations
│   │   ├── filter_column.py     # Filter state management
│   │   ├── contract_specs.py    # Contract specs & multipliers
│   │   ├── safe_div.py          # Safe math primitives
│   │   └── __init__.py
│   ├── models/                   # ML artifacts
│   │   ├── loader.py            # Model bundle loading
│   │   ├── ensemble/            # Ensemble optimization utilities
│   │   ├── {SYMBOL}_best.json   # Per-instrument model metadata
│   │   ├── {SYMBOL}_rf_model.pkl    # Per-instrument RF classifier
│   │   └── __init__.py
│   ├── evaluation/               # Performance metrics
│   │   ├── performance_metrics.py
│   │   └── __init__.py
│   ├── monitoring/               # Health monitoring
│   │   ├── performance_monitor.py
│   │   └── __init__.py
│   ├── utils/                    # Utilities
│   │   ├── discord_notifier.py  # Alert notifications
│   │   ├── trades_db.py         # Trade persistence
│   │   └── __init__.py
│   └── config.py                # Shared constants
├── research/                     # Backtesting & optimization
│   ├── backtest_runner.py       # Backtrader test harness
│   ├── extract_training_data.py # Historical data extraction
│   ├── portfolio_optimizer_*.py # Portfolio allocation tools
│   ├── portfolio_simulator.py   # Monte Carlo simulator
│   ├── production_retraining.py # Retraining pipeline
│   └── extract_*.sh             # Data extraction scripts
├── tests/                        # Integration & unit tests
│   ├── runner/                  # Live worker tests
│   ├── strategy/                # Strategy unit tests
│   ├── conftest.py
│   └── test_*.py
├── dashboard/                    # Monitoring dashboard
│   ├── app.py                   # Flask app
│   ├── metrics.py               # Metrics calculation
│   ├── utils.py
│   └── statistical_monitor.py
├── Data/                         # Static metadata
│   └── Databento_contract_map.yml
├── config.yml                    # Runtime configuration
├── config.example.yml            # Config template
├── .env.example                  # Secrets template
└── requirements.txt              # Dependencies
```

---

## 2. Current Architecture Pattern

### **Architecture Type**: Semi-Modular Monolith

**Current Organization**:
- **Tightly Coupled**: Single IBS strategy with hardcoded logic applied to all instruments
- **Layered but Integrated**: Data pipeline → Strategy → Execution in one flow
- **Limited Abstraction**: No strategy interface/plugin system for alternatives
- **Shared State**: Global configuration, commission rates, pair mappings applied uniformly

**Key Characteristic**: **Single-strategy, single-asset-class focus** = easy to understand but difficult to extend with alternative strategies

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Worker (Main Loop)                  │
│  - Lifecycle management, preflight checks, shutdown          │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    v                         v
┌─────────────────┐   ┌──────────────────┐
│  Backtrader     │   │  Databento       │
│  Cerebro (BT)   │   │  Live Stream     │
├─────────────────┤   ├──────────────────┤
│ Data Feeds:     │   │ Tick Aggregation │
│ - Hourly        │   │ → 1-min OHLCV    │
│ - Daily         │   │ → Per-symbol     │
│ - Reference     │   │    Queue         │
│ (TLT_day, etc)  │   │                  │
└────────┬────────┘   └──────────────────┘
         │
         v
    ┌──────────────────────────────────┐
    │   IbsStrategy (Single Instance)   │
    │                                   │
    │ - Per-symbol logic (hardcoded)    │
    │ - IBS calculations                │
    │ - ML veto models (per-symbol)     │
    │ - Risk controls                   │
    │ - Portfolio coordination          │
    └────────┬─────────────────────────┘
             │
             v
    ┌──────────────────────────────────┐
    │  TradersPost Webhook              │
    │  (Order/Trade Notifications)      │
    │                                   │
    │ Payload: symbol, qty, price,     │
    │          filters, ML score        │
    └──────────────────────────────────┘
```

---

## 3. Data Pipeline

### **Market Data Ingestion** (Databento → Live Strategy)

#### Entry Point: `LiveWorker.run()`

```
Databento Live Stream (Tick Data)
    ↓ (DatabentoSubscriber)
Per-symbol Tick Queue (thread-safe)
    ↓ (DatabentoLiveData + Aggregation)
1-minute OHLCV Bars per Symbol
    ↓ (Backtrader Feed)
IbsStrategy.next() → Indicator Calculation
```

**Key Components**:

1. **DatabentoSubscriber** (`databento_bridge.py`)
   - Connects to Databento Live API
   - Subscribes to contracts via product IDs or parent symbols
   - Aggregates `TradeMsg` ticks into 1-minute OHLCV bars
   - Handles reconnection and data gaps
   - Publishes bars to `QueueFanout`

2. **QueueFanout** (thread-safe container)
   - Per-symbol queues for concurrent readers
   - Tracks instrument ID ↔ root symbol mappings
   - Detects contract rolls via `SymbolMappingMsg`
   - Bounded queue sizes prevent runaway memory

3. **DatabentoLiveData** (Backtrader feed adapter)
   - Reads from per-symbol queues
   - Adapts to Backtrader's `DataBase` interface
   - Handles missing bars gracefully
   - Writes heartbeat file for monitoring

4. **Resample Feeds**
   - `HourlyResampledLiveData`: Rolls up minutes into hourly bars
   - `DailyResampledLiveData`: Rolls up minutes/hourly into daily bars
   - Required for multi-timeframe indicators (IBS daily vs hourly)

#### Warmup & Historical Data

```
Historical Loader (load_historical_data)
    ↓
Databento Historical API (lookback 252 days for daily, 15 for hourly)
    ↓
Compressed Bars (1h, 1d aggregation options)
    ↓
Backtrader Feed Warmup (before live stream starts)
```

**Feature**:
- Configurable lookback: `historical_lookback_days` (default 252 daily bars)
- Compression options: 1min, 1h, 1d
- Batch loading to manage memory
- Prevents indicator NaN during initial bars

#### Contract Roll Handling

- **Via SymbolMappingMsg**: Databento sends symbol changes during rolls
- **Tracked in QueueFanout**: Maps instrument ID to latest contract symbol
- **Passed to TradersPost**: Execution uses current contract for order routing

**Configuration** (in `config.yml`):
```yaml
load_historical_warmup: true
historical_lookback_days: 252
historical_warmup_compression: 1h
backfill: true
backfill_days: 4
resample_session_start: "23:00"
queue_maxsize: 4096
```

---

## 4. Strategy Implementation

### **Single Strategy**: `IbsStrategy` (extends `bt.Strategy`)

**Location**: `src/strategy/ibs_strategy.py` (6,246 LOC)

**Responsibility**: 
- Compute IBS (Internal Bar Strength) indicators
- Apply ML veto filters
- Generate buy/sell/exit signals
- Manage intraday position lifecycle
- Track P&L and performance

#### Core Workflow per Bar (`next()`)

```
1. Collect OHLC data from feeds
   ├─ Intraday (minute/hourly) for symbol
   ├─ Daily for symbol (reference data)
   └─ Daily for pair symbol (ES/NQ, CL/NG, etc.)

2. Calculate Indicators (feature_utils.py)
   ├─ IBS: (close - low) / (high - low)
   ├─ RSI (2 and 14 lengths)
   ├─ Bollinger Bands
   ├─ ATR (Average True Range)
   ├─ Z-scores (ATR, Volume, Distance)
   ├─ Moving averages & slopes
   ├─ Donchian breakouts
   ├─ Parabolic SAR
   └─ ~50+ indicators total

3. Apply Filters (filter_column.py)
   ├─ Time-of-day gate (skip low-liquidity periods)
   ├─ Volatility filters (ATR extremes)
   ├─ Trend confirmation (MA slopes)
   ├─ Pair correlation (cross-symbol IBS)
   └─ Previous bar momentum

4. Consult ML Veto Models (per-symbol)
   ├─ Load required features from state
   ├─ Call RF classifier: predict_proba()
   ├─ Compare vs. threshold (e.g., 65%)
   └─ Veto if confidence too low

5. Generate Signals
   ├─ LONG: IBS high + filters pass + ML ok
   ├─ SHORT: IBS low + filters pass + ML ok
   └─ EXIT: Opposite IBS or time limit

6. Order Management
   ├─ Check position limit (PortfolioCoordinator)
   ├─ Size from contract config (1 contract per instrument)
   ├─ Send to TradersPost on notify_order()
   └─ Track in portfolio state

7. Record ML Features (for retraining)
   ├─ Collect filter values
   ├─ Snapshot at signal generation time
   └─ Export via ml_feature_tracker
```

#### Key Signals

**Entry Conditions** (example for LONG):
```python
ibs_hourly > 70%  # High IBS in intraday
+ ibs_daily > 40% # Allow in daily trend
+ prev_bar_momentum > threshold
+ no time-of-day restrictions
+ pair_ibs < critical level
+ ML(predict_proba > threshold)
→ BUY signal
```

**Exit Conditions**:
```python
ibs < 30% (opposite extreme)
OR time_in_trade > max_duration
OR portfolio daily loss > limit
→ CLOSE position
```

#### Multi-Timeframe Approach

- **1-minute bars**: Base feed from Databento
- **Hourly bars**: Resampled for hourly IBS, ATR volatility
- **Daily bars**: Reference data for daily IBS, trends
- **Pair symbols**: Cross-instrument filtering (ES uses NQ daily, etc.)

**Example**: ES uses:
- ES minute/hourly (self)
- ES daily (reference)
- NQ daily (pair reference)
- TLT daily (market regime reference)

#### Feature Engineering (FilterColumn)

Each signal generates a snapshot of ~30-50 numeric features:
```python
{
    'ibs_pct': 75,              # IBS percentile
    'daily_ibs_pct': 55,        # Daily IBS
    'prev_bar_pct': 8.5,        # Intraday momentum
    'rsi_2': 92,                # RSI length 2
    'rsi_14': 68,               # RSI length 14
    'atr_z_pct': 88,            # ATR z-score percentile
    'daily_atr_z_pct': 45,      # Daily ATR z-score
    'pair_ibs_pct': 32,         # Pair IBS percentile
    'bb_high': 0.85,            # Bollinger position
    'donch': 72,                # Donchian proximity
    'ma_slope_fast': 0.3,       # MA slope
    'volume_z_pct': 78,         # Volume z-score
    ... (25+ more features)
}
```

These are **logged for ML training** but currently used only for **veto**, not generation.

#### ML Veto System

**Per-Instrument Model Loading** (`models/loader.py`):
```python
bundle = load_model_bundle("ES")
# bundle.model = trained RandomForestClassifier
# bundle.features = ['ibs_pct', 'prev_bar_pct', 'rsi_2', ...]
# bundle.threshold = 0.65  # confidence threshold
```

**Veto Application**:
```python
if signal_generated:
    features_df = extract_features(signal_snapshot, bundle.features)
    proba = bundle.model.predict_proba(features_df)[0][1]  # P(win)
    if proba < bundle.threshold:
        veto_trade()  # Don't execute
    else:
        allow_trade()  # Execute via TradersPost
```

**Current Approach**: ML is a **filter, not generator** (only filters bad signals)

---

## 5. Execution Layer

### **Broker Integration**: TradersPost → Tradovate

**Flow**:
```
IbsStrategy.notify_order()
    ↓ (Signal: buy, sell, exit)
TradersPostClient.post_order()
    ↓ (HTTP POST with retry)
TradersPost Webhook
    ↓ (Routing via configured strategy)
Tradovate Futures Account
    ↓ (Order execution at market)
```

#### Order Routing (`traderspost_client.py`)

**Webhook Payload** (per signal):
```json
{
  "event": "order",
  "symbol": "ES",
  "action": "buy",
  "quantity": 1,
  "price": 4512.50,
  "thresholds": {
    "ml_score": 0.78,
    "ibs_pct": 85,
    "prev_bar_pct": 12.3,
    "filter_snapshot": { ... }
  },
  "metadata": {
    "tradovate_symbol": "ES",
    "databento_product_id": "ES.FUT",
    "timestamp": "2024-11-08T14:30:00Z"
  }
}
```

**Retry Logic**:
- Exponential backoff: 0.5s, 1s, 2s, 4s (default 3 retries)
- Retriable: connection errors, 5xx responses
- Fatal: 4xx errors (auth failure, invalid symbol)
- Timeout: 10 seconds per attempt

**Trade Notifications**:
```python
notify_trade()
    ↓ (Trade closed)
post_trade() with:
    - entry price, exit price, P&L
    - entry/exit timestamps
    - exit snapshot data
```

#### Position Management Layers

**Layer 1**: IbsStrategy-internal
- Tracks intraday position (size, entry price, entry time)
- Exit triggers (opposite IBS, time limit)

**Layer 2**: PortfolioCoordinator
- Max concurrent positions across all symbols (default 2)
- Daily portfolio stop loss (default $2,500)
- Thread-safe: can block new entries atomically
- Registers open/closed positions with P&L

**Layer 3**: Contract Configuration
```yaml
contracts:
  ES:
    size: 1                # Number of contracts
    commission: 4.0        # Per-side cost
  NQ:
    size: 1
    commission: 4.0
```

#### Execution Symbols vs. Reference Symbols

**Execution** (can open positions): 9 symbols (from `config.yml`)
```yaml
portfolio:
  instruments:
    - 6A, 6B, 6C, 6M, 6N, 6S  # Currencies
    - CL, HG, SI                # Commodities
```

**Monitor/Analyze** (signals computed but not traded): ES, NQ, RTY, YM, etc.

---

## 6. Backtesting Framework

### **Backtrader Foundation**

**Library**: `backtrader` (community fork of original)

**Usage in Research**:
```python
# research/backtest_runner.py
cerebro = bt.Cerebro()
cerebro.broker.setcash(starting_cash)
cerebro.broker.setcommission(commission)

# Add data feeds
cerebro.adddata(CSVDataFeed(...))
cerebro.adddata(DailyResampledFeed(...))

# Add strategy
cerebro.addstrategy(IbsStrategy, **strategy_kwargs)

# Run
results = cerebro.run()
```

**Backtest Pipeline** (`research/extract_training_data.py` → Model Training):

```
Raw Databento Data
    ↓ (extract_training_data.py)
Per-symbol Chunks (CSV)
    ├─ OHLCV + Features + Signals
    └─ 2-year rolling windows
    
    ↓ (Backtrader Replay)
Backtest Simulation
    ├─ Run IbsStrategy on historical
    ├─ Collect entry/exit signals
    └─ Label outcomes (win/loss)

    ↓ (Feature Snapshot + Outcome)
Training Data
    ├─ One row per signal
    ├─ Features: [ibs_pct, prev_bar_pct, rsi_2, ...]
    └─ Label: [0=loss, 1=win]

    ↓ (sklearn RandomForestClassifier)
ML Training (production_retraining.py)
    ├─ Cross-validated fit
    ├─ Optimize threshold via ROC curve
    └─ Save model + metadata
```

**Cross-Validation Strategy** (`three_way_split`):
- Train: 60% of data
- Validation: 20% (threshold optimization)
- Test: 20% (final performance estimate)

**Portfolio Optimization** (`portfolio_optimizer_*.py`):
- Input: Individual symbol Sharpe ratios
- Method: Greedy allocation or Bayesian optimization
- Output: Position sizes per symbol to maximize portfolio Sharpe
- Constraint: Max drawdown, exposure limits

#### Limitations of Current Backtest Setup

1. **Simplified Execution**: No slippage, no partial fills
2. **Single Strategy**: Can't test multiple strategies side-by-side
3. **Manual Optimization**: No automated hyperparameter search for strategy rules
4. **Look-ahead Risk**: Features calculated with end-of-bar close (potential bias)
5. **No Realistic Commission**: Fixed per-contract, not account-dependent

---

## 7. Configuration Management

### **Three-Tier Configuration**

#### **Tier 1: Hardcoded Constants** (`src/config.py`)

```python
DEFAULT_COMMISSION_PER_SIDE = 1.25  # Per contract
DEFAULT_PAIR_MAP = {
    "ES": "NQ",    # ES uses NQ for cross-symbol filters
    "NQ": "ES",
    "RTY": "YM",
    "CL": "NG",
    # ... 10 pairs defined
}
REQUIRED_REFERENCE_FEEDS = ("TLT_day",)  # Daily TLT required
```

**Override via Environment**:
```bash
PINE_COMMISSION_PER_SIDE=2.0     # $2 instead of $1.25
PAIR_MAP='{"ES":"NQ","NQ":"ES"}'  # JSON override
```

#### **Tier 2: Runtime Configuration File** (`config.yml`)

```yaml
# Paths
contract_map: Data/Databento_contract_map.yml
models_path: src/models

# Data
symbols: [6A, 6B, ..., YM]  # All symbols to load
databento_api_key: ${DATABENTO_API_KEY}
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}

# Portfolio
portfolio:
  max_positions: 2
  daily_stop_loss: 2500
  instruments: [6A, 6B, CL, HG, SI]  # Actual trading symbols

# Runtime
starting_cash: 250000
load_historical_warmup: true
backfill: true
backfill_days: 4

# Preflight checks
preflight:
  enabled: true
  skip_ml_validation: false
  fail_fast: true
```

**Load Order**:
1. Parse YAML file
2. Expand `${VAR}` from environment
3. Validate schema
4. Create `RuntimeConfig` dataclass

#### **Tier 3: Instrument Overrides** (`contracts` section)

```yaml
contracts:
  ES:
    size: 1
    commission: 4.0
    multiplier: 50         # Override if different
    margin: 4000          # Account requirement
    strategy_overrides:   # Per-symbol strategy tweaks
      max_bars_in_trade: 30
      use_pair_filter: true
```

**Runtime Application**:
```python
config = load_runtime_config()
for symbol in config.symbols:
    instr_cfg = config.instrument(symbol)  # Resolved config
    strategy_kwargs = {
        'symbol': symbol,
        'contract_size': instr_cfg.size,
        'commission': instr_cfg.commission,
        **instr_cfg.strategy_overrides,
        **ml_bundle.strategy_kwargs(),  # Add ML model
    }
    cerebro.addstrategy(IbsStrategy, **strategy_kwargs)
```

#### **Contract Map** (`Data/Databento_contract_map.yml`)

Metadata for each trading instrument:
```json
{
  "symbol": "ES",
  "tradovate_symbol": "ES",
  "tradovate_description": "E-mini S&P 500 (CME)",
  "databento": {
    "dataset": "GLBX.MDP3",
    "product_id": "ES.FUT"
  },
  "roll": {
    "stype_in": "parent"
  },
  "optimized": true,
  "reference_feeds": [
    {
      "dataset": "GLBX.MDP3",
      "feed_symbol": null,
      "product_id": "TLT.FUT"
    }
  ]
}
```

---

## 8. State Management

### **Position & P&L Tracking**

**Hierarchy**:

```
┌─────────────────────────────────────────┐
│  Backtrader Broker                      │
│  (Account-level state)                  │
│  - Cash, portfolio value, margin used   │
└────────────────────┬────────────────────┘
                     │
                     v
┌─────────────────────────────────────────┐
│  PortfolioCoordinator (Thread-safe)     │
│                                         │
│  open_positions: {symbol → PositionInfo}│
│  daily_pnl: float (reset each day)      │
│  stopped_out: bool                      │
│  pending_positions: reserved slots      │
└────────────────────┬────────────────────┘
                     │
                     v
┌─────────────────────────────────────────┐
│  IbsStrategy (Per-Symbol)               │
│                                         │
│  position: backtrader position object   │
│  bars_in_trade: int                     │
│  entry_signal_snapshot: filter values   │
│  pending_exit: exit signal metadata     │
└─────────────────────────────────────────┘
```

#### **PortfolioCoordinator** (`src/runner/portfolio_coordinator.py`)

**Data Structures**:
```python
@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_time: datetime
    entry_price: Optional[float]

class PortfolioCoordinator:
    open_positions: Dict[str, PositionInfo]      # Currently active
    pending_positions: Dict[str, datetime]       # Order pending
    daily_pnl: float = 0.0
    stopped_out: bool = False
    current_day: Optional[date]
```

**Key Methods**:
```python
can_open_position(symbol) → (bool, reason_str)
    # Check: not already open, not pending, <max_pos, not stopped out
    
register_position_opened(symbol, size, entry_price)
    # Move from pending to open
    
register_position_closed(symbol, pnl, exit_time)
    # Remove from open, add to daily_pnl, check stop loss
    
reset_daily_state(trading_day)
    # Reset daily_pnl, stopped_out at session start
```

**Thread Safety**: `threading.RLock()` protects all mutable state

#### **IbsStrategy Internal State**

Per-symbol state maintained across bars:

```python
self.position: Optional[Position]              # Backtrader position object
self.bars_in_trade: int                        # Bars since entry
self.max_bars_in_trade: int = 100              # Exit trigger

# Signal snapshots for ML retraining
self.entry_signal_snapshot: Dict[str, float]   # Entry features
self.pending_exit: Dict[str, Any]              # Exit context
```

#### **Trade Persistence** (`src/utils/trades_db.py`)

Optional SQLite database for completed trades:

```python
TradesDB(db_path="trades.db")
    .record_trade({
        'symbol': 'ES',
        'entry_price': 4500.5,
        'exit_price': 4510.2,
        'entry_time': datetime(...),
        'exit_time': datetime(...),
        'pnl': 485.0,  # (4510.2 - 4500.5) * 50
        'bars_held': 15,
    })
```

#### **Heartbeat & Monitoring** (`live_worker.py`)

Periodic JSON file written to track system health:

```json
{
  "timestamp": "2024-11-08T14:30:00Z",
  "status": "running",
  "cash": 248750.0,
  "portfolio_value": 260000.0,
  "daily_pnl": 2500.0,
  "open_positions": ["ES", "NQ"],
  "ml_feature_readiness": {
    "ES": {
      "ready": true,
      "features_received": 47,
      "last_update": "2024-11-08T14:29:58Z"
    },
    "NQ": { "ready": true, ... },
    ...
  }
}
```

**File**: `/var/run/pine/worker_heartbeat.json` (configurable)
**Interval**: 30 seconds (configurable)
**Usage**: External monitors check file modification time for stalls

---

## 9. Entry Points & Execution Modes

### **Production Entry Point**

```bash
# Via environment variable
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python -m runner.main

# Or inline
python - <<'PY'
from runner import LiveWorker, load_runtime_config
config = load_runtime_config()
worker = LiveWorker(config)
worker.run()
PY
```

**Lifecycle** (`LiveWorker.run()`):

```
1. load_runtime_config()
   └─ Parse YAML, expand env vars, validate

2. run_preflight_checks() [if enabled]
   ├─ ML Models: load each symbol's bundle, smoke test
   ├─ TradersPost: POST health check
   ├─ Databento: metadata probe
   ├─ Reference Data: verify TLT, pair mappings
   └─ Data Feeds: confirm Backtrader feed registration

3. Initialize Components
   ├─ Create Backtrader Cerebro instance
   ├─ Instantiate DatabentoSubscriber (live feed)
   ├─ Load historical data [if enabled]
   ├─ Create PortfolioCoordinator
   └─ Create TradersPost client

4. Attach Data Feeds to Cerebro
   ├─ Per-symbol: minute feed + hourly resampled + daily resampled
   ├─ Per-pair: daily pair data
   └─ Reference: TLT daily

5. Attach Strategy
   └─ NotifyingIbsStrategy with order/trade callbacks

6. cerebro.run()
   └─ Live event loop (blocks)

7. Shutdown Handling (SIGINT/SIGTERM)
   ├─ Close open positions
   ├─ Flush heartbeat
   └─ Clean shutdown
```

### **Research Entry Points**

#### **Backtesting**:
```bash
python research/backtest_runner.py --symbols ES,NQ --start 2023-01-01 --end 2024-01-01
```

#### **Data Extraction**:
```bash
python research/extract_training_data.py --symbols ES --output training_data.csv
```

#### **Portfolio Optimization**:
```bash
python research/portfolio_optimizer_full.py --input symbol_sharpes.json --output allocation.json
```

#### **ML Model Training**:
```bash
python research/production_retraining.py --symbols ES,NQ --data training_data/ --output models/
```

### **Monitoring Entry Points**

#### **Preflight Validation**:
```bash
python scripts/worker_preflight.py
# Tests: ML models, TradersPost connection, Databento, reference data
```

#### **Feature Readiness Check**:
```bash
python scripts/inspect_ml_features.py --show-features
# Reads heartbeat file or instantiates worker to check ML feature readiness
```

#### **Dashboard**:
```bash
cd dashboard && python app.py
# Runs Flask app on port 5000 with real-time metrics
```

---

## 10. Dependencies Between Components

### **Dependency Graph**

```
┌──────────────────────────────┐
│   Runtime Config             │
│   (config.yml + env vars)    │
└─────────────┬────────────────┘
              │
    ┌─────────┼─────────┬──────────────┐
    │         │         │              │
    v         v         v              v
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Models  │ │ Contract │ │ Strategy │ │ Databento    │
│ Loader  │ │ Map      │ │ Config   │ │ Bridge       │
└────┬────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
     │           │            │              │
     └───────────┼────────────┼──────────────┘
                 │            │
                 v            v
    ┌────────────────────────────────────┐
    │  IbsStrategy (per symbol, in BT)   │
    │                                    │
    │ - Reads price feeds (minute/h/d)  │
    │ - Computes indicators from config │
    │ - Calls ML veto (per-symbol model)│
    │ - Generates entry/exit signals    │
    └───────────┬──────────────────────┘
                │
    ┌───────────┴──────────────────┐
    │                              │
    v                              v
┌──────────────────┐      ┌──────────────────────┐
│ PortfolioCoord   │      │ TradersPost Client   │
│                  │      │                      │
│ Check position   │      │ POST webhook with:   │
│ limit, stop loss │      │ - symbol, action, qty│
│ Register opens/  │      │ - price, filters     │
│ closes           │      │ - ML score snapshot  │
└──────────────────┘      └──────────────────────┘
                                 │
                                 v
                          ┌──────────────────────┐
                          │ TradersPost → Orders │
                          │ (Tradovate Broker)   │
                          └──────────────────────┘
```

### **Critical Coupling Points**

1. **IbsStrategy ↔ PortfolioCoordinator**
   - Strategy calls `can_open_position()`, `register_position_opened/closed()`
   - Decorator pattern: PortfolioCoordinator blocks entries

2. **IbsStrategy ↔ ML Models**
   - Strategy loads per-symbol models at init
   - Calls `predict_proba()` on feature snapshot
   - Hard-coupled: only works if model/features match

3. **IbsStrategy ↔ Configuration**
   - Symbol-specific contract sizes, commissions
   - Instrument-level filter thresholds
   - Pair mappings for cross-symbol filters

4. **LiveWorker ↔ Backtrader**
   - LiveWorker creates Cerebro, adds feeds & strategy
   - Backtrader drives `strategy.next()` on bar arrivals
   - Callbacks: `notify_order()`, `notify_trade()` → TradersPostClient

5. **DatabentoSubscriber ↔ IbsStrategy (via QueueFanout)**
   - Subscriber publishes bars to per-symbol queues
   - Strategy reads via DatabentoLiveData feed
   - Contract roll info passed through `Bar.contract_symbol`

### **Loose Coupling Opportunities**

1. **Strategy Interface**: Currently only `IbsStrategy` → Easy to add alternatives
2. **Data Feed**: Currently Databento only → Could support other vendors (e.g., IB API)
3. **Execution**: Currently TradersPost only → Could add direct API (e.g., IB, Broker)
4. **Configuration**: Flexible YAML + env vars → Easy to extend

---

## 11. System Characteristics Summary

### **Strengths**

✓ **Clear Separation of Concerns**: Runner/Strategy/Models layers distinct
✓ **Thread-Safe Position Management**: PortfolioCoordinator handles concurrency
✓ **Flexible Configuration**: YAML + environment variables + per-instrument overrides
✓ **Comprehensive Backtesting**: Full research pipeline from raw data → models
✓ **Multi-Timeframe Support**: Minute/hourly/daily indicator calculations
✓ **ML Integration**: Per-symbol veto models with thresholds
✓ **Production-Ready**: Heartbeat, preflight checks, graceful shutdown
✓ **Rich Feature Engineering**: 50+ technical indicators computed per signal

### **Weaknesses**

✗ **Single Strategy Only**: IbsStrategy hardcoded, no plugin system
✗ **Tightly Coupled ML**: Models hardwired into strategy; can't swap logic
✗ **Limited Abstraction**: No strategy interface, feed interface, or execution interface
✗ **Manual Optimization**: No auto hyperparameter tuning for strategy rules
✗ **Research/Live Sync Risk**: Changes in research code require manual sync to production
✗ **Scalability**: Adding new strategies requires code modification, not configuration
✗ **No Multi-Strategy Allocation**: Can't run multiple strategies or alpha blending

### **Refactoring Opportunities for Multi-Alpha**

1. **Strategy Factory Pattern**: 
   - Define `BaseStrategy` interface
   - Implement `IbsStrategy`, `MeanReversionStrategy`, `BreakoutStrategy`, etc.
   - Load from config: `strategies: [IbsStrategy, MeanReversionStrategy]`

2. **Modular ML Layer**:
   - Separate veto logic from signal generation
   - Allow per-strategy model selection
   - Support ensemble model predictions

3. **Data Abstraction**:
   - Define feed interface for Databento, IB, etc.
   - Support multiple data sources simultaneously

4. **Alpha Blending**:
   - Each strategy produces confidence scores
   - Portfolio allocator combines signals with position limits

5. **Configuration-Driven**:
   - Define strategies, parameters, allocation weights in config
   - Zero code changes for new strategy additions

---

## Technical Metrics

| Metric | Value |
|--------|-------|
| **Production Code LOC** | ~13,946 lines |
| **IbsStrategy LOC** | 6,246 lines |
| **Runner Module LOC** | 2,400+ lines |
| **Strategy Module LOC** | 6,600+ lines |
| **Supported Symbols** | 18 (9 traded, 9 reference) |
| **Number of Instruments** | 18 contracts |
| **ML Models** | 1 per symbol (18 total, Random Forest) |
| **Indicator Features** | 50+ computed per signal |
| **Backtrader Feeds** | Per symbol: 1-min, hourly, daily; Reference: daily |
| **Configuration Levels** | 3 (hardcoded, file, per-instrument) |
| **Test Coverage** | Integration tests for live worker & strategy |

---

## Conclusion

The Rooney Capital system is a **well-engineered single-strategy automated trading system** with:
- ✓ Clean data pipeline (Databento → Backtrader → Orders)
- ✓ Comprehensive risk management (portfolio limits, daily stop loss)
- ✓ Production-grade operationalization (heartbeat, preflight, graceful shutdown)
- ✓ Strong research foundation (historical backtesting, ML integration)

However, it's **monolithic at the strategy level**—the IBS logic is baked into the codebase. Refactoring to a **multi-alpha architecture** would require:

1. Defining a **Strategy interface** to support multiple signal sources
2. Creating a **Portfolio allocator** to blend multiple alphas
3. Decoupling **ML models** from strategy implementation
4. Making strategies **configuration-driven** instead of code-driven

This document provides the foundation for that refactoring effort.

