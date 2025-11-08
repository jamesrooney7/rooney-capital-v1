# Multi-Alpha Architecture Refactoring Roadmap

## Executive Summary

This roadmap outlines the transformation of the Rooney Capital trading system from a single-strategy monolith to a multi-alpha architecture supporting independent strategy workers. The refactoring will be completed over **4 weeks** with an aggressive timeline, enabling multiple strategies to run concurrently with complete isolation.

**Key Objectives:**
- ✅ Enable multiple independent strategies to run simultaneously
- ✅ Share only market data infrastructure (Redis-based data hub)
- ✅ Maintain separate broker accounts and P&L tracking per strategy
- ✅ Preserve existing backtesting and optimization framework
- ✅ Zero downtime migration from current system
- ✅ Foundation for rapid addition of new strategies

**Timeline:** 4 weeks (aggressive)
**Risk Level:** Medium (parallel running during migration reduces risk)
**Expected Outcome:** 2+ strategies running independently by end of Week 4

---

## Target Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    Data Hub Process                     │
│  - Databento Live Stream Subscriber                     │
│  - Tick Aggregation → 1-min OHLCV Bars                  │
│  - Publish to Redis Pub/Sub                             │
│  - Cache: market:{symbol}:{timeframe}                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Redis Server    │
              │  - Pub/Sub       │
              │  - Latest Bar    │
              │  - Historical    │
              └─────────┬────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ...
│ IBS Worker    │ │ Breakout      │ │ Strategy N    │
│               │ │ Worker        │ │ Worker        │
├───────────────┤ ├───────────────┤ ├───────────────┤
│ Process 1     │ │ Process 2     │ │ Process N     │
│               │ │               │ │               │
│ Shared:       │ │ Shared:       │ │ Shared:       │
│ - Redis Sub   │ │ - Redis Sub   │ │ - Redis Sub   │
│ - Feature Lib │ │ - Feature Lib │ │ - Feature Lib │
│               │ │               │ │               │
│ Independent:  │ │ Independent:  │ │ Independent:  │
│ - IBS Logic   │ │ - Breakout    │ │ - Strategy    │
│ - IBS Models  │ │   Logic       │ │   Logic       │
│ - Account 1   │ │ - Breakout    │ │ - Models N    │
│   ($150k)     │ │   Models      │ │ - Account N   │
│ - P&L Track   │ │ - Account 2   │ │   ($150k)     │
│ - Config      │ │   ($150k)     │ │ - P&L Track   │
│               │ │ - P&L Track   │ │ - Config      │
└───────────────┘ └───────────────┘ └───────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ TradersPost   │ │ TradersPost   │ │ TradersPost   │
│ Webhook 1     │ │ Webhook 2     │ │ Webhook N     │
└───────────────┘ └───────────────┘ └───────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────────────────────────────────────────────┐
│              Tradovate Broker Accounts                │
│  - Account 1 ($150k) - IBS                           │
│  - Account 2 ($150k) - Breakout                      │
│  - Account N ($150k) - Strategy N                    │
└───────────────────────────────────────────────────────┘
```

### Shared Components (The "Commissary Kitchen")

1. **Data Hub** - Single process publishing market data
2. **Feature Engineering Library** (`src/features/`) - Shared filter calculations
3. **Backtesting Framework** (`research/lib/`) - Shared evaluation methodology
4. **Portfolio Optimizer** - Greedy instrument selection (existing)
5. **Contract Specifications** - Instrument metadata (tick sizes, multipliers)
6. **Configuration Schema** - Common config structure

### Independent Components (The "Food Trucks")

1. **Strategy Logic** - Unique entry/exit rules per strategy
2. **ML Models** - Per-strategy, per-instrument trained models
3. **Broker Accounts** - Separate TradersPost webhooks → Tradovate accounts
4. **P&L Tracking** - Completely isolated performance tracking
5. **Risk Management** - Independent position limits, stop losses
6. **Process Lifecycle** - Start/stop/restart without affecting others

---

## Week-by-Week Roadmap

### **Week 1: Foundation - Build Multi-Alpha Architecture**

**Goal:** Create the foundational infrastructure for multi-alpha system without breaking existing system.

#### Phase 1.1: Data Hub Implementation (Days 1-2)

**Tasks:**

1. **Install and configure Redis**
   - [ ] Install Redis server on trading machine
   - [ ] Configure persistence settings (AOF for durability)
   - [ ] Set up monitoring (redis-cli, memory usage)
   - [ ] Create Redis connection wrapper (`src/data_hub/redis_client.py`)

2. **Create Data Hub Process** (`src/data_hub/`)
   - [ ] Create `data_hub_main.py` - Entry point for data hub
   - [ ] Port `DatabentoSubscriber` from `databento_bridge.py`
   - [ ] Implement Redis publisher:
     ```python
     def publish_bar(symbol, timeframe, bar):
         channel = f"market:{symbol}:{timeframe}"
         redis_client.publish(channel, json.dumps(bar))
         redis_client.set(f"market:{symbol}:{timeframe}:latest", json.dumps(bar))
     ```
   - [ ] Add bar aggregation (1min, hourly, daily) with resampling
   - [ ] Implement contract roll detection and publishing
   - [ ] Add heartbeat mechanism for monitoring
   - [ ] Add graceful shutdown handling (SIGTERM, SIGINT)

3. **Create Redis Feed Adapter** (`src/feeds/redis_feed.py`)
   - [ ] Implement `RedisLiveData` (Backtrader-compatible feed)
   - [ ] Subscribe to Redis channels: `market:{symbol}:1min`
   - [ ] Handle missing bars gracefully
   - [ ] Support resampling to hourly/daily
   - [ ] Add warmup data loading from Redis cache or Databento

4. **Testing Data Hub**
   - [ ] Unit tests for Redis publisher/subscriber
   - [ ] Integration test: Databento → Redis → Feed
   - [ ] Test contract roll handling
   - [ ] Test reconnection after Redis restart
   - [ ] Load test: All 18 instruments streaming

**Acceptance Criteria:**
- ✅ Data hub runs as standalone process
- ✅ Publishes all 18 instruments to Redis successfully
- ✅ Test consumer can read bars from Redis
- ✅ Handles contract rolls correctly
- ✅ Graceful shutdown without data loss

**Deliverables:**
- `src/data_hub/data_hub_main.py`
- `src/data_hub/redis_client.py`
- `src/feeds/redis_feed.py`
- `tests/test_data_hub.py`

---

#### Phase 1.2: Strategy Interface & Abstraction (Days 3-4)

**Tasks:**

1. **Define BaseStrategy Interface** (`src/strategy/base_strategy.py`)
   - [ ] Create abstract base class:
     ```python
     class BaseStrategy(bt.Strategy):
         @abstractmethod
         def should_enter_long(self, symbol):
             pass

         @abstractmethod
         def should_enter_short(self, symbol):
             pass

         @abstractmethod
         def should_exit(self, symbol):
             pass

         def __init__(self):
             # Common initialization (feeds, coordinator, models)
             self.setup_feeds()
             self.setup_portfolio_coordinator()
             self.setup_ml_models()
     ```
   - [ ] Implement common lifecycle methods:
     - `setup_feeds()` - Multi-timeframe feed registration
     - `setup_portfolio_coordinator()` - Position tracking
     - `setup_ml_models()` - Load per-instrument models
     - `next()` - Main bar processing loop (calls abstract methods)
   - [ ] Add ML veto framework (shared across all strategies)
   - [ ] Add order/trade notification handlers

2. **Refactor Feature Engineering** (`src/features/`)
   - [ ] Move `feature_utils.py` to `src/features/indicators.py`
   - [ ] Move `filter_column.py` to `src/features/filter_state.py`
   - [ ] Create `src/features/__init__.py` with clean API:
     ```python
     from src.features import calculate_ibs, calculate_rsi, calculate_atr
     ```
   - [ ] Add unit tests for each indicator function
   - [ ] Ensure compatibility with both backtesting and live

3. **Create Strategy Factory** (`src/strategy/strategy_factory.py`)
   - [ ] Implement strategy loader:
     ```python
     def load_strategy(strategy_name: str, config: dict) -> BaseStrategy:
         if strategy_name == "ibs":
             return IbsStrategy(**config)
         elif strategy_name == "breakout":
             return BreakoutStrategy(**config)
         else:
             raise ValueError(f"Unknown strategy: {strategy_name}")
     ```
   - [ ] Support dynamic strategy registration
   - [ ] Validate strategy config against schema

4. **Strategy Configuration Schema** (`src/config/strategy_schema.py`)
   - [ ] Define configuration dataclasses:
     ```python
     @dataclass
     class StrategyConfig:
         name: str
         enabled: bool
         broker_account: str
         starting_cash: float
         models_path: str
         max_positions: int
         daily_stop_loss: float
         strategy_params: Dict[str, Any]
     ```
   - [ ] Add validation logic
   - [ ] Support per-instrument overrides

**Acceptance Criteria:**
- ✅ `BaseStrategy` defines clear interface
- ✅ Feature engineering library works independently
- ✅ Strategy factory can load strategies from config
- ✅ Configuration schema validates correctly

**Deliverables:**
- `src/strategy/base_strategy.py`
- `src/features/indicators.py`
- `src/features/filter_state.py`
- `src/strategy/strategy_factory.py`
- `src/config/strategy_schema.py`
- `tests/test_base_strategy.py`

---

#### Phase 1.3: Configuration Refactoring (Day 5)

**Tasks:**

1. **Create New Master Config Structure**
   - [ ] Design `config.multi_alpha.yml` template:
     ```yaml
     # Global settings
     databento:
       api_key: ${DATABENTO_API_KEY}
       dataset: GLBX.MDP3

     data_hub:
       redis_host: localhost
       redis_port: 6379
       publish_channels: ["market"]

     # Instrument configs (global across all strategies)
     instruments:
       ES:
         size: 1
         commission: 4.0
         multiplier: 50
         margin: 4000
       # ... all 18 instruments

     # Strategy-specific configs
     strategies:
       ibs:
         enabled: true
         broker_account: ${TRADERSPOST_IBS_WEBHOOK}
         starting_cash: 150000
         models_path: src/models/ibs/
         max_positions: 2
         daily_stop_loss: 2500
         instruments: [ES, NQ, RTY, YM, CL, NG, 6A, 6B, 6C, 6M, 6N, 6S, HG, SI, GC, ZC, ZS, ZW]
         strategy_params:
           max_bars_in_trade: 100
           # IBS-specific params

       breakout:
         enabled: false  # Start disabled
         broker_account: ${TRADERSPOST_BREAKOUT_WEBHOOK}
         starting_cash: 150000
         models_path: src/models/breakout/
         max_positions: 2
         daily_stop_loss: 2500
         instruments: [ES, NQ, RTY, YM, CL, NG, 6A, 6B, 6C, 6M, 6N, 6S, HG, SI, GC, ZC, ZS, ZW]
         strategy_params:
           # Breakout-specific params

     # Dashboard config
     dashboard:
       port: 5000
       strategies: [ibs, breakout]
     ```
   - [ ] Create migration guide from old `config.yml`

2. **Implement Config Loader** (`src/config/config_loader.py`)
   - [ ] Parse YAML with environment variable expansion
   - [ ] Validate against schema
   - [ ] Load global instrument configs
   - [ ] Load per-strategy configs
   - [ ] Support config overrides via CLI args

3. **Update Environment Variables**
   - [ ] Add to `.env.example`:
     ```
     DATABENTO_API_KEY=your_key
     TRADERSPOST_IBS_WEBHOOK=https://...
     TRADERSPOST_BREAKOUT_WEBHOOK=https://...
     REDIS_HOST=localhost
     REDIS_PORT=6379
     ```

**Acceptance Criteria:**
- ✅ New config structure supports multiple strategies
- ✅ Config loader validates and parses correctly
- ✅ Environment variables expand properly
- ✅ Migration path from old config documented

**Deliverables:**
- `config.multi_alpha.yml`
- `src/config/config_loader.py`
- `docs/CONFIG_MIGRATION.md`

---

#### Phase 1.4: Strategy Worker Entry Point (Days 6-7)

**Tasks:**

1. **Create Strategy Worker Main** (`src/runner/strategy_worker.py`)
   - [ ] Implement CLI argument parsing:
     ```bash
     python -m runner.strategy_worker --strategy ibs --config config.multi_alpha.yml
     ```
   - [ ] Load strategy-specific config
   - [ ] Connect to Redis data hub
   - [ ] Create Backtrader Cerebro instance
   - [ ] Attach Redis feeds (via `RedisLiveData`)
   - [ ] Load strategy from factory
   - [ ] Attach portfolio coordinator
   - [ ] Attach TradersPost client (strategy-specific webhook)
   - [ ] Run Cerebro event loop
   - [ ] Handle graceful shutdown

2. **Port Portfolio Coordinator** (minor refactoring)
   - [ ] Ensure thread-safety still works
   - [ ] Support per-strategy configuration (max_positions, stop_loss)
   - [ ] Add strategy name to logging

3. **Port TradersPost Client** (minor refactoring)
   - [ ] Support per-strategy webhook URLs
   - [ ] Add strategy name to payloads for debugging
   - [ ] Maintain existing retry logic

4. **Add Monitoring & Heartbeat**
   - [ ] Write per-strategy heartbeat files:
     - `/var/run/pine/ibs_worker_heartbeat.json`
     - `/var/run/pine/breakout_worker_heartbeat.json`
   - [ ] Include strategy-specific metrics
   - [ ] Add logging with strategy name prefix

**Acceptance Criteria:**
- ✅ Strategy worker starts successfully with `--strategy` flag
- ✅ Connects to Redis and receives market data
- ✅ Loads strategy-specific config correctly
- ✅ Routes orders to correct TradersPost webhook
- ✅ Writes heartbeat file with strategy metrics
- ✅ Graceful shutdown on SIGTERM

**Deliverables:**
- `src/runner/strategy_worker.py`
- `tests/test_strategy_worker.py`

---

### **Week 2: Migration - Refactor IBS as First Strategy**

**Goal:** Migrate existing IBS strategy to new multi-alpha architecture while keeping old system running in parallel.

#### Phase 2.1: IBS Strategy Refactoring (Days 8-10)

**Tasks:**

1. **Create IbsStrategy subclass** (`src/strategy/ibs_strategy_v2.py`)
   - [ ] Inherit from `BaseStrategy`
   - [ ] Port existing IBS logic from `ibs_strategy.py`:
     - Copy indicator calculations (or use `src/features/` library)
     - Copy entry/exit logic
     - Copy ML veto logic
   - [ ] Implement abstract methods:
     ```python
     def should_enter_long(self, symbol):
         # IBS entry logic
         ibs_hourly = self.get_ibs(symbol, 'hourly')
         ibs_daily = self.get_ibs(symbol, 'daily')
         prev_bar_pct = self.get_prev_bar_momentum(symbol)

         if ibs_hourly > 70 and ibs_daily > 40 and prev_bar_pct > threshold:
             # Check ML veto
             if self.ml_veto_allows(symbol, 'long'):
                 return True
         return False
     ```
   - [ ] Ensure feature calculations match exactly (use existing code)
   - [ ] Port ML model loading and veto logic
   - [ ] Add comprehensive logging

2. **Update ML Model Paths**
   - [ ] Move models to `src/models/ibs/`:
     ```
     src/models/ibs/
     ├── ES_best.json
     ├── ES_rf_model.pkl
     ├── NQ_best.json
     ├── NQ_rf_model.pkl
     └── ... (all 18 instruments)
     ```
   - [ ] Update model loader to use strategy-specific paths

3. **Testing IBS Refactor**
   - [ ] Unit tests for entry/exit logic
   - [ ] Compare indicator calculations to original
   - [ ] Test ML veto decisions match original
   - [ ] Backtest on historical data and compare to original results

**Acceptance Criteria:**
- ✅ IBS strategy extends `BaseStrategy`
- ✅ Entry/exit logic matches original exactly
- ✅ ML veto works identically to original
- ✅ Backtest results match original system (±1% P&L)

**Deliverables:**
- `src/strategy/ibs_strategy_v2.py`
- `tests/test_ibs_strategy_v2.py`
- Backtest comparison report

---

#### Phase 2.2: Integration Testing (Days 11-12)

**Tasks:**

1. **End-to-End Integration Tests**
   - [ ] Test: Data Hub → Redis → IBS Worker → TradersPost
   - [ ] Simulate market data stream
   - [ ] Verify orders sent to correct webhook
   - [ ] Verify position tracking
   - [ ] Verify P&L calculations
   - [ ] Test graceful shutdown and restart

2. **Create Test Harness** (`tests/integration/test_multi_alpha_e2e.py`)
   - [ ] Mock Databento stream
   - [ ] Mock Redis (or use real Redis in test mode)
   - [ ] Mock TradersPost webhook
   - [ ] Replay historical bars
   - [ ] Assert expected trades generated

3. **Performance Testing**
   - [ ] Measure latency: Databento tick → Redis → Strategy → Order
   - [ ] Target: <100ms end-to-end
   - [ ] Test with all 18 instruments streaming
   - [ ] Monitor memory usage over 24 hours

**Acceptance Criteria:**
- ✅ End-to-end integration test passes
- ✅ IBS worker generates expected trades on historical replay
- ✅ Latency <100ms for bar processing
- ✅ No memory leaks over 24-hour test

**Deliverables:**
- `tests/integration/test_multi_alpha_e2e.py`
- Performance test results document

---

#### Phase 2.3: Parallel Running Setup (Days 13-14)

**Tasks:**

1. **Deploy New System Alongside Old**
   - [ ] Set up Redis on production server
   - [ ] Start data hub in background
   - [ ] Configure IBS worker with **paper trading webhook** (separate from live)
   - [ ] Keep old `LiveWorker` system running unchanged

2. **Monitoring Setup**
   - [ ] Set up log aggregation for new system
   - [ ] Create comparison dashboard:
     - Old system P&L vs. New system P&L
     - Old system trades vs. New system trades
     - Latency comparison
   - [ ] Alert on discrepancies >5%

3. **Create Deployment Scripts**
   - [ ] `scripts/start_data_hub.sh` - Start data hub
   - [ ] `scripts/start_ibs_worker.sh` - Start IBS worker
   - [ ] `scripts/stop_all.sh` - Graceful shutdown
   - [ ] `scripts/restart_strategy.sh` - Restart single strategy

4. **Process Management with Supervisor**
   - [ ] Create `/etc/supervisor/conf.d/rooney_capital.conf`:
     ```ini
     [program:data_hub]
     command=/home/user/rooney-capital-v1/scripts/start_data_hub.sh
     autostart=true
     autorestart=true
     user=user
     stdout_logfile=/var/log/rooney/data_hub.log

     [program:ibs_worker]
     command=/home/user/rooney-capital-v1/scripts/start_ibs_worker.sh
     autostart=true
     autorestart=true
     user=user
     stdout_logfile=/var/log/rooney/ibs_worker.log
     ```

**Acceptance Criteria:**
- ✅ New system runs in parallel with old system
- ✅ Paper trading generates trades matching old system
- ✅ Monitoring dashboard shows comparison metrics
- ✅ Supervisor manages processes correctly

**Deliverables:**
- Deployment scripts
- Supervisor config
- Monitoring dashboard updates

---

### **Week 3: Validation & Production Cutover**

**Goal:** Validate new system performance, switch to production, decommission old system.

#### Phase 3.1: Validation Period (Days 15-17)

**Tasks:**

1. **Paper Trading Validation**
   - [ ] Run new IBS worker in paper mode for 3 days
   - [ ] Compare trades to old system:
     - Entry/exit times within ±1 bar
     - Position sizes match
     - P&L within ±2% (accounting for execution timing)
   - [ ] Log discrepancies and investigate

2. **Fix Discrepancies**
   - [ ] Debug any differences in trade signals
   - [ ] Verify indicator calculations match
   - [ ] Check ML veto decisions
   - [ ] Ensure contract roll handling matches

3. **Performance Validation**
   - [ ] Verify latency acceptable (<100ms)
   - [ ] Verify memory stable over 3 days
   - [ ] Verify no data loss or missing bars
   - [ ] Check Redis memory usage

**Acceptance Criteria:**
- ✅ 3 consecutive days of paper trading with <2% P&L difference
- ✅ No unexplained trade discrepancies
- ✅ System stability confirmed (no crashes, memory leaks)
- ✅ Performance metrics acceptable

**Deliverables:**
- Validation report comparing old vs. new system
- Bug fixes for any discrepancies

---

#### Phase 3.2: Production Cutover (Days 18-19)

**Tasks:**

1. **Pre-Cutover Preparation**
   - [ ] Create rollback plan
   - [ ] Backup current system state
   - [ ] Update TradersPost webhook to point to new IBS worker
   - [ ] Schedule cutover during low-volatility period

2. **Cutover Execution**
   - [ ] Stop old `LiveWorker` gracefully (close positions if open)
   - [ ] Update IBS worker config to use **live TradersPost webhook**
   - [ ] Restart IBS worker with live config
   - [ ] Monitor first 2 hours closely

3. **Post-Cutover Monitoring**
   - [ ] Verify trades executing correctly
   - [ ] Check order routing to TradersPost
   - [ ] Monitor position tracking
   - [ ] Verify P&L calculations
   - [ ] Watch for errors in logs

4. **Rollback Plan (if needed)**
   - [ ] If critical issues found:
     - Stop new IBS worker
     - Restart old `LiveWorker`
     - Revert TradersPost webhook
   - [ ] Debug issues and retry cutover

**Acceptance Criteria:**
- ✅ New system running in production successfully
- ✅ Trades executing correctly via TradersPost
- ✅ No critical errors in first 2 hours
- ✅ Old system successfully decommissioned (if cutover successful)

**Deliverables:**
- Cutover runbook
- Production deployment checklist
- Rollback procedure document

---

#### Phase 3.3: Cleanup & Documentation (Days 20-21)

**Tasks:**

1. **Code Cleanup**
   - [ ] Archive old `LiveWorker` code (don't delete yet)
   - [ ] Remove unused imports
   - [ ] Update README with new architecture
   - [ ] Clean up old config files

2. **Documentation**
   - [ ] Write `docs/MULTI_ALPHA_ARCHITECTURE.md`:
     - Architecture overview
     - Component responsibilities
     - Data flow diagrams
     - Deployment guide
   - [ ] Write `docs/ADDING_NEW_STRATEGY.md`:
     - Step-by-step guide for adding new strategies
     - Example strategy implementation
     - Testing checklist
   - [ ] Update `README.md` with new entry points

3. **Research Framework Updates**
   - [ ] Update `research/README.md` with new structure
   - [ ] Create `research/lib/` with shared backtesting code
   - [ ] Create `research/strategies/ibs/` directory structure
   - [ ] Document research workflow

**Acceptance Criteria:**
- ✅ Documentation complete and accurate
- ✅ Old code archived but preserved
- ✅ README reflects new architecture
- ✅ Research framework documented

**Deliverables:**
- `docs/MULTI_ALPHA_ARCHITECTURE.md`
- `docs/ADDING_NEW_STRATEGY.md`
- `research/README.md` updates
- Updated main `README.md`

---

### **Week 4: Second Strategy Implementation**

**Goal:** Implement second strategy to validate multi-alpha architecture and prove independent operation.

#### Phase 4.1: Strategy Design & Backtesting (Days 22-24)

**Tasks:**

1. **Define Second Strategy** (e.g., Breakout Strategy)
   - [ ] Review strategy logic document (from user)
   - [ ] Define entry rules
   - [ ] Define exit rules
   - [ ] Define filter/indicator requirements

2. **Implement Strategy Class** (`src/strategy/breakout_strategy.py`)
   - [ ] Extend `BaseStrategy`
   - [ ] Implement `should_enter_long()`
   - [ ] Implement `should_enter_short()`
   - [ ] Implement `should_exit()`
   - [ ] Use shared `src/features/` library
   - [ ] Add strategy-specific parameters

3. **Backtesting**
   - [ ] Create `research/strategies/breakout/`
   - [ ] Create `backtest_breakout.py` using shared harness
   - [ ] Run backtest on 2-year historical data
   - [ ] Collect trade data for ML training
   - [ ] Calculate performance metrics (Sharpe, drawdown, win rate)

4. **ML Model Training**
   - [ ] Extract features from backtest trades
   - [ ] Train per-instrument Random Forest models
   - [ ] Optimize thresholds using CPCV
   - [ ] Save models to `src/models/breakout/`
   - [ ] Validate on test set

5. **Portfolio Optimization**
   - [ ] Run greedy optimizer on per-instrument Sharpe ratios
   - [ ] Select best instruments to trade (target max drawdown)
   - [ ] Update strategy config with selected instruments

**Acceptance Criteria:**
- ✅ Breakout strategy implemented and tested
- ✅ Backtest shows positive Sharpe ratio
- ✅ ML models trained for all instruments
- ✅ Portfolio optimization selects <10 best instruments
- ✅ Strategy ready for paper trading

**Deliverables:**
- `src/strategy/breakout_strategy.py`
- `research/strategies/breakout/backtest_breakout.py`
- `src/models/breakout/{SYMBOL}_rf_model.pkl` (all instruments)
- Backtest performance report

---

#### Phase 4.2: Deployment & Validation (Days 25-26)

**Tasks:**

1. **Configuration**
   - [ ] Update `config.multi_alpha.yml`:
     ```yaml
     strategies:
       breakout:
         enabled: true
         broker_account: ${TRADERSPOST_BREAKOUT_WEBHOOK}
         starting_cash: 150000
         models_path: src/models/breakout/
         max_positions: 2
         daily_stop_loss: 2500
         instruments: [ES, NQ, RTY, YM, CL]  # Optimized subset
     ```
   - [ ] Set up second TradersPost webhook (paper trading initially)
   - [ ] Set up second Tradovate account ($150k)

2. **Deploy Breakout Worker**
   - [ ] Add to supervisor config:
     ```ini
     [program:breakout_worker]
     command=/home/user/rooney-capital-v1/scripts/start_breakout_worker.sh
     autostart=true
     autorestart=true
     ```
   - [ ] Start breakout worker in paper mode
   - [ ] Verify it connects to Redis data hub
   - [ ] Verify it receives market data
   - [ ] Monitor for trades

3. **Parallel Operation Testing**
   - [ ] Run IBS and Breakout workers simultaneously
   - [ ] Verify both receive same market data
   - [ ] Verify they operate independently (no interference)
   - [ ] Test: Stop one worker, other continues
   - [ ] Test: Restart one worker, other unaffected
   - [ ] Verify separate P&L tracking

4. **Dashboard Updates**
   - [ ] Add "Breakout" tab to dashboard
   - [ ] Show per-strategy metrics:
     - IBS: P&L, positions, trades, Sharpe
     - Breakout: P&L, positions, trades, Sharpe
   - [ ] Add combined metrics (total P&L, total positions)

**Acceptance Criteria:**
- ✅ Breakout worker runs in paper mode successfully
- ✅ IBS and Breakout run simultaneously without issues
- ✅ Both strategies operate independently
- ✅ Dashboard shows both strategies separately
- ✅ Stopping/restarting one doesn't affect the other

**Deliverables:**
- Deployed breakout worker
- Updated supervisor config
- Updated dashboard with two strategy tabs
- Test report showing independence

---

#### Phase 4.3: Production Launch & Monitoring (Days 27-28)

**Tasks:**

1. **Production Deployment**
   - [ ] Switch breakout worker to live TradersPost webhook
   - [ ] Monitor first trades closely
   - [ ] Verify order routing to correct account
   - [ ] Verify position tracking

2. **Multi-Strategy Monitoring**
   - [ ] Set up alerts for each strategy independently
   - [ ] Monitor combined position count across strategies
   - [ ] Track correlation between strategy P&L
   - [ ] Verify no unintended interactions

3. **Documentation Updates**
   - [ ] Document second strategy in architecture docs
   - [ ] Update runbook for managing multiple strategies
   - [ ] Create troubleshooting guide

4. **Final Validation**
   - [ ] Run both strategies in production for 2 days
   - [ ] Verify independence (can add/remove strategies without affecting others)
   - [ ] Verify data hub handles both strategies efficiently
   - [ ] Measure system resource usage (CPU, memory, Redis)

**Acceptance Criteria:**
- ✅ Both strategies running in production successfully
- ✅ Independent operation validated
- ✅ No performance degradation with 2 strategies
- ✅ System resources acceptable
- ✅ Documentation complete

**Deliverables:**
- Production deployment confirmation
- Multi-strategy monitoring dashboard
- Final architecture validation report
- Complete documentation set

---

## Risk Mitigation

### High-Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Data loss during Redis migration** | High | Low | Run old system in parallel during validation; Redis persistence enabled (AOF) |
| **Performance degradation with Redis** | Medium | Medium | Load test before production; Redis is in-memory (faster than current queue) |
| **Trade discrepancies between old/new IBS** | High | Medium | 3-day validation period; detailed comparison dashboard; rollback plan ready |
| **ML models not compatible with refactored code** | Medium | Low | Port model loading code carefully; test on historical data first |
| **Strategies interfere with each other** | High | Low | True process isolation; no shared state beyond data |
| **Redis single point of failure** | High | Medium | Redis persistence (AOF); monitoring with auto-restart; can fallback to direct Databento per strategy |

### Rollback Plan

If critical issues arise at any phase:

1. **Week 1-2**: No rollback needed (old system still primary)
2. **Week 3**: Revert TradersPost webhook to old system, restart old `LiveWorker`
3. **Week 4**: Disable new strategy in config, old IBS continues running

### Monitoring & Alerts

- [ ] Redis health checks (memory, connections, latency)
- [ ] Data hub heartbeat monitoring
- [ ] Per-strategy heartbeat monitoring
- [ ] Trade comparison alerts (new vs. old system during validation)
- [ ] Latency alerts (>200ms bar processing)
- [ ] Memory leak alerts (>10% growth per hour)

---

## Testing Strategy

### Unit Tests
- Data hub publish/subscribe logic
- Redis feed adapter
- Base strategy interface methods
- Feature engineering functions
- Strategy factory loading

### Integration Tests
- End-to-end: Databento → Redis → Strategy → TradersPost
- Multi-strategy parallel operation
- Contract roll handling
- Graceful shutdown and restart

### Performance Tests
- Latency: Tick → bar → signal → order (<100ms target)
- Throughput: 18 instruments streaming 1-min bars
- Memory: 24-hour stability test
- Redis: Pub/sub with multiple subscribers

### Validation Tests
- Backtest comparison: Old IBS vs. New IBS (±1% P&L)
- Paper trading: 3 days matching old system
- Signal comparison: Entry/exit decisions match

---

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Bar Processing Latency** | <100ms | Tick arrival → strategy.next() completion |
| **End-to-End Latency** | <200ms | Tick arrival → TradersPost webhook sent |
| **Memory Stability** | <5% growth/day | Monitor over 7 days |
| **Redis Throughput** | 18 symbols × 1 bar/min | Sustained load test |
| **Strategy Independence** | 100% | Stop/start/crash one, others unaffected |

### Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **IBS P&L Match** | ±2% | New system vs. old system during validation |
| **System Uptime** | >99.5% | During Week 4 (both strategies running) |
| **Time to Add Strategy** | <1 week | From idea → backtested → deployed (Week 4 proves this) |
| **Downtime During Migration** | 0 hours | Parallel running ensures no downtime |

### Architectural Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Shared Components** | Data hub, feature lib, backtest framework | ✅ Week 1 |
| **Independent Components** | Strategy logic, models, accounts, P&L | ✅ Week 2 |
| **Process Isolation** | Separate processes per strategy | ✅ Week 2 |
| **Configuration-Driven** | Add strategy via config, not code | ✅ Week 1 |
| **Zero-Coupling** | No inter-strategy communication | ✅ Week 2 |

---

## Post-Implementation: Continuous Improvement

### Month 2+
- [ ] Add third strategy to further validate architecture
- [ ] Optimize Redis performance (pipelining, compression)
- [ ] Add cross-strategy analytics (correlation, combined Sharpe)
- [ ] Implement automated strategy backtesting pipeline (CI/CD)
- [ ] Add strategy versioning (A/B test strategy improvements)

### Research Framework Enhancements
- [ ] Shared CPCV methodology implementation
- [ ] Shared hyperparameter optimization framework (Bayesian)
- [ ] Shared performance metrics calculation
- [ ] Strategy comparison tools (overlay equity curves, metrics table)

### Operational Enhancements
- [ ] Automated health checks for all components
- [ ] Slack/Discord notifications for each strategy separately
- [ ] Per-strategy performance reports (daily, weekly)
- [ ] Strategy capacity monitoring (when to stop adding capital)

---

## Appendix A: Key Code Locations

### New Components (to be created)

```
src/
├── data_hub/
│   ├── data_hub_main.py          # Data hub entry point
│   ├── redis_client.py           # Redis wrapper
│   └── __init__.py
│
├── feeds/
│   ├── redis_feed.py             # Redis → Backtrader adapter
│   └── __init__.py
│
├── strategy/
│   ├── base_strategy.py          # Abstract base class
│   ├── ibs_strategy_v2.py        # Refactored IBS
│   ├── breakout_strategy.py      # New strategy
│   ├── strategy_factory.py       # Strategy loader
│   └── __init__.py
│
├── features/                      # Shared feature engineering
│   ├── indicators.py             # Technical indicators
│   ├── filter_state.py           # Filter calculations
│   └── __init__.py
│
├── config/
│   ├── config_loader.py          # Multi-alpha config loader
│   ├── strategy_schema.py        # Config validation
│   └── __init__.py
│
└── models/
    ├── ibs/                       # IBS models
    │   ├── ES_rf_model.pkl
    │   └── ...
    └── breakout/                  # Breakout models
        ├── ES_rf_model.pkl
        └── ...

research/
├── lib/                           # Shared backtesting framework
│   ├── backtest_engine.py
│   ├── cpcv.py
│   ├── portfolio_optimizer.py
│   └── metrics.py
│
└── strategies/
    ├── ibs/
    │   ├── backtest_ibs.py
    │   ├── train_models.py
    │   └── results/
    └── breakout/
        ├── backtest_breakout.py
        ├── train_models.py
        └── results/
```

---

## Appendix B: Configuration Examples

### Data Hub Config

```yaml
# config.multi_alpha.yml

databento:
  api_key: ${DATABENTO_API_KEY}
  dataset: GLBX.MDP3

data_hub:
  redis_host: localhost
  redis_port: 6379
  publish_channels: ["market"]
  heartbeat_file: /var/run/pine/data_hub_heartbeat.json
  heartbeat_interval: 30

instruments:
  ES:
    databento_product_id: "ES.FUT"
    tradovate_symbol: "ES"
    size: 1
    commission: 4.0
    multiplier: 50
    margin: 4000
  NQ:
    databento_product_id: "NQ.FUT"
    tradovate_symbol: "NQ"
    size: 1
    commission: 4.0
    multiplier: 20
    margin: 4000
  # ... all 18 instruments

strategies:
  ibs:
    enabled: true
    broker_account: ${TRADERSPOST_IBS_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/ibs/
    max_positions: 2
    daily_stop_loss: 2500
    instruments: [ES, NQ, RTY, YM, CL, NG, 6A, 6B, 6C, 6M, 6N, 6S, HG, SI, GC, ZC, ZS, ZW]
    strategy_params:
      max_bars_in_trade: 100

  breakout:
    enabled: true
    broker_account: ${TRADERSPOST_BREAKOUT_WEBHOOK}
    starting_cash: 150000
    models_path: src/models/breakout/
    max_positions: 2
    daily_stop_loss: 2500
    instruments: [ES, NQ, RTY, YM, CL]
    strategy_params:
      breakout_lookback: 20

dashboard:
  port: 5000
  strategies: [ibs, breakout]
```

### Supervisor Config

```ini
# /etc/supervisor/conf.d/rooney_capital.conf

[program:data_hub]
command=/usr/bin/python -m src.data_hub.data_hub_main --config /home/user/rooney-capital-v1/config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/data_hub.log
stderr_logfile=/var/log/rooney/data_hub_error.log
environment=PATH="/home/user/.venv/bin"

[program:ibs_worker]
command=/usr/bin/python -m src.runner.strategy_worker --strategy ibs --config /home/user/rooney-capital-v1/config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/ibs_worker.log
stderr_logfile=/var/log/rooney/ibs_worker_error.log
environment=PATH="/home/user/.venv/bin"

[program:breakout_worker]
command=/usr/bin/python -m src.runner.strategy_worker --strategy breakout --config /home/user/rooney-capital-v1/config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/breakout_worker.log
stderr_logfile=/var/log/rooney/breakout_worker_error.log
environment=PATH="/home/user/.venv/bin"
```

---

## Appendix C: Timeline Summary

| Week | Phase | Deliverable | Status |
|------|-------|-------------|--------|
| **1** | Foundation | Data hub, BaseStrategy, Config, Worker | 🔲 Not Started |
| **2** | Migration | IBS refactored, Integration tests, Parallel running | 🔲 Not Started |
| **3** | Validation | Paper trading, Production cutover, Cleanup | 🔲 Not Started |
| **4** | Expansion | Second strategy, Deployment, Validation | 🔲 Not Started |

**Total Duration:** 28 days (4 weeks)
**End State:** 2+ strategies running independently with complete isolation

---

## Next Steps

1. **Review this roadmap** - Confirm timeline, milestones, and approach
2. **Clarify second strategy** - Share strategy logic document to inform design
3. **Kick off Week 1** - Begin data hub implementation
4. **Daily standups** - Track progress, blockers, and adjustments

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Author:** Claude (with Rooney Capital requirements)
