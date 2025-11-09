# Multi-Alpha Migration Status

This document tracks the progress of migrating from the legacy monolithic system (`live_worker.py`) to the new multi-alpha architecture (data hub + strategy workers).

---

## Overall Progress

```
████████████████████░░░░  75% Complete
```

**Status**: 🚧 Phase 3 - Testing & Validation

---

## Phase 1: Core Infrastructure ✅ COMPLETE

### Data Hub
- [x] Create `src/data_hub/data_hub_main.py`
- [x] Databento live feed subscription
- [x] Trade aggregation to 1-minute OHLCV bars
- [x] Redis pub/sub integration
- [x] Bar publishing to `market:{symbol}:1min` channels
- [x] Reconnection and error handling
- [x] Control channel for system commands

### Redis Integration
- [x] Redis client wrapper (`src/data_hub/redis_client.py`)
- [x] Pub/sub message formatting (JSON)
- [x] Channel naming conventions
- [x] Redis feeds for Backtrader (`src/feeds/redis_feed.py`)
  - [x] `RedisLiveData` (minute bars)
  - [x] `RedisResampledData` (hourly/daily aggregation)

### Configuration Schema
- [x] Multi-strategy configuration format (`config.multi_alpha.yml`)
- [x] Strategy definitions with instrument assignments
- [x] Portfolio-level settings
- [x] Instrument specifications
- [x] Configuration loader (`src/config/config_loader.py`)
- [x] Strategy schema validation (`src/config/strategy_schema.py`)

---

## Phase 2: Strategy Workers ✅ COMPLETE

### Strategy Worker Framework
- [x] Create `src/runner/strategy_worker.py`
- [x] Command-line interface (--strategy, --config)
- [x] Strategy-specific configuration loading
- [x] Cerebro initialization
- [x] Feed setup (minute, hourly, daily)
- [x] Strategy instantiation with parameters
- [x] Broker configuration

### Portfolio Coordination
- [x] `PortfolioCoordinator` class
- [x] Max positions limit enforcement
- [x] Daily stop loss tracking
- [x] Redis-based state sharing
- [x] Position counting across workers

### ML Model Integration
- [x] Per-symbol model loading
- [x] Feature list extraction
- [x] Threshold configuration
- [x] Model bundle validation
- [x] Graceful handling of missing models

### Reference Feeds
- [x] Market index feeds (ES, NQ, RTY, YM)
- [x] Commodity feeds (NG, SI, ZC, ZS, ZW)
- [x] Currency cross feeds (6B, 6N, 6S)
- [x] ZB as TLT_day mapping
- [x] Optional feed handling (VIX, PL, 6E, 6J)

### Warmup Data Loading
- [x] Databento historical API integration
- [x] Contract map resolution
- [x] Daily bars loading (250 days)
- [x] Hourly bars loading (15 days)
- [x] Bar conversion to Backtrader format
- [x] Batched loading with capacity management
- [x] Memory cleanup after each symbol

---

## Phase 3: Testing & Validation 🚧 IN PROGRESS (75%)

### Warmup Replay Mechanism ✅ FIXED
- [x] **Issue**: IndexError in safe_div.py due to indicators accessing arrays before warmup
- [x] **Root Cause**: Warmup bars loaded but not replayed properly through Backtrader
- [x] **Fix**: Added warmup replay mechanism from legacy system
  - [x] `_set_strategies_warmup_mode()` - Enable/disable fast warmup on strategies
  - [x] `_monitor_warmup_drain()` - Background thread monitors warmup completion
  - [x] `cerebro.run(runonce=False)` - Proper bar processing mode
  - [x] Automatic warmup mode disabling after bars drained
- [x] **Commit**: e82ff38 "Add warmup bar replay mechanism and fix ZB hourly feed"

### ZB Hourly Feed ✅ FIXED
- [x] **Issue**: Warning "ZB: feed 'ZB_hour' not found in self.data_feeds"
- [x] **Fix**: Added ZB_hour feed (60-min resampling from ZB_minute)
- [x] **Commit**: e82ff38 (same commit as warmup replay fix)

### Environment Variable Loading ⚠️ PENDING
- [ ] **Issue**: `.env` file exists but not automatically loaded
- [ ] **Workaround**: Manually export variables: `export $(grep -v '^#' .env | xargs)`
- [ ] **Required**: Add python-dotenv integration
  - [ ] Install `python-dotenv` in requirements.txt
  - [ ] Add `load_dotenv()` to data_hub_main.py
  - [ ] Add `load_dotenv()` to strategy_worker.py
  - [ ] Document in MULTI_ALPHA_QUICK_START.md

### End-to-End Testing ⏳ PENDING
- [ ] Start data hub successfully
- [ ] Verify Redis bar publishing
- [ ] Start IBS A worker successfully
- [ ] Verify warmup data loading
- [ ] Verify warmup replay completes
- [ ] Verify live bars flow to strategy
- [ ] Verify ML models make predictions
- [ ] Verify orders sent to TradersPost (paper trading)
- [ ] Run for 1 full trading session
- [ ] Verify position tracking
- [ ] Verify daily stop loss enforcement

### Multi-Worker Testing ⏳ PENDING
- [ ] Start IBS A and IBS B workers concurrently
- [ ] Verify portfolio coordination works
- [ ] Verify max positions enforced across workers
- [ ] Test worker crash/restart independence
- [ ] Verify no race conditions

---

## Phase 4: Production Deployment ⏳ PENDING (0%)

### Systemd Integration
- [ ] Create `pine-datahub.service`
- [ ] Create `pine-worker@.service` (templated)
- [ ] Add service dependencies (Redis, network)
- [ ] Configure restart policies
- [ ] Add resource limits
- [ ] Document service management

### Monitoring
- [ ] Heartbeat file per worker
- [ ] Health check endpoints (HTTP)
- [ ] Alerting on worker failures
- [ ] Alerting on data gaps
- [ ] Daily summary reports
- [ ] Position reconciliation checks

### Graceful Shutdown
- [ ] SIGTERM handling in data hub
- [ ] SIGTERM handling in strategy workers
- [ ] Close all Redis connections
- [ ] Flush pending orders
- [ ] Write final heartbeat
- [ ] Clean shutdown within 30s timeout

### Documentation
- [ ] API reference documentation
- [ ] Operational runbooks
- [ ] Troubleshooting guide (expanded)
- [ ] Performance tuning guide
- [ ] Security hardening guide
- [ ] Disaster recovery procedures

---

## Known Issues & Workarounds

### 1. Environment Variables Not Auto-Loading

**Issue**: `.env` file exists but variables not loaded automatically

**Impact**: Data hub fails with "CRAM authentication failed" due to empty API key

**Workaround**:
```bash
export $(grep -v '^#' .env | xargs)
```

**Permanent Fix**: Add python-dotenv to requirements.txt and call `load_dotenv()` at startup

**Priority**: HIGH (blocks first-time users)

---

### 2. Redis Connection Warnings

**Issue**: DeprecationWarning about `retry_on_timeout` parameter

**Impact**: Cosmetic warning in logs, no functional impact

**Workaround**: Ignore warning for now

**Permanent Fix**: Update Redis client initialization to remove deprecated parameter

**Priority**: LOW (cosmetic only)

---

### 3. Scikit-learn Version Mismatch

**Issue**: ML models trained with sklearn 1.7.1, server has 1.7.2

**Impact**: Warning about potential breaking changes (but works in practice)

**Workaround**: Ignore warning if models work correctly

**Permanent Fix**: Retrain models with current sklearn version OR pin sklearn==1.7.1

**Priority**: MEDIUM (potential risk)

---

### 4. VIX Feed Unavailable

**Issue**: VIX data not in Databento contract map

**Impact**: Strategy uses neutral defaults (0.0 for VIX z-scores)

**Workaround**: This is expected behavior (VIX handled gracefully)

**Permanent Fix**: Add VIX.IND to contract map if needed

**Priority**: LOW (optional enhancement)

---

## Testing Checklist

### Pre-Deployment Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Paper trading for 1 week
- [ ] All ML models load successfully
- [ ] All symbols receive data
- [ ] Warmup completes without errors
- [ ] Orders reach TradersPost
- [ ] Position tracking accurate
- [ ] Daily stop loss works
- [ ] Max positions enforced

### Production Validation

- [ ] Small position sizes (1 contract)
- [ ] Monitor for 1 week
- [ ] Compare to legacy system performance
- [ ] Verify fills match signals
- [ ] Check logs daily
- [ ] Reconcile positions daily
- [ ] Gradually increase position sizes

---

## Lessons Learned

### 1. Variable Name Collisions
**Problem**: Loop variable `start` overwrote datetime variable `start`
**Solution**: Use descriptive names like `batch_start` for loops
**Prevention**: Code review for shadowing variables

### 2. Databento Schema Names
**Problem**: Used `ohlcv-1min` but correct is `ohlcv-1m`
**Solution**: Reference Databento docs for exact schema names
**Prevention**: Schema name constants in code

### 3. DBNStore Object Handling
**Problem**: Assumed DBNStore supports `len()` but it doesn't
**Solution**: Just check `if data is not None`
**Prevention**: Read API docs before assuming standard methods

### 4. Warmup Replay Requirements
**Problem**: Warmup bars loaded but not replayed through Backtrader
**Solution**: Need explicit warmup mode + monitoring thread
**Prevention**: Compare to working legacy implementation systematically

### 5. Contract Map API
**Problem**: Used `ContractMap.load()` (doesn't exist) instead of `load_contract_map()`
**Solution**: Use the function, not a class method
**Prevention**: Check actual API in code before using

---

## Next Immediate Steps

1. **Add dotenv loading** (fixes environment variable issue)
2. **Test IBS A worker end-to-end** (verify all fixes work together)
3. **Document remaining issues** found during testing
4. **Create systemd service files** (for production deployment)
5. **Write operational runbooks** (start/stop/monitor procedures)

---

## Migration Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Core Infrastructure | 2 weeks | ✅ Complete |
| Phase 2: Strategy Workers | 3 weeks | ✅ Complete |
| Phase 3: Testing & Validation | 1 week | 🚧 In Progress (75%) |
| Phase 4: Production Deployment | 1 week | ⏳ Pending |

**Estimated Completion**: 1 week remaining

---

## References

- **Architecture**: MULTI_ALPHA_ARCHITECTURE.md
- **Quick Start**: MULTI_ALPHA_QUICK_START.md
- **Legacy System**: README.md, SYSTEM_GUIDE.md
- **Configuration**: config.multi_alpha.yml
- **Contract Map**: Data/Databento_contract_map.yml
