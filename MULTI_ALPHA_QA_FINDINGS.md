# Multi-Alpha System - QA Findings & Required Fixes

**Date**: 2025-11-09
**Purpose**: Comprehensive comparison between legacy system (`live_worker.py`) and new multi-alpha architecture (`strategy_worker.py` + `data_hub_main.py`)

**Status**: 🔴 CRITICAL GAPS FOUND - Features missing from multi-alpha implementation

---

## Executive Summary

After systematic comparison between the legacy system (documented in README.md and SYSTEM_GUIDE.md) and our new multi-alpha architecture, I've identified **critical missing features** that were working in the legacy system but are absent from the new implementation.

### Missing Features (CRITICAL)

1. ❌ **Discord Notifications** - No alerts for trades, P&L, or system events
2. ❌ **Trade Database (SQLite)** - No persistence of trade history
3. ❌ **ML Feature Tracker** - Missing warmup feature tracking
4. ❌ **Dashboard Integration** - No real-time monitoring capability
5. ❌ **Auto-loading `.env` files** - Manual environment variable export required

### Configuration Gaps

6. ⚠️ **Discord webhook URL** not in multi_alpha config schema
7. ⚠️ **Heartbeat file path** configuration missing
8. ⚠️ **Historical warmup compression** settings missing
9. ⚠️ **Killswitch** configuration missing

### Good News

✅ Portfolio coordinator implementation looks correct
✅ TradersPost client implementation matches legacy
✅ ML model loading implemented correctly
✅ Warmup data loading fixed (recent commit)
✅ Contract map integration works

---

## Component-by-Component Analysis

### 1. Discord Notifications ❌ MISSING

**Legacy Implementation** (✅ Working):
- Location: `src/utils/discord_notifier.py`
- Integration: `src/runner/live_worker.py` lines 783-799
- Features:
  - System startup alerts
  - Trade entry notifications (symbol, side, price)
  - Trade exit notifications (P&L, duration, exit price)
  - Daily summaries (total P&L, win rate, profit factor, best/worst trades)
  - Emergency alerts (portfolio stop loss hit)
  - System status updates

**Multi-Alpha Implementation** (❌ Not Implemented):
- `src/runner/strategy_worker.py`: No Discord imports or initialization
- Missing all notification capabilities

**Impact**:
- **HIGH** - No visibility into live trading activity
- No alerts when positions open/close
- No daily performance summaries
- No emergency alerts for stop loss events

**Fix Required**:
```python
# In strategy_worker.py __init__:
from src.utils.discord_notifier import DiscordNotifier

self.discord_notifier = None
if self.strategy_config.discord_webhook_url:
    self.discord_notifier = DiscordNotifier(self.strategy_config.discord_webhook_url)
    self.discord_notifier.send_system_alert(
        title="Strategy Worker Started",
        message=f"{self.strategy_name} initialized with {len(self.strategy_config.instruments)} instruments"
    )
```

**Configuration Update Required**:
```yaml
# In config.multi_alpha.yml strategies section:
strategies:
  ibs_a:
    discord_webhook_url: ${DISCORD_WEBHOOK_URL}  # ADD THIS
```

---

### 2. Trade Database (SQLite) ❌ MISSING

**Legacy Implementation** (✅ Working):
- Location: `src/utils/trades_db.py`
- Integration: `src/runner/live_worker.py` line 2166-2168
- Database: `/opt/pine/runtime/trades.db`
- Features:
  - Persists all trade history (entry/exit times, P&L, prices)
  - Used by dashboard for real-time display
  - Enables portfolio analytics and reporting
  - Survives worker restarts

**Multi-Alpha Implementation** (❌ Not Implemented):
- No database integration in strategy_worker.py
- Trades only exist in memory during worker lifetime
- Dashboard cannot display trade history

**Impact**:
- **HIGH** - Loss of trade history on worker restart
- Dashboard unusable (no data source)
- Cannot reconcile positions after crash
- No audit trail for compliance
- Cannot calculate historical performance

**Fix Required**:
```python
# In strategy_worker.py __init__:
from src.utils.trades_db import TradesDB

self.trades_db = TradesDB()  # Uses default path /opt/pine/runtime/trades.db

# In trade notification callback:
def _on_trade(self, symbol: str, trade):
    """Callback when trade closes."""
    if trade.isclosed:
        self.trades_db.insert_trade(
            symbol=symbol,
            entry_time=trade.dtopen,
            entry_price=trade.price,
            exit_time=trade.dtclose,
            exit_price=trade.pnl / trade.size + trade.price,
            pnl=trade.pnl,
            size=trade.size,
            direction='LONG' if trade.long else 'SHORT'
        )
```

---

### 3. ML Feature Tracker ❌ MISSING

**Legacy Implementation** (✅ Working):
- Location: `src/runner/ml_feature_tracker.py`
- Integration: `src/runner/live_worker.py` line 46, 801-827
- Features:
  - Tracks ML feature readiness during warmup
  - Monitors which features are available per symbol
  - Coordinates feature collection across strategies
  - Heartbeat includes ML feature status

**Multi-Alpha Implementation** (❌ Not Implemented):
- No ML feature tracking in strategy_worker.py
- Cannot verify ML models have required features before trading

**Impact**:
- **MEDIUM** - ML models may make predictions without full feature set
- Reduced model accuracy during warmup phase
- No visibility into feature readiness in heartbeat

**Fix Required**:
```python
# In strategy_worker.py __init__:
from src.runner.ml_feature_tracker import MlFeatureTracker

self.ml_feature_tracker = MlFeatureTracker(
    symbols=list(self.strategy_config.instruments.keys())
)

# Pass to strategies via callback
```

---

### 4. Dashboard Integration ❌ BROKEN

**Legacy Implementation** (✅ Working):
- Location: `dashboard/app.py`
- Data Source: `/opt/pine/runtime/trades.db` (SQLite)
- Features:
  - Real-time open positions
  - Recent completed trades
  - Daily P&L summary
  - Win rate and profit factor
  - Best/worst trades

**Multi-Alpha Implementation** (❌ Broken):
- Dashboard code exists but has no data source
- Strategy worker doesn't write to trades.db
- Dashboard shows "No trades" or empty

**Impact**:
- **HIGH** - No real-time monitoring capability
- Cannot view current positions
- Cannot track performance during trading day

**Fix Required**:
- Implement trade database (see #2 above)
- Dashboard should work automatically once trades.db is populated

---

### 5. Environment Variable Auto-Loading ❌ MISSING

**Legacy Implementation** (✅ Working):
- Uses environment variables directly via `os.environ.get()`
- `.env` file exists but user must manually export

**Multi-Alpha Implementation** (❌ Manual Only):
- Same as legacy - requires manual export
- User must run: `export $(grep -v '^#' .env | xargs)`
- Documented in MULTI_ALPHA_QUICK_START.md as workaround

**Impact**:
- **LOW** - Operational friction for new users
- Error-prone (users forget to export)
- Not production-ready

**Fix Required**:
```python
# At top of data_hub_main.py and strategy_worker.py:
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
```

**Also update requirements.txt**:
```
python-dotenv>=1.0.0
```

---

## Configuration Schema Gaps

### 6. Discord Webhook URL ⚠️ MISSING FROM SCHEMA

**Legacy Config** (`config.yml`):
```yaml
discord_webhook_url: ${DISCORD_WEBHOOK_URL}
```

**Multi-Alpha Config** (`config.multi_alpha.yml`):
- ❌ Not present in schema
- ❌ Not in strategy-level config either
- ❌ Not in StrategyConfig dataclass

**Fix Required**:
Add to strategy schema in `src/config/strategy_schema.py`:
```python
discord_webhook_url: Optional[str] = None
```

And to global config for system-wide notifications.

---

### 7. Heartbeat Configuration ⚠️ PARTIALLY MISSING

**Legacy Config** (`config.yml`):
```yaml
heartbeat_interval: 30
heartbeat_file: /var/run/pine/worker_heartbeat.json
heartbeat_write_interval: 30
```

**Multi-Alpha Implementation**:
- ✅ Heartbeat writing implemented in strategy_worker.py
- ⚠️ Uses hardcoded values instead of config
- ⚠️ No heartbeat_file path in multi_alpha config

**Fix Required**:
Add to multi_alpha config:
```yaml
heartbeat:
  interval: 30
  file_path: /var/run/pine/worker_{strategy_name}_heartbeat.json
  write_interval: 30
```

---

### 8. Historical Warmup Compression ⚠️ MISSING

**Legacy Config** (`config.yml`):
```yaml
historical_warmup_compression: 1h  # Use 1h/1d to aggregate warmup bars
```

**Multi-Alpha Implementation**:
- ❌ Hardcoded to load daily (250 days) + hourly (15 days)
- ❌ No configuration option to change compression
- ❌ Cannot optimize warmup speed for strategies that don't need minute data

**Impact**:
- **LOW** - Current implementation works but not configurable
- Slower startup than necessary for daily-only strategies

**Fix Required**:
Add to strategy config schema:
```yaml
strategies:
  ibs_a:
    historical_warmup_compression: 1h  # Options: 1min, 1h, 1d
```

---

### 9. Killswitch Configuration ⚠️ MISSING

**Legacy Config** (`config.yml`):
```yaml
killswitch: ${POLICY_KILLSWITCH}
```

**Legacy Implementation**:
- Stops all trading when `POLICY_KILLSWITCH=true`
- Emergency stop mechanism

**Multi-Alpha Implementation**:
- ❌ Not implemented in strategy_worker.py
- ❌ No killswitch check before placing orders

**Impact**:
- **MEDIUM** - No emergency stop capability
- Must manually stop all workers individually

**Fix Required**:
```python
# In strategy_worker.py before placing orders:
if os.environ.get('POLICY_KILLSWITCH', 'false').lower() == 'true':
    logger.warning("KILLSWITCH ENABLED - No orders will be placed")
    return
```

---

## TradersPost Time-In-Force Discrepancy ⚠️ DOCUMENTATION ERROR

**SYSTEM_GUIDE.md Documentation**:
> "1-hour GTD (Good Till Date) time-in-force"

**Actual Code** (`src/runner/traderspost_client.py` line 319):
```python
"timeInForce": "day",  # Active until market close, then auto-cancels
```

**Finding**:
- Documentation is **incorrect**
- Code uses "day" not "GTD 1-hour"
- This is a **documentation bug**, not a code bug

**Fix Required**:
Update SYSTEM_GUIDE.md line 69 to say:
> "Day time-in-force (active until market close)"

---

## Good Implementations ✅

These components were correctly ported from legacy to multi-alpha:

### 1. Portfolio Coordinator ✅
- Correctly integrated in strategy_worker.py
- Max positions enforcement works
- Daily stop loss tracking works
- Redis-based state sharing implemented

### 2. TradersPost Client ✅
- Correctly integrated
- Order formatting matches legacy
- Retry logic preserved
- Metadata attachment works

### 3. ML Model Loading ✅
- Per-symbol model loading works
- Feature list extraction works
- Threshold configuration correct
- Graceful handling of missing models

### 4. Contract Map Integration ✅
- Correctly loads from Data/Databento_contract_map.yml
- Symbol resolution works
- Reference feed handling correct

### 5. Warmup Data Loading ✅
- Daily bars loading works (250 days)
- Hourly bars loading works (15 days)
- Databento API integration correct
- Memory cleanup implemented
- **Recent fix**: Warmup replay mechanism added (commit e82ff38)

### 6. Redis Feeds ✅
- RedisLiveData works correctly
- RedisResampledData works correctly
- Channel subscriptions correct
- Bar format matches Backtrader requirements

---

## Priority Fix List

### P0 (Critical - Blocks Production)
1. **Add Discord notifications** - No visibility into trades
2. **Add trade database** - No trade history persistence
3. **Add auto-loading `.env`** - Too error-prone without it

### P1 (High - Important for Operations)
4. **Add ML feature tracker** - Ensure model accuracy
5. **Add killswitch** - Emergency stop capability
6. **Add heartbeat configuration** - Standardize monitoring

### P2 (Medium - Nice to Have)
7. **Add warmup compression config** - Performance optimization
8. **Fix documentation** - Correct time-in-force description

### P3 (Low - Future Enhancement)
9. **Dashboard multi-worker support** - Show all strategies in one view
10. **Centralized logging** - Aggregate logs from all workers

---

## Testing Checklist

After implementing fixes, verify:

- [ ] Discord notifications sent for trade entry
- [ ] Discord notifications sent for trade exit
- [ ] Discord daily summary sent (first trade of new day)
- [ ] Discord emergency alert sent (stop loss hit)
- [ ] Trade database populated with all trades
- [ ] Dashboard shows real-time positions
- [ ] Dashboard shows completed trades
- [ ] Heartbeat file updates every 30 seconds
- [ ] Killswitch stops trading when enabled
- [ ] `.env` file auto-loaded on startup
- [ ] ML feature tracker shows feature readiness
- [ ] All 6A/6C/6M/CL/GC/HG ML models work

---

## Lessons Learned

### What Went Wrong

1. **Too much refactoring at once** - Changed architecture AND removed features
2. **Insufficient reference to legacy** - Didn't systematically port all features
3. **No feature inventory** - Didn't create checklist of legacy features before starting
4. **Documentation gap** - Multi-alpha docs didn't mention these features

### What Went Right

1. **Core architecture sound** - Data hub + workers separation is good
2. **Redis integration clean** - Pub/sub pattern works well
3. **Warmup mechanism fixed** - Systematic comparison with legacy solved it
4. **Configuration schema clean** - Multi-strategy config is better than legacy

### Recommendations

1. **Create feature parity checklist** before starting any refactor
2. **Port features first, optimize second** - Get it working, then make it better
3. **Document what changed** - Explicit "removed features" section
4. **Test end-to-end early** - Would have caught missing features sooner

---

## Implementation Plan

### Phase 1: Critical Fixes (This Week)

**Day 1**:
- [ ] Add python-dotenv to requirements.txt
- [ ] Add dotenv loading to data_hub_main.py
- [ ] Add dotenv loading to strategy_worker.py
- [ ] Test: Verify env vars load without manual export

**Day 2**:
- [ ] Add Discord integration to strategy_worker.py
- [ ] Update config schema for discord_webhook_url
- [ ] Test: Verify Discord notifications work

**Day 3**:
- [ ] Add trade database integration to strategy_worker.py
- [ ] Test: Verify trades persist to database
- [ ] Test: Verify dashboard shows trades

### Phase 2: High Priority (Next Week)

**Day 4**:
- [ ] Add ML feature tracker integration
- [ ] Test: Verify feature tracking in heartbeat

**Day 5**:
- [ ] Add killswitch functionality
- [ ] Add heartbeat configuration
- [ ] Test: Verify killswitch stops trading
- [ ] Test: Verify heartbeat path configurable

### Phase 3: Documentation (Following Week)

**Day 6**:
- [ ] Update MULTI_ALPHA_ARCHITECTURE.md with Discord/DB features
- [ ] Update MULTI_ALPHA_QUICK_START.md with setup instructions
- [ ] Fix time-in-force documentation error
- [ ] Create migration guide for missing features

---

## Comparison Summary Table

| Feature | Legacy Status | Multi-Alpha Status | Priority | Impact |
|---------|---------------|-------------------|----------|--------|
| Discord Notifications | ✅ Working | ❌ Missing | P0 | HIGH |
| Trade Database (SQLite) | ✅ Working | ❌ Missing | P0 | HIGH |
| Auto-load .env | ⚠️ Manual | ❌ Missing | P0 | LOW |
| ML Feature Tracker | ✅ Working | ❌ Missing | P1 | MEDIUM |
| Killswitch | ✅ Working | ❌ Missing | P1 | MEDIUM |
| Heartbeat Config | ✅ Configured | ⚠️ Hardcoded | P1 | LOW |
| Warmup Compression | ✅ Configured | ⚠️ Hardcoded | P2 | LOW |
| Portfolio Coordinator | ✅ Working | ✅ Working | - | - |
| TradersPost Client | ✅ Working | ✅ Working | - | - |
| ML Model Loading | ✅ Working | ✅ Working | - | - |
| Contract Map | ✅ Working | ✅ Working | - | - |
| Warmup Data Loading | ✅ Working | ✅ Fixed | - | - |
| Redis Feeds | N/A | ✅ New | - | - |
| Data Hub | N/A | ✅ New | - | - |

---

## Conclusion

The multi-alpha architecture is **sound** but **incomplete**. The data hub + strategy workers separation is a good design improvement over the monolithic legacy system. However, we removed several operational features during the refactor that need to be restored:

1. Discord notifications (critical for monitoring)
2. Trade database (critical for persistence)
3. ML feature tracker (important for model accuracy)
4. Killswitch (important for emergency stops)

Once these are added back, the multi-alpha system will have **feature parity** with the legacy system while providing better:
- Scalability (horizontal)
- Isolation (worker crashes don't affect others)
- Resource efficiency (single Databento connection)
- Development velocity (test strategies independently)

**Estimated time to feature parity**: 1 week (5 working days)

**Recommendation**: Implement P0 fixes immediately before production deployment.
