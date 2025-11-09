# Multi-Alpha Architecture vs Legacy System

## Data Flow Comparison

### Legacy System (databento_bridge.py + live_worker.py)

**Live Data Flow:**
1. `DatabentoSubscriber` connects to Databento live gateway
2. Receives trades, aggregates into 1-minute bars
3. Publishes `Bar` objects to `QueueFanout` (per-symbol queues)
4. `DatabentoLiveData` reads from queues
5. `ResampledLiveData` aggregates minute bars with **SESSION-AWARE** boundaries:
   - Daily bars: Close at 23:00 UTC (CME session end)
   - Hourly bars: Close at top of hour (minute == 0)
   - Uses `session_end_hour=23` parameter

**Warmup Data Flow:**
1. Load historical data from Databento Historical API
2. `_convert_databento_to_bt_bars()` converts to pandas DataFrame
3. Resamples to target compression
4. Converts to `Bar` objects
5. Calls `extend_warmup()` on feeds
6. Feeds drain warmup bars before live data

### New Multi-Alpha System

**Live Data Flow:**
1. `DataHub` connects to Databento live gateway
2. Receives trades, aggregates into 1-min/hourly/daily bars
3. Publishes JSON to Redis pub/sub channels
4. `RedisLiveData` subscribes to Redis channels
5. `RedisResampledData` wraps `RedisLiveData` (no aggregation)

**Warmup Data Flow:**
1. Load historical data from Databento Historical API
2. `_convert_databento_to_bars()` converts to pandas DataFrame
3. Resamples to target compression
4. Converts to `Bar` objects
5. Calls `extend_warmup()` on feeds
6. Feeds drain warmup bars before live data

## CRITICAL ISSUES

### Issue 1: Session Boundaries Not Respected ⚠️

**Problem:**
- Data hub uses calendar day boundaries (00:00 UTC)
- Legacy system uses session boundaries (23:00 UTC)
- Daily bars will have different timestamps and OHLC values

**Location:** `src/data_hub/data_hub_main.py` line 120
```python
# WRONG: Uses calendar day
day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

# SHOULD BE: Use session day (23:00 previous day)
```

**Impact:** **CRITICAL** - Strategies rely on correct daily bar boundaries!

### Issue 2: Warmup Resampling Doesn't Match Live

**Problem:**
- Warmup uses pandas `.resample("1d")` which uses calendar day
- Live data (should) use session day boundaries
- Warmup bars won't match live bars

**Location:** `src/runner/strategy_worker.py` line 452
```python
# Uses simple pandas resample without session awareness
aggregation = ohlcv.resample(compression).agg(...)
```

**Impact:** **MAJOR** - Warmup indicators will be slightly different from live

### Issue 3: No Source Feed Aggregation

**Problem:**
- Legacy `ResampledLiveData` aggregates minute bars from source feed
- New `RedisResampledData` just subscribes to pre-aggregated channel
- Not a true "resampler"

**Impact:** **MINOR** - Architecturally OK, but naming is misleading

## REQUIRED FIXES

### Fix 1: Data Hub Session Boundaries

Update `BarAggregator` to use session-aware daily bars:

```python
def apply_trade(self, symbol, price, size, timestamp, ...):
    # Minute bar - unchanged
    minute = timestamp.replace(second=0, microsecond=0)

    # Hourly bar - unchanged
    hour = timestamp.replace(minute=0, second=0, microsecond=0)

    # Daily bar - USE SESSION BOUNDARIES
    # Session ends at 23:00 UTC, so day starts at 23:00 previous day
    if timestamp.hour >= 23:
        # After 23:00, belongs to next session day
        day = timestamp.replace(hour=23, minute=0, second=0, microsecond=0)
    else:
        # Before 23:00, belongs to current session day (started yesterday 23:00)
        day = (timestamp - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
```

### Fix 2: Warmup Resampling with Session Boundaries

Update `_convert_databento_to_bars()` to resample with session awareness:

```python
# For daily resampling, use session boundaries
if compression == "1d":
    # Resample with offset to align to 23:00 UTC
    aggregation = ohlcv.resample("1D", offset="23H").agg(...)
```

### Fix 3: Document Architectural Difference

Add comments explaining that data hub pre-aggregates (unlike legacy queue-based system).

## TESTING CHECKLIST

- [ ] Data hub emits daily bars at correct session boundaries
- [ ] Warmup daily bars match live daily bars
- [ ] Hourly bars align correctly
- [ ] Strategy indicators initialize with correct values
- [ ] No duplicate or missing bars during session transitions

## COMPATIBILITY

**Breaking Changes:** Yes - daily bar timestamps will change from 00:00 to 23:00 UTC

**Migration:** Existing backtests/strategies need to be aware of session boundary change
