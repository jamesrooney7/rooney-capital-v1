# Multi-Alpha Trading System Architecture

## Overview

The multi-alpha architecture refactors the monolithic `live_worker.py` into a **centralized data hub** + **independent strategy workers** design. This enables:

- **Multiple strategies running concurrently** (IBS A, IBS B, Breakout, etc.)
- **Isolated strategy workers** that can crash/restart independently
- **Centralized market data distribution** via Redis pub/sub
- **Better resource utilization** and horizontal scaling

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Databento Live API                        │
│              (GLBX.MDP3 - CME Futures Data)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Trade messages (live)
                           ▼
              ┌────────────────────────┐
              │      Data Hub          │
              │ (data_hub_main.py)     │
              │                        │
              │  - Subscribes to       │
              │    Databento live feed │
              │  - Aggregates trades   │
              │    into 1-min OHLCV    │
              │  - Publishes to Redis  │
              └────────────┬───────────┘
                           │
                           │ Publishes to Redis
                           ▼
              ┌────────────────────────┐
              │      Redis Pub/Sub     │
              │   localhost:6379       │
              │                        │
              │  Channels:             │
              │  - market:ES:1min      │
              │  - market:NQ:1min      │
              │  - market:6A:1min      │
              │  - ...                 │
              └────────────┬───────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
│  Strategy       │ │  Strategy   │ │  Strategy   │
│  Worker: IBS A  │ │  Worker:    │ │  Worker:    │
│                 │ │  IBS B      │ │  Breakout   │
│  - 6A, 6C, 6M   │ │             │ │             │
│  - CL, GC, HG   │ │  - 6B, 6N   │ │  - ES, NQ   │
│                 │ │  - 6S, SI   │ │  - RTY, YM  │
│  - Subscribes   │ │  - YM       │ │             │
│    to Redis     │ │             │ │             │
│  - Runs         │ │  - ML veto  │ │  - Trend    │
│    Backtrader   │ │  - IBS      │ │    breakout │
│  - ML models    │ │    signals  │ │    signals  │
│  - TradersPost  │ │             │ │             │
└─────────────────┘ └─────────────┘ └─────────────┘
```

---

## Components

### 1. Data Hub (`src/data_hub/data_hub_main.py`)

**Purpose**: Centralized market data ingestion and distribution

**Responsibilities**:
- Connects to Databento live API (GLBX.MDP3 dataset)
- Subscribes to all instruments across all strategies
- Aggregates trade messages into 1-minute OHLCV bars
- Publishes completed bars to Redis channels (`market:{SYMBOL}:1min`)
- Handles reconnection and error recovery
- Provides control channel for system-wide commands

**Key Features**:
- Single Databento connection (reduces API costs)
- Automatic reconnection with exponential backoff
- Bar aggregation with session-aware timestamps
- Redis pub/sub for distribution (no file I/O)

**Starting the Data Hub**:
```bash
export $(grep -v '^#' .env | xargs)
python3 -m src.data_hub.data_hub_main --config config.multi_alpha.yml
```

---

### 2. Strategy Workers (`src/runner/strategy_worker.py`)

**Purpose**: Independent strategy execution processes

**Responsibilities**:
- Subscribe to Redis channels for assigned instruments
- Load historical warmup data from Databento (for indicator initialization)
- Run Backtrader Cerebro with strategy-specific configuration
- Execute ML veto models for signal filtering
- Send orders/trades to TradersPost via webhooks
- Portfolio coordination (max positions, daily stop loss)
- Heartbeat monitoring

**Key Features**:
- **Isolated processes**: Each strategy runs independently
- **Dual-timeframe warmup**: Loads daily (250 bars) + hourly (338 bars) for each symbol
- **Redis live feeds**: Custom `RedisLiveData` and `RedisResampledData` feeds
- **ML model loading**: Automatic model loading per symbol from `models/` directory
- **Portfolio coordination**: Shared Redis-based position tracking
- **Graceful degradation**: Missing reference feeds (VIX, PL, 6E, 6J) handled with neutral defaults

**Starting a Strategy Worker**:
```bash
export $(grep -v '^#' .env | xargs)
python3 -m src.runner.strategy_worker --strategy "ibs_a" --config config.multi_alpha.yml
```

---

### 3. Redis Pub/Sub Layer

**Purpose**: High-performance message bus for market data

**Why Redis**:
- Sub-millisecond latency for bar distribution
- Native pub/sub pattern (no polling)
- Simple deployment (single-node sufficient)
- Automatic subscriber management
- No persistence needed (ephemeral data)

**Channel Naming Convention**:
- `market:{SYMBOL}:1min` - Live 1-minute bars from data hub
- `datahub:control` - System control messages (shutdown, reload, etc.)

**Data Format** (JSON):
```json
{
  "symbol": "ES",
  "timestamp": "2025-11-09T20:00:00",
  "open": 4500.25,
  "high": 4501.50,
  "low": 4499.75,
  "close": 4500.00,
  "volume": 1250
}
```

---

### 4. Configuration (`config.multi_alpha.yml`)

**Purpose**: Single source of truth for multi-strategy configuration

**Structure**:
```yaml
# Global settings
databento:
  api_key: ${DATABENTO_API_KEY}
  dataset: GLBX.MDP3

redis:
  host: localhost
  port: 6379

contract_map_path: Data/Databento_contract_map.yml
models_path: models

# Portfolio-level settings
portfolio:
  starting_cash: 250000
  max_positions: 2
  daily_stop_loss: 2500

# Strategy definitions
strategies:
  ibs_a:
    enabled: true
    strategy_type: ibs
    instruments:
      6A: {size: 1, commission: 4.5, ml_enabled: true}
      6C: {size: 1, commission: 4.5, ml_enabled: true}
      6M: {size: 1, commission: 4.5, ml_enabled: true}
      CL: {size: 1, commission: 4.5, ml_enabled: true}
      GC: {size: 1, commission: 4.5, ml_enabled: true}
      HG: {size: 1, commission: 4.5, ml_enabled: true}

  ibs_b:
    enabled: true
    strategy_type: ibs
    instruments:
      6B: {size: 1, commission: 4.5, ml_enabled: true}
      6N: {size: 1, commission: 4.5, ml_enabled: true}
      6S: {size: 1, commission: 4.5, ml_enabled: true}
      SI: {size: 1, commission: 4.5, ml_enabled: true}
      YM: {size: 1, commission: 4.5, ml_enabled: true}

# Instrument definitions (contract specs)
instruments:
  ES: {multiplier: 50, margin: 13200, commission: 4.0, databento_product_id: "ES.FUT"}
  NQ: {multiplier: 20, margin: 17600, commission: 4.0, databento_product_id: "NQ.FUT"}
  # ... all 19 instruments
```

---

## Data Flow

### 1. Market Data Flow (Live Trading)

```
Databento → Data Hub → Redis → Strategy Workers → Backtrader
```

1. **Databento**: Sends trade messages for subscribed symbols
2. **Data Hub**: Aggregates trades into 1-min bars, publishes to Redis
3. **Redis**: Distributes bars to all subscribed strategy workers
4. **Strategy Workers**: Feed bars into Backtrader via `RedisLiveData`
5. **Backtrader**: Processes bars, triggers strategy logic
6. **Strategy**: Generates signals, sends to TradersPost

### 2. Historical Warmup Flow (Startup)

```
Databento API → Strategy Worker → Backtrader Warmup
```

1. **Strategy Worker**: Requests historical data from Databento on startup
   - Daily bars: Last 250 trading days (for daily indicators)
   - Hourly bars: Last 15 days × 24h (for hourly indicators)
2. **Databento API**: Returns DBNStore objects with historical OHLCV
3. **Strategy Worker**: Converts to Backtrader Bar objects
4. **Strategy Worker**: Feeds warmup bars to Backtrader feeds via `extend_warmup()`
5. **Backtrader**: Replays warmup bars to initialize indicators
6. **Warmup Monitor Thread**: Watches for warmup completion, disables warmup mode
7. **Strategy**: Begins live processing with fully initialized indicators

---

## Key Differences from Legacy System

| Aspect | Legacy (`live_worker.py`) | Multi-Alpha (`strategy_worker.py`) |
|--------|---------------------------|-------------------------------------|
| **Architecture** | Monolithic single process | Data hub + multiple workers |
| **Strategies** | One strategy per worker | Multiple strategies concurrently |
| **Data Source** | Direct Databento connection | Redis pub/sub from data hub |
| **Scalability** | Vertical (bigger server) | Horizontal (add more workers) |
| **Isolation** | One crash kills everything | Workers crash independently |
| **Configuration** | Single `config.yml` per worker | Shared `config.multi_alpha.yml` |
| **Warmup Loading** | Built-in historical loader | Same pattern (Databento API) |
| **Feed Types** | `DatabentoLiveData` | `RedisLiveData`, `RedisResampledData` |
| **Deployment** | One systemd service | Multiple services (hub + workers) |

---

## Advantages of Multi-Alpha Architecture

### 1. **Strategy Isolation**
- Each strategy runs in its own process
- One strategy crash doesn't affect others
- Easy to disable/restart individual strategies

### 2. **Resource Efficiency**
- Single Databento connection (saves API costs)
- Shared Redis instance (minimal overhead)
- Workers can run on different servers if needed

### 3. **Development Velocity**
- Test new strategies without affecting production
- Deploy strategy updates without full system restart
- Debug individual strategies in isolation

### 4. **Operational Flexibility**
- Start/stop strategies independently
- Different strategies can have different update schedules
- Easy to add new strategies without code changes

### 5. **Monitoring & Debugging**
- Per-strategy logs and heartbeats
- Easier to identify which strategy has issues
- Independent performance metrics

---

## Migration Path

### Phase 1: Core Infrastructure ✅
- [x] Create data hub with Redis pub/sub
- [x] Build strategy worker framework
- [x] Implement Redis live feeds
- [x] Port configuration schema

### Phase 2: Strategy Workers ✅
- [x] Port IBS strategy to strategy worker
- [x] Implement warmup data loading
- [x] Add portfolio coordination
- [x] ML model loading per strategy

### Phase 3: Testing & Validation 🚧 IN PROGRESS
- [x] Fix warmup bar replay mechanism
- [x] Fix ZB hourly feed setup
- [ ] Validate IBS A worker startup
- [ ] Test end-to-end data flow
- [ ] Verify ML models work correctly

### Phase 4: Production Deployment ⏳ PENDING
- [ ] Add environment variable loading (dotenv)
- [ ] Systemd service files
- [ ] Monitoring and alerting
- [ ] Graceful shutdown handling
- [ ] Production validation

---

## Troubleshooting

### Common Issues

**1. "DATABENTO_API_KEY not found"**
- Environment variables not loaded
- Run: `export $(grep -v '^#' .env | xargs)` before starting services

**2. "Redis connection refused"**
- Redis not running
- Run: `sudo systemctl start redis-server`
- Check: `redis-cli ping` should return `PONG`

**3. "Warmup bars not replaying"**
- Missing warmup replay mechanism (fixed in commit e82ff38)
- Ensure `_set_strategies_warmup_mode()` is called after loading warmup data

**4. "IndexError in safe_div.py"**
- Indicators accessing arrays before warmup complete
- Fixed by warmup monitoring thread and `runonce=False`

**5. "Feed not found"**
- Missing feed setup for reference symbols
- Check that all required feeds (minute, hourly, daily) are created
- Special case: ZB uses `ZB_hour` and `TLT_day`

---

## Next Steps

1. **Add dotenv loading**: Automatically load `.env` file in data hub and strategy workers
2. **Systemd services**: Create service files for data hub and strategy workers
3. **Health checks**: Add HTTP endpoints for monitoring
4. **Graceful shutdown**: Improve signal handling for clean exits
5. **Documentation**: Complete API reference and operational runbooks

---

## References

- **Legacy System**: README.md, SYSTEM_GUIDE.md (describes single-worker architecture)
- **Configuration**: config.multi_alpha.yml (multi-strategy configuration)
- **Data Hub**: src/data_hub/data_hub_main.py
- **Strategy Worker**: src/runner/strategy_worker.py
- **Redis Feeds**: src/feeds/redis_feed.py
- **Contract Map**: Data/Databento_contract_map.yml
