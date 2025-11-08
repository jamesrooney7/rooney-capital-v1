# Multi-Alpha Trading System - Setup Guide

This guide explains how to run the multi-alpha trading system with multiple independent strategies.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Hub Process                     │
│  - Connects to Databento                                │
│  - Aggregates ticks → 1min/hourly/daily bars            │
│  - Publishes to Redis pub/sub                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Redis Server    │
              │  (localhost:6379)│
              └─────────┬────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ IBS Worker    │ │ Breakout      │ │ Strategy N    │
│               │ │ Worker        │ │ Worker        │
│ Process 1     │ │ Process 2     │ │ Process N     │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Components

### 1. Data Hub
**Single process** that connects to Databento and publishes market data to Redis.

- **File**: `src/data_hub/data_hub_main.py`
- **Publishes**: Bars to Redis channels (`market:{symbol}:{timeframe}`)
- **Caches**: Latest bars for strategy worker warmup
- **Heartbeat**: `/var/run/pine/data_hub_heartbeat.json`

### 2. Strategy Workers
**Independent processes** that run trading strategies.

- **File**: `src/runner/strategy_worker.py`
- **Subscribes**: To Redis channels for market data
- **Executes**: Orders to TradersPost webhooks
- **Heartbeat**: `/var/run/pine/{strategy}_worker_heartbeat.json`

### 3. Redis Server
**In-memory data store** for pub/sub and caching.

- **Default**: localhost:6379
- **Channels**: `market:ES:1min`, `market:ES:hourly`, etc.
- **Cache**: `market:ES:1min:latest` for warmup

---

## Prerequisites

### 1. Install Redis
```bash
# Redis should already be installed
redis-server --version

# Start Redis
redis-server --daemonize yes --port 6379
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create `.env` file:
```bash
# Databento
DATABENTO_API_KEY=your_databento_key

# TradersPost webhooks (one per strategy)
TRADERSPOST_IBS_WEBHOOK=https://traderspost.io/webhooks/your_ibs_webhook
TRADERSPOST_BREAKOUT_WEBHOOK=https://traderspost.io/webhooks/your_breakout_webhook
```

Load environment variables:
```bash
source .env
# Or use: export $(cat .env | xargs)
```

---

## Configuration

### 1. Copy Example Config
```bash
cp config.multi_alpha.example.yml config.multi_alpha.yml
```

### 2. Edit Configuration

Edit `config.multi_alpha.yml`:

```yaml
# Databento settings
databento:
  api_key: ${DATABENTO_API_KEY}
  dataset: GLBX.MDP3

# Instrument configurations (global)
instruments:
  ES:
    size: 1
    commission: 4.0
    # ... etc

# Strategy configurations
strategies:
  ibs:
    enabled: true  # Enable/disable strategy
    broker_account: ${TRADERSPOST_IBS_WEBHOOK}
    starting_cash: 150000
    max_positions: 2
    daily_stop_loss: 2500
    instruments:  # Instruments this strategy will trade
      - ES
      - NQ
      - CL
    strategy_params:  # Strategy-specific parameters
      ibs_entry_high: 0.7
      use_ml_filter: true
```

---

## Running the System

### Option 1: Manual Start (Development)

**Terminal 1 - Start Data Hub:**
```bash
./scripts/start_data_hub.sh config.multi_alpha.yml
```

**Terminal 2 - Start IBS Strategy:**
```bash
./scripts/start_strategy_worker.sh ibs config.multi_alpha.yml
```

**Terminal 3 - Start Breakout Strategy (when ready):**
```bash
./scripts/start_strategy_worker.sh breakout config.multi_alpha.yml
```

### Option 2: Start All at Once (Testing)
```bash
./scripts/start_all.sh config.multi_alpha.yml
```

**Note**: This runs all processes in background. To stop:
```bash
# Find PIDs
ps aux | grep -E "data_hub|strategy_worker"

# Kill processes
kill <DATA_HUB_PID> <WORKER_PID_1> <WORKER_PID_2>
```

### Option 3: Supervisor (Production)

Install supervisor:
```bash
sudo apt-get install supervisor
```

Create supervisor config `/etc/supervisor/conf.d/rooney_capital.conf`:
```ini
[program:data_hub]
command=/home/user/rooney-capital-v1/scripts/start_data_hub.sh config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/data_hub.log
stderr_logfile=/var/log/rooney/data_hub_error.log

[program:ibs_worker]
command=/home/user/rooney-capital-v1/scripts/start_strategy_worker.sh ibs config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/ibs_worker.log
stderr_logfile=/var/log/rooney/ibs_worker_error.log

[program:breakout_worker]
command=/home/user/rooney-capital-v1/scripts/start_strategy_worker.sh breakout config.multi_alpha.yml
directory=/home/user/rooney-capital-v1
autostart=true
autorestart=true
user=user
stdout_logfile=/var/log/rooney/breakout_worker.log
stderr_logfile=/var/log/rooney/breakout_worker_error.log
```

Reload supervisor:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

Control processes:
```bash
sudo supervisorctl start data_hub
sudo supervisorctl start ibs_worker
sudo supervisorctl stop breakout_worker
sudo supervisorctl restart ibs_worker
```

---

## Monitoring

### 1. Check Heartbeat Files

```bash
# Data hub status
cat /var/run/pine/data_hub_heartbeat.json | jq .

# Strategy worker status
cat /var/run/pine/ibs_worker_heartbeat.json | jq .
cat /var/run/pine/breakout_worker_heartbeat.json | jq .
```

### 2. Monitor Redis

```bash
# Subscribe to see live bars
redis-cli
> SUBSCRIBE market:ES:1min
> SUBSCRIBE market:NQ:1min

# Check cached latest bars
> GET market:ES:1min:latest
> GET market:ES:hourly:latest
```

### 3. Check Logs

```bash
# Data hub logs
tail -f /var/log/rooney/data_hub.log

# Strategy worker logs
tail -f /var/log/rooney/ibs_worker.log
tail -f /var/log/rooney/breakout_worker.log
```

### 4. Monitor Processes

```bash
# Check if processes are running
ps aux | grep -E "data_hub|strategy_worker"

# Check Redis connections
redis-cli CLIENT LIST | grep -E "data_hub|strategy_worker"
```

---

## Testing

### 1. Test Redis Connection

```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# Test pub/sub
redis-cli
> SUBSCRIBE test
# In another terminal:
> PUBLISH test "hello"
```

### 2. Test Data Hub (without Databento)

Create a mock data hub test that publishes sample bars to Redis:

```python
# test_data_hub.py
from src.data_hub.redis_client import RedisClient
import time
from datetime import datetime

client = RedisClient()

# Publish test bar
bar_data = {
    'symbol': 'ES',
    'timestamp': datetime.utcnow().isoformat(),
    'open': 4500.0,
    'high': 4510.0,
    'low': 4495.0,
    'close': 4505.0,
    'volume': 1000.0
}

client.publish_and_cache('ES', '1min', bar_data)
print("Published test bar to market:ES:1min")
```

### 3. Test Strategy Worker (Paper Trading)

Set `broker_account` to empty string in config to prevent real orders:

```yaml
strategies:
  ibs:
    enabled: true
    broker_account: ""  # No orders will be sent
    starting_cash: 150000
```

---

## Troubleshooting

### Issue: "Could not connect to Redis"

**Solution:**
```bash
# Start Redis
redis-server --daemonize yes --port 6379

# Check if running
redis-cli ping
```

### Issue: "Strategy not found in config"

**Solution:**
- Check strategy name matches config file
- Ensure strategy is `enabled: true`
- Verify config file path is correct

### Issue: "Failed to load ML models"

**Solution:**
- Check `models_path` in config
- Ensure models exist: `src/models/ibs/ES_rf_model.pkl`
- Models are optional - strategy will run without them

### Issue: "No bars received from Redis"

**Solution:**
1. Check data hub is running: `ps aux | grep data_hub`
2. Check Redis channels: `redis-cli SUBSCRIBE market:ES:1min`
3. Check Databento connection in data hub logs
4. Verify `DATABENTO_API_KEY` is set

### Issue: "Orders not sent to TradersPost"

**Solution:**
1. Check webhook URL is set: `echo $TRADERSPOST_IBS_WEBHOOK`
2. Check worker logs for TradersPost errors
3. Verify webhook URL is correct in TradersPost dashboard
4. Test webhook manually with curl

---

## Adding a New Strategy

1. **Implement Strategy Class**

Create `src/strategy/my_strategy.py`:
```python
from src.strategy.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def should_enter_long(self):
        # Your logic here
        return self.data.close[0] > self.data.open[0]

    def should_enter_short(self):
        return self.data.close[0] < self.data.open[0]

    def should_exit(self):
        return self.bars_in_trade > 10
```

2. **Register Strategy**

Add to `src/strategy/strategy_factory.py`:
```python
from .my_strategy import MyStrategy
register_strategy("my_strategy", MyStrategy)
```

3. **Add to Config**

Add to `config.multi_alpha.yml`:
```yaml
strategies:
  my_strategy:
    enabled: true
    broker_account: ${TRADERSPOST_MY_STRATEGY_WEBHOOK}
    starting_cash: 150000
    instruments:
      - ES
      - NQ
```

4. **Start Worker**

```bash
./scripts/start_strategy_worker.sh my_strategy
```

---

## Directory Structure

```
rooney-capital-v1/
├── src/
│   ├── data_hub/          # Data hub (central data distribution)
│   │   ├── data_hub_main.py
│   │   └── redis_client.py
│   ├── feeds/             # Backtrader feed adapters
│   │   └── redis_feed.py
│   ├── strategy/          # Strategy implementations
│   │   ├── base_strategy.py
│   │   ├── ibs_strategy.py
│   │   └── strategy_factory.py
│   ├── features/          # Shared feature engineering
│   │   ├── indicators.py
│   │   └── filter_state.py
│   ├── config/            # Configuration system
│   │   ├── config_loader.py
│   │   └── strategy_schema.py
│   ├── runner/            # Strategy worker & coordination
│   │   ├── strategy_worker.py
│   │   ├── portfolio_coordinator.py
│   │   └── traderspost_client.py
│   └── models/            # ML models per strategy
│       ├── ibs/
│       └── breakout/
├── scripts/               # Startup scripts
│   ├── start_data_hub.sh
│   ├── start_strategy_worker.sh
│   └── start_all.sh
├── config.multi_alpha.yml # Main configuration
└── .env                   # Environment variables
```

---

## Next Steps (Week 2+)

1. **Week 2**: Migrate existing IBS strategy to new architecture
2. **Week 3**: Validation & production cutover
3. **Week 4**: Add second strategy (breakout)

For detailed roadmap, see `MULTI_ALPHA_REFACTORING_ROADMAP.md`
