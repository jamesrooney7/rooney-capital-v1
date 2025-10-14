# Rooney Capital Trading Strategy v1.0

Automated futures trading system using the Internal Bar Strength (IBS) edge with
machine-learning veto models and live order routing through TradersPost.

## Table of Contents
- [System Overview](#system-overview)
  - [End-to-end Flow](#end-to-end-flow)
  - [Supported Markets](#supported-markets)
- [Repository Layout](#repository-layout)
- [Core Modules](#core-modules)
  - [`src/config.py`](#srcconfigpy)
  - [`src/models`](#srcmodels)
  - [`src/strategy`](#srcstrategy)
  - [`src/runner`](#srcrunner)
- [Machine Learning Artefacts](#machine-learning-artefacts)
- [Setup](#setup)
- [Runtime Configuration](#runtime-configuration)
  - [Required Fields](#required-fields)
  - [Sample Configuration](#sample-configuration)
- [Launching a Worker](#launching-a-worker)
  - [Local/Paper Trading](#localpaper-trading)
  - [Production via systemd](#production-via-systemd)
- [Operations & Safety](#operations--safety)
- [Additional References](#additional-references)

## System Overview

The Rooney Capital stack ingests live CME Globex data from Databento, evaluates
multi-asset IBS signals inside Backtrader, and routes execution events to
TradersPost for Tradovate brokerage accounts. Each futures root has a dedicated
Random Forest model that can veto discretionary signals when confidence falls
below a trained threshold.

### End-to-end Flow

1. **Contract governance** – `Data/` YAML files and helper utilities in
   `src/runner/contract_map.py` define Databento datasets, roll rules, and
   TradersPost metadata for each futures root.
2. **Live market data ingestion** – `src/runner/databento_bridge.py` subscribes
   to the Databento live gateway, aggregates ticks into one-minute bars, and
   pushes them into per-symbol queues that behave like Backtrader feeds.
3. **Strategy orchestration** – `src/runner/live_worker.py` creates the
   Backtrader `Cerebro` instance, wires a `DatabentoLiveData` feed per symbol,
   and attaches `IbsStrategy` with instrument-specific overrides.
4. **ML veto models** – `src/models/loader.py` loads instrument bundles so the
   strategy can filter trades based on ML probability scores.
5. **Trading logic** – `src/strategy/ibs_strategy.py` implements signal
   generation, position management, and portfolio-level risk controls.
6. **Execution webhook** – `src/runner/traderspost_client.py` serialises order
   and trade notifications and posts them to TradersPost with retry logic.
7. **Configuration & utilities** – `src/config.py` and helpers in
   `src/strategy/feature_utils.py` expose shared constants and indicator
   calculations used across the stack.

### Supported Markets

The default deployment tracks the following roots: `ES`, `NQ`, `RTY`, `YM`,
`GC`, `SI`, `HG`, `CL`, `NG`, `6A`, `6B`, and `6E`. Extend the universe by
updating the contract map and provisioning ML bundles for the new symbols.

## Repository Layout

```
src/
├── strategy/     # Trading logic, feature engineering, risk controls
├── runner/       # Databento bridge, live worker, TradersPost client
└── models/       # Instrument-specific Random Forest veto bundles
Data/             # Contract metadata and roll rules
docs/             # Deployment notes and historical planning docs
requirements.txt  # Python dependencies
```

Additional context:

- **Data/** – Static metadata (`Databento_contract_map.yml`) that drive
  subscription choices and audit rolls. Reference subscriptions live inside the
  `reference_feeds` block of this contract map.
- **docs/** – Includes the deployment roadmap and background notes preserved for
  operational reference.

## Core Modules

### `src/config.py`
Centralises runtime constants so workers and backtests remain in sync. Notable
exports include:

- `COMMISSION_PER_SIDE` – Default commission per side with environment overrides.
- `PAIR_MAP` – Cross-instrument relationships used for pair-based filters.
- `REQUIRED_REFERENCE_FEEDS` – Guard ensuring daily reference data is loaded
  before the strategy initialises.

### `src/models`
Holds Random Forest artefacts (`*_rf_model.pkl`) and optimisation metadata
(`*_best.json`) for each supported symbol. The loader (`loader.py`):

- Hydrates the scikit-learn model, feature list, and veto probability threshold.
- Normalises filesystem paths so workers can resolve artefacts locally or from a
  mounted models directory.
- Provides `strategy_kwargs_from_bundle` to translate a bundle into keyword
  arguments consumed by `IbsStrategy` (including the ML veto hook).

### `src/strategy`

- **`ibs_strategy.py`** – Core Backtrader strategy implementing the IBS system.
  It consumes intraday and reference data feeds per symbol, computes indicators
  via `feature_utils.py`, `filter_column.py`, and `safe_div.py`, and applies
  multiple risk controls (ML veto, pair IBS thresholds, time-of-day gating,
  cooldowns). Notification hooks forward order/trade events to the live worker.
- **`feature_utils.py`, `filter_column.py`, `safe_div.py`** – Encapsulate shared
  indicator calculations, safe math primitives, and column filtering used during
  feature selection.
- **`contract_specs.py`** – Reference data for contract multipliers, tick sizes,
  and leverage that inform sizing logic.

### `src/runner`

- **`contract_map.py`** – Parses `Data/Databento_contract_map.yml`, validates the
  schema, and exposes helpers for Databento subscriptions, TradersPost metadata,
  and reference feed lookups.
- **`databento_bridge.py`** – Implements the live market data bridge. A
  `DatabentoSubscriber` normalises `TradeMsg` streams into one-minute OHLCV bars
  which `DatabentoLiveData` adapts to Backtrader data feeds with heartbeat and
  shutdown handling.
- **`traderspost_client.py`** – Formats order/trade notifications into
  TradersPost-compatible payloads with exponential backoff retries and metadata
  attachments for audit trails.
- **`live_worker.py`** – Orchestrates ingestion, ML bundle loading, strategy
  attachment, and the lifecycle of the Backtrader engine. Provides both blocking
  (`run`) and asyncio (`run_async`) entry points so workers can be embedded under
  systemd or integration tests.

## Machine Learning Artefacts

Each instrument has a dedicated Random Forest model trained on
instrument-specific features (e.g. `ES_rf_model.pkl`, `NQ_rf_model.pkl`). Use
`src.models.load_model_bundle` to hydrate the trained estimator and associated
strategy kwargs:

```python
from models import load_model_bundle
from strategy.ibs_strategy import IbsStrategy

bundle = load_model_bundle("ES")
cerebro.addstrategy(IbsStrategy, **bundle.strategy_kwargs())
```

> **Note:** Model artefacts are stored with Git LFS. Run `git lfs pull` after
> cloning to ensure large files are available before attempting to load them.

## Setup

1. **Clone** the repository and install Git LFS if it is not already present:
   `git lfs install` followed by `git lfs pull`.
2. **Create a virtual environment** (Python 3.10+) and install dependencies:
   `pip install -r requirements.txt`.
3. **Populate data metadata** as needed in `Data/Databento_contract_map.yml` so
   the worker can subscribe to the desired products. Reference subscriptions are
   defined via the `reference_feeds` block inside this file.
4. **Ensure credentials** (Databento API key, TradersPost webhook URL) are
   available via environment variables or the runtime configuration file.

## Runtime Configuration

Workers load their runtime configuration from a JSON or YAML file. Provide the
path explicitly to `load_runtime_config()` or set the `PINE_RUNTIME_CONFIG`
environment variable before starting the process.

### Required Fields

- `contract_map` – Path to the Databento contract metadata file.
- `symbols` – Iterable of root symbols to trade (e.g. `["ES", "NQ"]`).
- `databento_api_key` – API key if not provided via `DATABENTO_API_KEY`.
- `models_path` – Directory containing ML bundles (defaults to `src/models`).
- `traderspost_webhook` – URL for order/trade notifications (or set
  `TRADERSPOST_WEBHOOK_URL`).
- `contracts` – Instrument-level overrides keyed by symbol (size, commission,
  multiplier, optional strategy overrides).
- Optional tuning knobs: `starting_cash`, `backfill`, `queue_maxsize`,
  `heartbeat_interval`, `heartbeat_file`, `heartbeat_write_interval`,
  `poll_interval`, `preflight.*` toggles for ML and connectivity validation or
  fail-fast behaviour.

### Sample Configuration

```yaml
contract_map: Data/Databento_contract_map.yml
models_path: src/models
symbols: ["ES", "NQ", "RTY"]
databento_api_key: ${DATABENTO_API_KEY}
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}
starting_cash: 250000
backfill: true
  queue_maxsize: 4096
  heartbeat_interval: 30
  heartbeat_file: /var/run/pine/worker_heartbeat.json
  heartbeat_write_interval: 30
  preflight:
    enabled: true
  skip_ml_validation: false
  skip_connection_checks: false
  fail_fast: true
contracts:
  ES:
    size: 1
    commission: 4.0
    strategy_overrides:
      pair_threshold: 0.6
  NQ:
    size: 1
    commission: 4.0
  RTY:
    size: 2
    commission: 4.5
```

Store the file securely (e.g. `/opt/pine/runtime/config.yml`) and reference it
with `PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml` when launching.

## Launching a Worker

### Local/Paper Trading

Run the live worker from a shell once the configuration file is in place:

```bash
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python - <<'PY'
from runner import LiveWorker, load_runtime_config

config = load_runtime_config()
worker = LiveWorker(config)
worker.run()
PY
```

This launches the Databento subscribers, hydrates ML bundles, and begins routing
orders to TradersPost. The worker handles `SIGINT`/`SIGTERM` for graceful
shutdown.

### Production via systemd

On the production host the workflow is:

1. Pull the latest code: `cd /opt/pine/rooney-capital-v1 && git pull`.
2. Ensure the runtime config (`PINE_RUNTIME_CONFIG`) and `.env` files are up to
   date with current credentials.
3. Restart the service: `sudo systemctl restart pine-runner.service`.
4. Monitor logs (`journalctl -u pine-runner.service -f`) for heartbeat or
   webhook errors. Use `POLICY_KILLSWITCH=true` in the environment to halt all
   trading if needed.

## Operations & Safety

- Never commit real credential files (`.env`, runtime configs) to source control.
- Always paper trade before enabling live capital and validate new models.
- Keep `POLICY_KILLSWITCH=true` available for emergency stops.
- Watch for data gaps >90 seconds and investigate Databento connectivity.
- Confirm TradersPost strategies are pointed at the desired contract before
  go-live; expiry handling lives in TradersPost rather than the worker.

### Pre-flight Validation Checks

`LiveWorker.run_preflight_checks()` executes a startup gauntlet before any live
orders are sent. The checks run in the following order and abort the launch if a
failure is detected (unless `preflight.fail_fast` is set to `false`):

1. **ML Models** – Ensure each configured symbol has a loadable model bundle,
   exposes `predict_proba`, defines at least one feature, and includes a veto
   threshold. A smoke `predict_proba` call guards against serialization drift.
2. **TradersPost Connection** – POST a health-check payload to the configured
   webhook and surface HTTP/connection/timeout errors with actionable log
   messages.
3. **Databento Connection** – Instantiate the live client and invoke a metadata
   probe to confirm the API key is valid and the gateway responds within 10
   seconds.
4. **Reference Data** – Verify required reference feeds exist in the contract
   map, confirm `CONTRACT_SPECS` covers every trading symbol, and warn when pair
   mappings are missing.
5. **Data Feeds** – Ensure Backtrader registered hourly and daily feeds for each
   symbol plus mandatory reference feeds such as `TLT_day`.

Configuration toggles under `preflight.*` allow skipping ML or connectivity
checks during local development while still documenting the bypass in logs.

## Additional References

- `docs/deployment_roadmap.md` – Historical production roadmap, monitoring
  guidance, and operational checklists.
