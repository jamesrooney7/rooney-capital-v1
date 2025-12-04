# Rooney Capital Trading Strategy v1.0

Automated futures trading system using the Strategy Factory methodology with
ML meta-labeling models and live order routing through TradersPost.

---

## 📖 **Complete System Guide**

**👉 For operations, deployment, monitoring, and troubleshooting, see [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**

This unified guide contains:
- Current configuration (9 instruments, 2 max positions)
- Configuration management (unified in config.yml)
- Portfolio optimization workflow
- Deployment procedures
- Operations & monitoring
- Troubleshooting
- Quick command reference

---

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
- [Setup and Configuration](#setup-and-configuration)
  - [Initial Setup](#initial-setup)
  - [Running the Application](#running-the-application)
  - [Startup Script](#startup-script)
- [Runtime Configuration](#runtime-configuration)
  - [Required Fields](#required-fields)
  - [Sample Configuration](#sample-configuration)
- [Launching a Worker](#launching-a-worker)
  - [Local/Paper Trading](#localpaper-trading)
  - [Production via systemd](#production-via-systemd)
- [Operations & Safety](#operations--safety)
- [Ubuntu Launch Guide](#ubuntu-launch-guide)
- [Additional References](#additional-references)

## System Overview

The Rooney Capital stack ingests live CME Globex data from Databento, evaluates
multi-asset strategy signals on 15-minute bars inside Backtrader, and routes
execution events to TradersPost for Tradovate brokerage accounts. Each symbol/
strategy combination has a dedicated ML meta-labeling model that filters signals
based on probability thresholds trained via walk-forward optimization.

### End-to-end Flow

1. **Contract governance** – `Data/` YAML files and helper utilities in
   `src/runner/contract_map.py` define Databento datasets, roll rules, and
   TradersPost metadata for each futures root.
2. **Live market data ingestion** – `src/runner/databento_bridge.py` subscribes
   to the Databento live gateway, aggregates ticks into one-minute bars, and
   pushes them into per-symbol queues that behave like Backtrader feeds.
3. **Strategy orchestration** – `src/runner/live_worker.py` creates the
   Backtrader `Cerebro` instance, wires a `DatabentoLiveData` feed per symbol,
   and attaches `StrategyFactoryAdapter` with instrument-specific overrides.
4. **ML meta-labeling models** – `src/models/factory_loader.py` loads models
   from `models/factory/` so the strategy can filter trades based on ML
   probability scores.
5. **Trading logic** – `src/strategy/factory_adapter.py` implements signal
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
└── models/       # Model loader utilities
models/
└── factory/      # ML meta-labeling models per symbol/strategy combination
tests/            # Integration and unit coverage for live worker & strategy
Data/             # Contract metadata and roll rules
docs/             # Deployment notes and operational documentation
research/         # Strategy Factory research pipeline and ML training
config.example.yml # Template runtime configuration with environment expansion hints
.env.example      # Template environment variables (copy to .env with real secrets)
requirements.txt  # Python dependencies
```

Additional context:

- **Data/** – Static metadata (`Databento_contract_map.yml`) that drive
  subscription choices and audit rolls. Reference subscriptions live inside the
  `reference_feeds` block of this contract map.
- **docs/** – Includes the deployment roadmap and background notes preserved for
  operational reference.
- **tests/** – Validates the live worker orchestration and strategy behaviour,
  providing integration and unit-level examples contributors can extend.
- **config.example.yml** – Reference runtime configuration showing required keys
  and `${VAR}` expansion syntax. Copy to `config.yml` when provisioning a host.
- **.env.example** – Safe template for Databento and TradersPost credentials
  plus safety toggles. Duplicate as `.env` and fill with environment-specific
  secrets.

## Core Modules

### `src/config.py`
Centralises runtime constants so workers and backtests remain in sync. Notable
exports include:

- `COMMISSION_PER_SIDE` – Default commission per side with environment overrides.
- `PAIR_MAP` – Cross-instrument relationships used for pair-based filters.
- `REQUIRED_REFERENCE_FEEDS` – Guard ensuring daily reference data is loaded
  before the strategy initialises.

### `src/models`
Contains model loader utilities. The production models reside in `models/factory/`:

- `factory_loader.py` – Loads ML meta-labeling models from `models/factory/`.
- Models are named `{SYMBOL}_{STRATEGY}_ml_meta_labeling_final_model.pkl`.
- Each model is trained via walk-forward optimization on 15-minute bar data.
- Provides probability-based signal filtering with configurable thresholds.

### `src/strategy`

- **`factory_adapter.py`** – Core Backtrader strategy implementing the Strategy
  Factory system. It consumes intraday and reference data feeds per symbol,
  computes indicators via `feature_utils.py`, and applies ML meta-labeling
  probability filters along with portfolio-level risk controls. Notification
  hooks forward order/trade events to the live worker.
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

Each symbol/strategy combination has a dedicated ML meta-labeling model trained
via walk-forward optimization. Models are stored in `models/factory/` with naming
convention `{SYMBOL}_{STRATEGY}_ml_meta_labeling_final_model.pkl`.

Current production models (14 total):
- Energy: `CL_ATRBuyDip`, `CL_MomentumBreakout`, `NG_RangeReversion`
- Metals: `GC_VolatilityBreakout`, `HG_OpeningRangeBreakout`, `SI_MomentumBreakout`
- Indices: `ES_OpeningRangeBreakout`, `NQ_OpeningRangeBreakout`, `RTY_RangeReversion`, `YM_RangeReversion`
- Currencies: `6A_VolatilityBreakout`, `6B_MomentumBreakout`, `6E_MomentumBreakout`, `6E_TrendFollowing`

> **Note:** Model artefacts are stored with Git LFS. Run `git lfs pull` after
> cloning to ensure large files are available before attempting to load them.

## Setup and Configuration

### Initial Setup

1. **Create configuration file**:

   ```bash
   cp config.example.yml config.yml
   ```

2. **Create environment file**:

   ```bash
   cat > .env << 'EOF'
   DATABENTO_API_KEY=your_databento_api_key_here
   TRADERSPOST_WEBHOOK_URL=your_traderspost_webhook_url_here
   POLICY_KILLSWITCH=true
   EOF
   ```

3. **Verify contract map exists**:
   Ensure `Data/Databento_contract_map.yml` exists and is up to date with your
   Databento account products.

4. **Install dependencies**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the Application

**Important**: Run from the project root directory with `PYTHONPATH` set:

```bash
cd /path/to/rooney-capital-v1
source venv/bin/activate
PYTHONPATH=/path/to/rooney-capital-v1/src python -m runner.main
```

Or create a startup script (see below).

### Startup Script

The repository includes `scripts/start.sh`, which enforces the required
configuration files, activates the local virtual environment, sets
`PYTHONPATH`, and launches `python -m runner.main` from the project root. Make
it executable after cloning:

```bash
chmod +x scripts/start.sh
```

Run the script directly to start the worker:

```bash
./scripts/start.sh
```

### Additional Setup Notes

1. **Clone** the repository and install Git LFS if it is not already present:
   `git lfs install` followed by `git lfs pull`.
2. **Populate data metadata** as needed in `Data/Databento_contract_map.yml` so
   the worker can subscribe to the desired products. Reference subscriptions are
   defined via the `reference_feeds` block inside this file.
3. **Ensure credentials** (Databento API key, TradersPost webhook URL) are
   available via environment variables or the runtime configuration file. See
   [Credentials & Secrets Management](docs/credentials_and_secrets.md) for
   detailed guidance on storage, rotation, and environment separation.

## Runtime Configuration

Workers load their runtime configuration from a JSON or YAML file. Provide the
path explicitly to `load_runtime_config()` or set the `PINE_RUNTIME_CONFIG`
environment variable before starting the process. String-valued entries support
`${VAR}` placeholders which are expanded from the current environment prior to
validation so secrets can remain outside the configuration file.

### Required Fields

- `contract_map` – Path to the Databento contract metadata file.
- `symbols` – Iterable of root symbols to trade (e.g. `["ES", "NQ"]`).
- `databento_api_key` – API key if not provided via `DATABENTO_API_KEY`.
- `models_path` – Directory containing ML models (defaults to `models/factory`).
- `traderspost_webhook` – URL for order/trade notifications (or set
  `TRADERSPOST_WEBHOOK_URL`).
- `contracts` – Instrument-level overrides keyed by symbol (size, commission,
  multiplier, optional strategy overrides).
- Optional tuning knobs: `starting_cash`, `backfill`, `backfill_minutes`
  (or `backfill_hours`), `resample_session_start`, `queue_maxsize`,
  `heartbeat_interval`, `heartbeat_file`, `heartbeat_write_interval`,
  `poll_interval`, `preflight.*` toggles for ML and connectivity validation or
  fail-fast behaviour.

> **Operational note:** TradersPost currently exposes only the webhook
> integration used for order and trade notifications. Portfolio reconciliation
> must be performed manually via the TradersPost console or broker statements
> because the REST API is not available.

### Sample Configuration

```yaml
contract_map: Data/Databento_contract_map.yml
models_path: models/factory
symbols: ["ES", "NQ", "RTY"]
databento_api_key: ${DATABENTO_API_KEY}
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}
starting_cash: 250000
backfill: true
backfill_minutes: 180
resample_session_start: "23:00"
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

Run the live worker from a shell once the configuration file is in place. The
repository includes `scripts/launch_worker.py`, which adjusts `PYTHONPATH`
automatically and prints safety warnings around the kill switch:

```bash
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python scripts/launch_worker.py
```

If you prefer to inline the command (for example on ephemeral hosts) the
following snippet is equivalent:

```bash
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python - <<'PY'
from runner import LiveWorker, load_runtime_config

config = load_runtime_config()
worker = LiveWorker(config)
worker.run()
PY
```

To verify configuration and connectivity without running the live loop, use the
bundled preflight helper:

```bash
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python scripts/worker_preflight.py
```

This launches the Databento subscribers, hydrates ML bundles, and begins routing
orders to TradersPost. The worker handles `SIGINT`/`SIGTERM` for graceful
shutdown.

To review ML feature readiness without starting the trading loop, inspect the
heartbeat file (or the live worker directly) using the bundled helper:

```bash
export PINE_RUNTIME_CONFIG=/path/to/config.yml
python scripts/inspect_ml_features.py --show-features
```

By default the script prefers the heartbeat file configured in the runtime
config and falls back to instantiating `LiveWorker` if the heartbeat is
unavailable. Use `--source heartbeat` or `--source worker` to force a specific
data source.

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

1. **ML Models** – Ensure each configured symbol/strategy combination has a
   loadable model in `models/factory/`, exposes `predict_proba`, and meets
   probability threshold requirements for signal filtering.
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

## Ubuntu Launch Guide

Need a step-by-step, copy-and-paste friendly provisioning flow for a fresh
Ubuntu host? See the dedicated [Ubuntu launch guide](docs/ubuntu_launch_guide.md)
for four deployment sprints covering dependency installs, credential setup,
validation, and systemd hardening.

## Additional References

- `docs/ubuntu_launch_guide.md` – End-to-end Ubuntu provisioning guide covering
  dependency installation, runtime configuration, validation, and systemd
  deployment.
- `docs/supporting_assets_assurance.md` – Catalogue of supporting
  documentation, deployment runbooks, and test coverage that provide operational
  context beyond the runtime modules.
- `docs/deployment_roadmap.md` – Historical production roadmap, monitoring
  guidance, and operational checklists.
- `docs/ml_bundle_review.md` – Deep dive into ML bundle metadata, highlighting
  feature mismatches between saved snapshots and model expectations plus
  remediation steps.
- `docs/monitoring_and_health_checks.md` – Operational playbook for configuring
  heartbeat files, lightweight probes, and staged alert coverage as monitoring
  matures.
- `tests/README.md` – Overview of the automated test suite that guards the live
  worker, strategy behaviour, and supporting loaders.
