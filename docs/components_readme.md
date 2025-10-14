# Rooney Capital Trading System Overview

This document walks through every major component that powers the Rooney Capital
IBS trading stack.  It explains what lives in each directory, how the runtime
moves market data from Databento through the Backtrader engine, and how
execution events reach TradersPost.  Use it as the canonical map when onboarding
developers or reviewing the live deployment plan.

## End-to-end Flow

1. **Contract governance** – `Data/` YAML files and the helpers in
   `src/runner/contract_map.py` declare the Databento datasets, roll rules, and
   TradersPost metadata for each futures root.
2. **Live market data ingestion** – `src/runner/databento_bridge.py` subscribes
   to the Databento live gateway, aggregates ticks into one-minute bars, and
   pushes them into per-symbol queues that behave like Backtrader data feeds.
3. **Strategy orchestration** – `src/runner/live_worker.py` creates the
   Backtrader `Cerebro` instance, wires in a `DatabentoLiveData` feed per root,
   and attaches `IbsStrategy` with the Random Forest veto models.
4. **ML veto models** – `src/models/loader.py` loads instrument-specific model
   bundles so the strategy can filter trades based on ML probability scores.
5. **Trading logic** – `src/strategy/ibs_strategy.py` implements signal
   generation, position management, and portfolio-level risk controls.
6. **Execution webhook** – `src/runner/traderspost_client.py` serialises order
   and trade notifications and posts them to TradersPost with retry logic.
7. **Configuration + feature utilities** – `src/config.py` and utilities in
   `src/strategy/feature_utils.py` expose shared constants and computed
   indicators used across the stack.

The live worker keeps looping: new bars arrive via Databento, `Cerebro` steps
through the feeds, the strategy evaluates ML-qualified signals, and the runtime
posts resulting executions to TradersPost.

## Repository Structure

### Root Level

- **README.md** – High-level summary, supported symbols, and operational
  checklist.
- **requirements.txt** – Python dependencies needed to run the strategy,
  runners, and tests.
- **docs/** – Additional documentation including the deployment roadmap and
  this component-level overview.
- **Data/** – Static metadata required for roll decisions and contract
  governance.
- **Examples/** – Removed in this snapshot; future walkthroughs can live in ad
  hoc notebooks alongside feature branches instead of this placeholder.
- **bin/** – Removed pending a replacement for the former heartbeat helper that
  depended on an unreleased `deployment_runtime` package.
- **tests/** – Historical pytest suite (configuration parsing, contract map
  loaders, TradersPost client retries) that has been retired pending a new test
  plan aligned with the current runtime.

### `src/config.py`

Centralised configuration module that keeps the live workers and backtests in
sync.  It exposes:

- `COMMISSION_PER_SIDE` – Default per-side commission with environment variable
  overrides.
- `PAIR_MAP` – Cross-instrument mappings used by `IbsStrategy` for pair-based
  filters.
- `REQUIRED_REFERENCE_FEEDS` – Safety guard to ensure daily reference data is
  present before the strategy initialises.

The module supports JSON or `key:value` overrides so operations can tune
commission or pairings without modifying source code.

### `src/models/`

Holds Random Forest artefacts (`*_rf_model.pkl`) and optimisation metadata
(`*_best.json`) for each supported symbol.  The loader (`loader.py`):

- Hydrates the scikit-learn model, feature list, and probability threshold.
- Normalises filesystem paths so the worker can locate artefacts relative to the
  repository or an injected models directory.
- Provides `strategy_kwargs_from_bundle` which translates a model bundle into
  keyword arguments accepted by `IbsStrategy` (including the ML veto hook).

### `src/strategy/`

#### `ibs_strategy.py`

The core Backtrader strategy implementing the Internal Bar Strength system:

- Consumes intraday and reference data feeds per symbol plus a companion
  instrument for pair-based filters.
- Computes indicators using helpers from `feature_utils.py`, `filter_column.py`,
  and `safe_div.py`.
- Applies multiple risk controls: ML veto, pair IBS thresholds, time-of-day
  gating, and trade cooldowns.
- Exposes notification hooks (`notify_order`, `notify_trade`) which the live
  worker uses to push execution events to TradersPost.

#### `feature_utils.py`, `filter_column.py`, `safe_div.py`

Utility modules encapsulating repeated calculations:

- Feature engineering for the ML classifier (rolling highs/lows, RSI-2, ATR,
  etc.).
- Safe mathematical operations that guard against divide-by-zero or missing
  data.
- Column filtering logic used during feature selection.

#### `contract_specs.py`

Reference information about contract multipliers, tick sizes, and leverage
characteristics.  These specs inform sizing logic in both backtests and live
runs.

### `src/runner/`

#### `contract_map.py`

Parses `Data/Databento_contract_map.yml` (plus optional JSON variants) and
validates the schema.  It exposes `ContractMap`, which resolves:

- Active contract metadata and Databento subscription parameters.
- Tradovate/TradersPost symbology and descriptive fields for webhook payloads.
- Reference feed definitions for daily bars required by the strategy.

#### `databento_bridge.py`

Implements the live market data bridge:

- `DatabentoSubscriber` connects to the Databento live gateway and normalises
  `TradeMsg` streams into one-minute OHLCV bars.
- `QueueFanout` keeps per-symbol queues, manages contract roll updates, and
  distributes bars to Backtrader consumers.
- `DatabentoLiveData` adapts queue output into Backtrader’s `DataBase` interface
  with heartbeat handling, reset signals, and graceful shutdown semantics.

#### `traderspost_client.py`

Responsible for resilient webhook delivery:

- Formats order and trade notifications into TradersPost-compatible payloads.
- Implements exponential backoff retries on transient HTTP errors.
- Attaches ML threshold snapshots and contract metadata for downstream audit.

#### `live_worker.py`

The live orchestrator that stitches everything together:

- Loads runtime configuration (API keys, webhook URLs, instrument sizing) via
  `load_runtime_config` and optional JSON/YAML overlays.
- Instantiates `DatabentoSubscriber` and `QueueFanout`, then registers one
  `DatabentoLiveData` feed per symbol.
- Hydrates Random Forest model bundles with `load_model_bundle` and attaches
  `NotifyingIbsStrategy` wrappers so execution events can be forwarded to the
  TradersPost client.
- Runs an event loop that steps `Cerebro` whenever new bars arrive, handles
  graceful shutdown on signals, and records structured logs for observability.

### `tests/`

This snapshot does not include an active automated test suite.  The previous
pytest collection covering configuration parsing, contract map lookups, and the
TradersPost client was removed while the team defines updated coverage targets
for the live runtime components.

## Supporting Documentation

- **docs/deployment_roadmap.md** – Historical notes describing the production
  environment topology and operations checklist.
- **docs/components_readme.md** (this file) – Component-level reference for the
  current repository snapshot.

## How to Use This Document

- **Onboarding** – Read the end-to-end flow and relevant component sections to
  understand how live trading is assembled.
- **Troubleshooting** – Trace data or execution issues by walking through the
  flow: start at Databento ingestion and follow the queue to strategy callbacks
  and webhook delivery.
- **Planning** – Reference module responsibilities when scoping new features or
  operational tasks to ensure changes land in the appropriate layer.

