# Gap Analysis: Databento → IBS Strategy → TradersPost Flow

This note reviews the repository against the intended production path:

> Databento minute bars → run `IbsStrategy` → keep trades passing the ML filter → post a webhook to TradersPost.

## What already exists

- **Strategy & filter plumbing** – `src/strategy/ibs_strategy.py` contains the trading logic and already knows how to accept an `ml_model`, `ml_features`, and `ml_threshold` argument so that a classifier can veto orders.【F:src/strategy/ibs_strategy.py†L392-L409】【F:src/strategy/ibs_strategy.py†L888-L911】
- **Model bundles** – `src/models/loader.py` and the artefacts in `src/models/` supply the tuned Random Forest payloads plus production thresholds, which can be converted into the kwargs that `IbsStrategy` expects.【F:src/models/loader.py†L1-L91】【F:src/models/loader.py†L118-L148】
- **Operational roadmap** – `docs/deployment_roadmap.md` sketches the desired end-to-end environment, including Databento ingestion, Backtrader feeds, and TradersPost routing, but it reads as a plan rather than implemented code.【F:docs/deployment_roadmap.md†L1-L116】

## Gaps to close for the end-to-end flow

1. **Live data bridge** – The roadmap calls for a Databento subscriber, in-memory queues, and a Backtrader `DataBase` subclass, yet the repository does not ship any of that runtime code. `src/runner/` is empty, so there is currently no process that connects live bars to `IbsStrategy`. Implementing the feed handler (subscription, roll mapping, queue fan-out, and a Backtrader data feed adapter) is the first blocker.【F:docs/deployment_roadmap.md†L69-L95】【F:src/runner/__init__.py†L1-L1】
2. **Strategy orchestrator** – Beyond the strategy class itself, there is no script or service that instantiates Cerebro, loads model bundles per symbol, attaches the strategy, and advances the engine in real time. Creating a runner module that loops over the configured futures universe, calls `load_model_bundle(symbol)`, and injects the resulting kwargs into `cerebro.addstrategy` will wire the ML veto into live trading.【F:src/models/loader.py†L118-L148】【F:src/strategy/ibs_strategy.py†L888-L909】
3. **Execution plumbing** – The roadmap references a `traderspost_client` with retry logic, but the codebase lacks any webhook client or order formatting utilities. We still need a component that listens for strategy events (order notifications, trades) and translates them into TradersPost API payloads, including production thresholds, sizing, and error handling.【F:docs/deployment_roadmap.md†L96-L129】
4. **Configuration surface** – `IbsStrategy` imports `config` for commission rates and pair mappings, yet no `config.py` exists in the repo. Until we add a configuration module (or replace these dependencies with something under source control), the strategy cannot be instantiated without throwing an `ImportError`. Resolving configuration is required for both live and backtest runners.【F:src/strategy/ibs_strategy.py†L10-L15】
5. **Databento symbol governance** – The `Data/` folder carries static YAML maps, but we have no loader utilities inside `src/` that consume them to drive roll decisions or translate between Databento instruments and TradersPost symbols. Building those helpers (or porting them from the deployment runtime described in the roadmap) is necessary so the live bridge can choose the right contract per root symbol.【F:Data/Databento_contract_map.yml†L1-L40】【F:docs/deployment_roadmap.md†L33-L68】

## Suggested next steps

1. Stand up a `src/runner/live_worker.py` (or similar) that:
   - loads config + contract metadata,
   - spins up the Databento client with queues per symbol,
   - instantiates Backtrader with one feed per root, and
   - attaches `IbsStrategy` per symbol with `strategy_kwargs_from_bundle`.
2. Add a TradersPost webhook client module with retries and logging, then hook it into the Backtrader order notification callbacks.
3. Backfill the missing `config` module (commission, symbol pairs, policy toggles) so local tests and the runner import cleanly.
4. Write integration tests (or at least dry-run scripts) that simulate a few bars through the bridge to verify ML filtering and webhook emission before deploying.

Documenting these gaps clarifies why the repo currently stops at the strategy and model assets stage and what must be implemented to achieve the fully automated flow.
