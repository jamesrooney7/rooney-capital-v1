# Roadmap: Databento → Backtrader → TradersPost (Tradovate)

## 0) Outcomes (Acceptance Criteria)
- Data: Stable 1-minute bars for ES, NQ, RTY, YM, GC, SI, HG, 6A, 6B, 6E (front contracts) arrive every minute with ≤3s grace.
- Strategy: Existing Python strategy runs unchanged inside Backtrader; receives those bars and produces signals at bar-close.
- Execution: Signals post to TradersPost webhook with correct symbol, side, size, and metadata; fills/positions are reconciled every 5 minutes.
- Ops: Runs under systemd (auto-restart), with logs + basic alerts when a worker misses bars for >90s.

## 1) Repo & Process Layout
/opt/pine/pine-scripts/
├─ deployment_runtime/
│  ├─ contract_map.py
│  ├─ contract_codes.py
│  ├─ data_mapping.py
│  ├─ filter_config.py
│  ├─ hourly_runner.py
│  ├─ notifier.py
│  ├─ scheduler.py
│  ├─ traderspost_client.py
│  └─ tradable_universe.py
├─ Deployment/
│  ├─ contract_map.yml
│  ├─ reference_contracts.yml
│  ├─ Documentation/
│  ├─ env/
│  │  ├─ datahub.env.example
│  │  ├─ heartbeat.env.example
│  │  ├─ pine-trader.env.example
│  │  └─ worker.env.example
│  └─ systemd/
│     ├─ pine-datahub.service
│     ├─ pine-worker@.service
│     └─ pine-heartbeat.service
├─ Final Strategies/
│  ├─ <SYMBOL>_best.json
│  └─ Strategies/
├─ strategies/
├─ tests/
│  ├─ test_contract_map_loader.py
│  ├─ test_hourly_runner.py
│  ├─ test_scheduler.py
│  └─ …
├─ Deployment/Documentation/
│  └─ databento_backtrader_traderspost_roadmap.txt
├─ configs/
│  ├─ registry.yml
│  └─ trades_registry.yml
└─ deployment scripts, research notebooks, and analytics utilities as needed

## 2) Configs to Fill
### Deployment/contract_map.yml
Provides the canonical mapping between Databento product codes and the continuous root symbols pushed downstream. It is loaded by `deployment_runtime.contract_map.ContractMap` so every worker, notifier, and scheduler process can translate between Databento feeds and TradersPost payloads without manual file parsing.

### Deployment/reference_contracts.yml
Defines reference expiries and roll annotations for audits. The helper `deployment_runtime.reference_contracts` reads this file to validate that each root symbol points at the intended front-month when publishing continuous symbols.

> **Legacy note:** The modern stack in this repository now encodes reference
> subscriptions via the `reference_feeds` block inside
> `Data/Databento_contract_map.yml`; the previous `Data/data_reference_contracts.yml`
> file is kept out of tree for historical reference only.

### deployment_runtime/filter_config.py
Validates each instrument bundle referenced by the live runtime. `deployment_runtime.hourly_runner` imports this to enforce score thresholds, minimum signals, and other guardrails that mirror the offline evaluation pipeline.

### Final Strategies/<SYMBOL>_best.json
Each deployment bundle must publish a `Model_Path` that resolves to the serialized Random Forest artifact under [`Deployment/RF Models/`](../RF%20Models). The live loader (`deployment_runtime.filter_config`) validates this field and the `Prod_Threshold` so [`deployment_runtime.hourly_runner`](../../deployment_runtime/hourly_runner.py) can enforce the same cutoffs observed offline. Keep the JSON in source control so auditors can trace which model generated a live decision and where its pickled bundle resides.

### Deployment/env/pine-trader.env.example
Holds the Databento API key, TradersPost secrets, and other runtime credentials. Copy to `Deployment/env/pine-trader.env`, populate the environment variables, and `chmod 600` before deploying. Companion templates exist for worker/datahub/heartbeat roles.

## 3) Data Ingest (Databento)
- Subscribe to GLBX.MDP3 via Databento Python client.
- Aggregate to 1-minute if needed.
- Push closed bars into per-symbol queues keyed by the continuous root symbol.

## 4) Backtrader Live Feed Bridge
- Custom DataBase subclass reads from the queue.
- Converts (ts, o,h,l,c,v) to Backtrader lines.

## 5) Running the Strategy
- One worker per batch (~5 symbols).
- Add BTLiveFeed for each symbol.
- Run cerebro with strategy code unchanged.

## 6) Order Routing → TradersPost
- `deployment_runtime.traderspost_client.TradersPostClient` formats and posts orders to the TradersPost webhook using continuous root symbols.
- Retries with backoff on failure.

## 7) Position & Fill Reconciliation
- `deployment_runtime.notifier` and scheduler tasks poll TradersPost fills every N minutes.
- Compare local vs broker positions, fix drift.

## 8) TradersPost Expiry Handling
- Webhook payloads always use the continuous root symbol (ES, NQ, etc.).
- Configure TradersPost strategies to select the appropriate front-month contract; expiry logic now lives there.
- Keep `Deployment/contract_map.yml` and `Deployment/reference_contracts.yml` updated as reference metadata when auditing TradersPost settings.

## 9) Supervison, Systemd & Monitoring
- Systemd units manage startup, restart, logging.
- Heartbeat files for each worker; alert if >90s gap.

## 10) Security & Secrets
- Store API keys in env files, chmod 600. Follow the
  [Credentials & Secrets Management guide](./credentials_and_secrets.md) for
  environment-specific storage locations, rotation steps, and developer
  practices.
- Never log raw secrets.

## 11) Test Plan (Bring-up → Paper → Live)
- Local dry run with mock data (`pytest tests/test_hourly_runner.py`).
- Paper test with Databento live + TradersPost paper account.
- Canary live batch.
- Full live rollout.

## 12) Known Watchouts
- Bar delays: 3s grace, log misses.
- Reconnects: retry with backoff.
- TradersPost expiry policy: verify each strategy points at the desired contract before go-live since the runner no longer rolls locally.
- Webhook failures: retries then degrade symbol.

## 13) Time & Cost
- Build & wire: 1–3 dev days.
- Databento CME plan: ~$179/mo.
- TradersPost/Tradovate: per your broker plan.

## 14) Handoff Notes
- Strategy code unchanged.
- Easy swap of data vendor later.
- Runbook documents Databento credential rotation and TradersPost webhook ownership.
