# Supporting Assets & Assurance Catalogue

This catalogue summarises the non-runtime assets that provide operational
context, deployment guidance, and behavioural assurance for the Rooney Capital
trading stack. Use it as the entry point when you need to understand how the
repository is organised outside the `src/` runtime modules, what deployment
artefacts are maintained, and which automated tests cover the live trading
workflow.

## Repository Layout References

- **Top-level overview** – The [`README.md`](../README.md#repository-layout)
  call-out diagram lists the major directories and describes how
  configuration, runtime modules, and documentation are partitioned.
- **Contract metadata** – `Data/` holds
  `Databento_contract_map.yml`, the canonical mapping between Databento product
  codes, TradersPost metadata, and reference feed requirements. The
  [contract map loader tests](../tests/test_contract_map.py) assert that the
  schema, roll annotations, and reference feeds remain in sync with the runtime
  expectations.
- **Model artefacts** – `src/models/` packages Random Forest veto bundles per
  symbol. The loader is documented inline and validated by
  [`tests/test_models_loader.py`](../tests/test_models_loader.py) so changes to
  the bundle format are caught before impacting live workers.

## Deployment Notes & Operational Runbooks

The `docs/` directory curates runbooks and planning material that supplement the
runtime modules:

- [`deployment_roadmap.md`](./deployment_roadmap.md) records the historical
  roadmap for wiring Databento, Backtrader, and TradersPost. It doubles as a
  deployment runbook by capturing acceptance criteria, configuration files to
  populate, rollout sequencing, and watch-outs observed in prior launches.
- [`credentials_and_secrets.md`](./credentials_and_secrets.md) provides the
  policies for storing Databento and TradersPost secrets across environments,
  rotation procedures, and access management expectations.
- [`monitoring_and_health_checks.md`](./monitoring_and_health_checks.md)
  documents liveness checks, heartbeat files, and alerting hooks that keep live
  workers observable in production.
- [`ml_bundle_review.md`](./ml_bundle_review.md) captures the audit checklist for
  vetting new machine-learning bundles before they are promoted into the live
  deployment.

These notes ensure contributors have concrete references for the recurring
operational tasks that sit alongside the runtime code.

## Test Coverage Focused on Live Behaviour

Automated tests ensure the live worker and strategy behaviour remain stable:

- [`tests/test_live_worker_preflight.py`](../tests/test_live_worker_preflight.py)
  exercises the worker bootstrap path, validating that configuration checks,
  bundle loading, and data feed wiring surface actionable errors before a live
  deployment.
- [`tests/test_databento_bridge.py`](../tests/test_databento_bridge.py) covers
  the minute-bar aggregation and Backtrader feed adaptation logic so data
  regressions trigger test failures instead of live incidents.
- [`tests/test_ibs_strategy_ml.py`](../tests/test_ibs_strategy_ml.py) asserts the
  strategy integrates machine-learning veto signals and risk controls as
  expected, keeping signal generation predictable.
- [`tests/test_contract_map.py`](../tests/test_contract_map.py) and
  [`tests/test_models_loader.py`](../tests/test_models_loader.py) provide
  assurance that the metadata and artefact loaders the live worker depends on
  remain structurally sound.

Refer to [`tests/README.md`](../tests/README.md) for a quick summary of each test
module and how to extend the suite when adding new runtime features.

## How to Keep This Documentation Current

- Update the [`README.md`](../README.md) layout section whenever directories are
  added or repurposed so contributors always have an accurate map of the
  repository.
- Append to the runbooks in `docs/` as new deployment learnings, monitoring
  hooks, or credential policies are introduced.
- Extend the automated test suite and the accompanying documentation whenever
  new live worker code paths or strategy variants are introduced. Treat
  behavioural tests as gatekeepers for production stability.

Keeping these supporting assets in sync with the runtime modules ensures new
contributors inherit the full operational context required to maintain and
extend the trading system safely.
