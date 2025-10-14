# Test Suite Overview

The automated tests focus on verifying the live trading behaviour and critical
loaders used by the Rooney Capital stack. Use this guide to identify where to
add new coverage when evolving the runtime modules.

## Live Worker & Data Bridge

- **`test_live_worker_preflight.py`** – Exercises worker bootstrap logic,
  asserting configuration validation, ML bundle hydration, and feed wiring all
  surface actionable failures before deployment.
- **`test_databento_bridge.py`** – Validates the Databento-to-Backtrader bridge
  by simulating tick ingestion, minute-bar aggregation, and heartbeat handling.

## Strategy Behaviour

- **`test_ibs_strategy_ml.py`** – Confirms the IBS strategy integrates machine
  learning veto signals, position sizing, and risk controls as expected when
  running inside Backtrader.

## Metadata & Artefact Loaders

- **`test_contract_map.py`** – Guards the schema and integrity of
  `Data/Databento_contract_map.yml`, ensuring metadata the live worker relies on
  does not drift.
- **`test_models_loader.py`** – Verifies model bundle deserialisation and the
  translation of artefacts into strategy keyword arguments.

## Shared Fixtures

- **`conftest.py`** – Provides reusable fixtures for fake Databento data,
  ephemeral model bundles, and configuration overrides. Extend this module when
  new integration pathways require complex setup/teardown behaviour.

## Extending the Suite

When adding new runtime features:

1. Prefer integration-style tests that simulate the live worker workflow to
   catch regressions in orchestration, configuration, and risk controls.
2. Co-locate unit tests for helper modules next to the live behaviour tests so
   contributors can quickly discover related coverage.
3. Update this README with the new tests so future maintainers understand the
   assurance story for the repository.
