# Credentials & Secrets Management Guide

This document explains how to manage credentials for the Databento data feed, TradersPost webhook/API access, and any
auxiliary services that the live worker depends on. It supplements the brief notes in the `README.md` and
`docs/deployment_roadmap.md` files with explicit, actionable procedures.

## 1. Where secrets live

| Environment | Storage location | Notes |
| --- | --- | --- |
| **Production & paper servers** | `/opt/pine/Deployment/env/*.env` (`pine-trader.env`, `worker.env`, etc.) owned by the service account and `chmod 600` | Files loaded by systemd units or exported prior to running the worker. Keep them out of source control. |
| **Developer laptops / terminal servers** | A `.env` file outside the repo (e.g. `~/pine.env`) or exported directly into the shell session | Do **not** check credentials into Git. Prefer per-shell exports for short-lived tests. |
| **CI / automation** | CI secret store (e.g. GitHub Actions secrets, Hashicorp Vault, AWS/GCP secret manager) | Inject into the environment at job runtime; never store in plaintext artifacts. |

Runtime configuration files (`config.yml`) may interpolate environment variables (`${DATABENTO_API_KEY}`) so that the
plaintext secret never leaves the environment. The loader in `src/runner/live_worker.py` automatically expands these
placeholders (via `os.path.expandvars`) before any other validation occurs.

## 2. Loading secrets at runtime

1. Store the secret values in the environment via `export` or a sourced `.env` file.
2. Point the worker at the runtime config file: `export PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml`.
3. The loader in `src/runner/live_worker.py` reads secrets in the following priority order:
   - Explicit values inside the runtime config file.
   - Environment variables (`DATABENTO_API_KEY`, `TRADERSPOST_WEBHOOK_URL`, `TRADERSPOST_API_KEY`, etc.).

Prefer the environment-variable path so that configs can remain in Git without hard-coded credentials.

## 3. Paper vs. live credentials

Maintain **separate credentials** for each environment:

- Databento allows multiple API keys per account; generate one for paper/testing and one for production.
- TradersPost provides distinct webhook URLs and API keys for paper accounts vs. live brokers.
- Encode the mapping inside two `.env` files (e.g. `pine-trader.paper.env`, `pine-trader.live.env`) and copy/link the
  appropriate file before launching.

Document which key belongs to which environment in the runbook so operations staff can validate deployments quickly.

## 4. Rotation procedures

1. Generate the new credential via the provider console (Databento or TradersPost).
2. Update the secret storage location:
   - Replace the value in the managed secret store **or**
   - Edit the appropriate `.env` file (`Deployment/env/pine-trader.env`) on each host.
3. Restart affected services (`systemctl restart pine-worker@ES.service`, etc.) to pick up the new environment.
4. Revoke the old credential once the new one is confirmed working.
5. Record the change in the operations log / runbook for auditability.

For Databento keys, confirm connectivity by running the worker's pre-flight checks. For TradersPost webhooks, send a
manual `curl` health check (identical to what `LiveWorker.run_preflight_checks()` does) before cutting over live trading.

## 5. Handling secrets on a terminal server

If you are running the worker manually on a shared terminal server:

- Store credentials in a user-specific directory (`~/secure/pine.env`) with permissions `600`.
- Run `set -a; source ~/secure/pine.env; set +a` in each session before launching the worker so the environment variables
  populate without exposing them in your shell history.
- Clear the environment (`unset DATABENTO_API_KEY TRADERSPOST_WEBHOOK_URL TRADERSPOST_API_KEY`) when done.
- Never embed credentials directly in notebooks, scripts committed to Git, or long-running tmux sessions shared between
  operators.

## 6. Developer enablement

To help developers avoid accidental leaks:

- Ship an example env template (`Deployment/env/pine-trader.env.example`) with placeholder values and a prominent warning
  about not committing real secrets.
- Add pre-commit hooks or CI checks that fail if the repo contains patterns resembling API keys.
- Document the `dotenv` loading pattern in developer onboarding materials so local tests can run without modifying
  checked-in files.

Following these practices keeps live credentials separate, enables quick rotation, and reduces the chance of leaking
secrets via Git history or logs.
