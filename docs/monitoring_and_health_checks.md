# Monitoring and Health Checks

This document outlines the first steps for monitoring the live worker and
raising an alert if it stops processing data.

## 1. Enable the built-in heartbeat file

The live worker can now write a JSON heartbeat file that records the worker's
status, timestamp, PID, and configured symbol universe. Configure the runtime
with:

```yaml
heartbeat_file: /var/run/pine/worker_heartbeat.json
heartbeat_write_interval: 30  # seconds between updates while healthy
```

The worker updates the file when it starts, every `heartbeat_write_interval`
seconds while running, and again when it stops. A failed pre-flight check or
startup reconciliation will mark the heartbeat with a `failed` status so that an
external monitor can escalate immediately.

## 2. Add a lightweight heartbeat checker

Deploy a systemd timer/cron job (e.g., every minute) that inspects the file's
`updated_at` timestamp. When the timestamp is older than your tolerance window
(90 seconds is the existing operational target), emit an alert through your
preferred channel (PagerDuty, Slack, etc.). A simple shell probe looks like:

```bash
#!/usr/bin/env bash
set -euo pipefail

HEARTBEAT_FILE="/var/run/pine/worker_heartbeat.json"
THRESHOLD_SECONDS=90

if [[ ! -s "${HEARTBEAT_FILE}" ]]; then
  echo "heartbeat file missing" >&2
  exit 2
fi

last_update=$(jq -r '.updated_at' "${HEARTBEAT_FILE}")
last_epoch=$(date --date="${last_update}" +%s)
now_epoch=$(date +%s)

if (( now_epoch - last_epoch > THRESHOLD_SECONDS )); then
  echo "worker heartbeat stale: $(( now_epoch - last_epoch ))s" >&2
  exit 1
fi
```

Return codes `1`/`2` can be mapped to warning/critical alerts depending on your
monitoring stack.

## 3. Extend alert coverage over time

Once the heartbeat alarm is in place, layer on additional guards:

- **Databento ingest health** – compare the last bar timestamp in logs or the
  heartbeat `details` field (future extension) to detect stalled market data.
- **Webhook delivery** – trigger alerts on repeated TradersPost errors or
  retries exceeding a threshold.
- **Position drift** – periodically compare TradersPost positions against local
  expectations and alert if they diverge for more than one run interval.

Each enhancement should surface a concise error in the heartbeat `details`
section so every alert is tied to a machine-readable root cause.
