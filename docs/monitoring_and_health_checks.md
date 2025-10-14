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
seconds while running, and again when it stops. A failed pre-flight check will
mark the heartbeat with a `failed` status so that an external monitor can
escalate immediately.

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

- **Databento ingest health** – compare the last bar timestamp reported in the
  heartbeat `details.databento.subscribers[*].last_emitted_minute` field to
  detect stalled market data.
- **Webhook delivery** – trigger alerts using
  `details.traderspost.last_error` when TradersPost returns a failure.
- **Position drift** – periodically compare TradersPost positions against local
  expectations and alert if they diverge for more than one run interval.

Each enhancement should surface a concise error in the heartbeat `details`
section so every alert is tied to a machine-readable root cause.

### Heartbeat `details` fields

The worker now records additional telemetry in the heartbeat file:

```json
{
  "status": "running",
  "details": {
    "preflight": {
      "status": "passed",
      "checked_at": "2024-05-12T09:31:00Z",
      "failed_checks": []
    },
    "databento": {
      "queue_fanout": {
        "known_symbols": ["ES", "NQ"],
        "queue_depths": {"ES": 0, "NQ": 0},
        "mapped_instruments": 2
      },
      "subscribers": [
        {
          "dataset": "GLBX.MDP3",
          "product_codes": ["ESH4"],
          "last_emitted_minute": {"ES": "2024-05-12T09:30:00Z"},
          "last_error": null
        }
      ]
    },
    "traderspost": {
      "last_success": {
        "at": "2024-05-12T09:32:17Z",
        "kind": "order",
        "symbol": "ES",
        "side": "BUY",
        "size": 1
      },
      "last_error": null
    }
  }
}
```

With `jq` installed you can extract key indicators quickly:

```bash
jq -r '.details.preflight.status' /var/run/pine/worker_heartbeat.json
jq -r '.details.databento.subscribers[].last_emitted_minute | to_entries[] | "\(.key): \(.value)"' \
  /var/run/pine/worker_heartbeat.json
jq -r '.details.traderspost.last_error // "(no webhook errors)"' \
  /var/run/pine/worker_heartbeat.json
```
