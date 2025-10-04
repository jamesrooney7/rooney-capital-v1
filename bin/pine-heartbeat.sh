#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HEARTBEAT_COMMAND:-}" ]]; then
  echo "HEARTBEAT_COMMAND is not set; provide a URL or command to execute." >&2
  exit 1
fi

notify_failure() {
  local status="$1"
  HEARTBEAT_EXIT_CODE="$status" python3 - <<'PY'
import os
import sys

from deployment_runtime import notifier

command = os.environ.get("HEARTBEAT_COMMAND", "")
exit_code_raw = os.environ.get("HEARTBEAT_EXIT_CODE", "0")

try:
    exit_code = int(exit_code_raw)
except ValueError:
    exit_code = 0

try:
    notifier.send_heartbeat_alert(command=command, exit_code=exit_code)
except notifier.NotifierConfigError as exc:
    print(f"SMTP notifier not configured: {exc}", file=sys.stderr)
except notifier.NotificationError as exc:
    print(f"Failed to send heartbeat alert: {exc}", file=sys.stderr)
PY
}

set +e
if [[ "$HEARTBEAT_COMMAND" =~ ^https?:// ]]; then
  curl_bin="$(command -v curl || true)"
  if [[ -z "$curl_bin" ]]; then
    echo "curl not found on PATH" >&2
    status=1
  else
    "$curl_bin" --fail --silent --show-error "$HEARTBEAT_COMMAND"
    status=$?
  fi
else
  /bin/sh -c "$HEARTBEAT_COMMAND"
  status=$?
fi
set -e

if [[ $status -ne 0 ]]; then
  notify_failure "$status"
fi

exit "$status"
