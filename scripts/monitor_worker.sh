#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-pine-runner.service}
HEARTBEAT_FILE=${HEARTBEAT_FILE:-/var/run/pine/worker_heartbeat.json}
MAX_AGE_SECONDS=${MAX_AGE_SECONDS:-120}

log_section() {
  echo "=== $1 ==="
}

check_service() {
  log_section "Service Status"
  if systemctl list-unit-files "${SERVICE_NAME}" >/dev/null 2>&1; then
    systemctl is-active "${SERVICE_NAME}" || true
  else
    echo "service ${SERVICE_NAME} is not installed"
  fi
  echo
}

check_heartbeat() {
  log_section "Heartbeat"
  if [[ -f "${HEARTBEAT_FILE}" ]]; then
    echo "Heartbeat file: ${HEARTBEAT_FILE}"
    local file_age
    file_age=$(( $(date +%s) - $(stat -c %Y "${HEARTBEAT_FILE}") ))
    echo "Age: ${file_age}s"
    if (( file_age > MAX_AGE_SECONDS )); then
      echo "⚠️  WARNING: heartbeat is older than ${MAX_AGE_SECONDS}s"
    else
      echo "✓ Heartbeat is fresh"
    fi
    echo
    echo "Latest entry:"
    if command -v jq >/dev/null 2>&1; then
      jq . "${HEARTBEAT_FILE}" || cat "${HEARTBEAT_FILE}"
    else
      cat "${HEARTBEAT_FILE}"
    fi
  else
    echo "✗ Heartbeat file not found"
  fi
  echo
}

show_logs() {
  log_section "Recent Logs"
  if systemctl list-unit-files "${SERVICE_NAME}" >/dev/null 2>&1; then
    if command -v journalctl >/dev/null 2>&1; then
      sudo journalctl -u "${SERVICE_NAME}" -n 20 --no-pager || true
    else
      echo "journalctl not available"
    fi
  else
    echo "service ${SERVICE_NAME} is not installed"
  fi
}

check_service
check_heartbeat
show_logs
