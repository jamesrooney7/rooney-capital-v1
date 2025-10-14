#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_DIR=${RUNTIME_DIR:-/opt/pine/runtime}
SERVICE_NAME=${SERVICE_NAME:-pine-runner.service}
MODELS_DIR=${MODELS_DIR:-${REPO_ROOT}/src/models}
CONFIG_FILE=${CONFIG_FILE:-${RUNTIME_DIR}/config.yml}
ENV_FILE=${ENV_FILE:-${RUNTIME_DIR}/.env}

model_count=$(ls -1 "${MODELS_DIR}"/*_rf_model.pkl 2>/dev/null | wc -l || true)
config_exists="$( [[ -f "${CONFIG_FILE}" ]] && echo "✓" || echo "✗" )"
env_exists="$( [[ -f "${ENV_FILE}" ]] && echo "✓" || echo "✗" )"
contract_exists="$( [[ -f "${REPO_ROOT}/Data/Databento_contract_map.yml" ]] && echo "✓" || echo "✗" )"
venv_exists="$( [[ -d "${REPO_ROOT}/venv" ]] && echo "✓" || echo "✗" )"
if systemctl list-unit-files "${SERVICE_NAME}" >/dev/null 2>&1; then
  service_installed="✓"
else
  service_installed="✗"
fi

killswitch_state="unknown"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  if [[ "${POLICY_KILLSWITCH:-}" == "true" ]]; then
    killswitch_state="⚠️  ENABLED (paper mode)"
  elif [[ "${POLICY_KILLSWITCH:-}" == "false" ]]; then
    killswitch_state="✓ Disabled (live mode)"
  else
    killswitch_state="? POLICY_KILLSWITCH not set"
  fi
else
  killswitch_state="✗ Env file missing"
fi

cat <<OUT
=== Pre-Launch Checklist ===

[1/7] ML models: $( [[ ${model_count} -gt 0 ]] && echo "✓ Found ${model_count}" || echo "✗ None found" )
[2/7] Runtime config: ${config_exists} $( [[ ${config_exists} == "✓" ]] && echo "${CONFIG_FILE}" || echo "missing" )
[3/7] Environment file: ${env_exists} $( [[ ${env_exists} == "✓" ]] && echo "${ENV_FILE}" || echo "missing" )
[4/7] Contract map: ${contract_exists} ${REPO_ROOT}/Data/Databento_contract_map.yml
[5/7] Virtualenv: ${venv_exists} ${REPO_ROOT}/venv
[6/7] Systemd service: ${service_installed} ${SERVICE_NAME}
[7/7] Kill switch: ${killswitch_state}

Checklist complete.
OUT
