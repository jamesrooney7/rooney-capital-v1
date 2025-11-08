#!/bin/bash
# Start IBS A Strategy Worker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start IBS A worker
python -m src.runner.strategy_worker \
    --strategy ibs_a \
    --config config.multi_alpha.yml \
    2>&1 | tee -a logs/ibs_a_worker.log
