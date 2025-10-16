#!/bin/bash
# Rooney Capital - Application Startup Script

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for required files
if [ ! -f "config.yml" ]; then
    echo "ERROR: config.yml not found. Copy config.example.yml to config.yml and configure it."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Create .env with required credentials."
    exit 1
fi

if [ ! -f "Data/Databento_contract_map.yml" ]; then
    echo "ERROR: Data/Databento_contract_map.yml not found."
    exit 1
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

source venv/bin/activate

# Set PYTHONPATH and run
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
python -m src.runner.main

