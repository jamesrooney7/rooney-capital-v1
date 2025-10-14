# Rooney Capital Trading System - Ubuntu Launch Guide

## Quick Start Overview

This guide will walk you through deploying the trading system in 4 sprints:
1. **Sprint 1**: Install dependencies, clone repo, setup Python environment (~15 min)
2. **Sprint 2**: Configure credentials and runtime settings (~10 min)
3. **Sprint 3**: Test and validate everything works (~30 min)
4. **Sprint 4**: Deploy as production systemd service (~20 min)

**Total setup time:** ~75 minutes + paper trading validation time

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Ubuntu 20.04+ server with sudo access
- [ ] Databento API key
- [ ] TradersPost webhook URL (note: TradersPost webhook only, no REST API)
- [ ] Git and Git LFS access
- [ ] Python 3.10 or higher available
- [ ] At least 10GB free disk space
- [ ] Stable internet connection

### Verify Prerequisites

Run these commands to check your system:

```bash
# Check Ubuntu version
lsb_release -a

# Check Python version (must be 3.10+)
python3.10 --version

# Check Git
git --version

# Check available disk space
df -h /opt

# Check you have sudo access
sudo -v && echo "✓ Sudo access confirmed"
```

---

## Initial Setup - Set Your Variables

**Good news:** Your repo URL is pre-configured! Everything below is copy-pastable.

```bash
# Repository URL (pre-configured)
export REPO_URL="https://github.com/jamesrooney7/rooney-capital-v1.git"

# Verify it's set
echo "Repository URL: $REPO_URL"

# Optional: Save to your shell profile so it persists
echo "export REPO_URL=\"$REPO_URL\"" >> ~/.bashrc
```

**What you'll need to manually provide:**
1. ✅ Repository URL (already set above!)
2. ✅ API credentials (you'll edit `.env` file in Sprint 2, step 2.1)

**What's automatic:**
- ✅ Username detection (uses `$USER` automatically)
- ✅ All directory paths
- ✅ All configuration files
- ✅ Service setup

---

## Sprint 1: Initial Setup & Environment

### 1.1 Update System and Install Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3.10 python3.10-venv python3-pip git git-lfs curl

# Verify Python version (should be 3.10+)
python3.10 --version
```

### 1.2 Set Up Git LFS

```bash
# Initialize Git LFS
git lfs install

# Verify installation
git lfs version
```

### 1.3 Create Directory Structure

```bash
# Create base directory
sudo mkdir -p /opt/pine
sudo chown $USER:$USER /opt/pine
cd /opt/pine

# Create runtime directories
mkdir -p /opt/pine/runtime
mkdir -p /opt/pine/logs
mkdir -p /var/run/pine
sudo chown $USER:$USER /var/run/pine
```

### 1.4 Clone Repository

```bash
cd /opt/pine

# Clone repository using the REPO_URL you set earlier
git clone $REPO_URL rooney-capital-v1

cd rooney-capital-v1

# Pull LFS files (ML models)
git lfs pull

# Verify key files exist
echo ""
echo "Verifying repository structure..."
ls -lh requirements.txt
ls -lh Data/Databento_contract_map.yml

# Verify ML models are NOT LFS pointers (critical!)
echo ""
echo "Verifying ML models are actual files (not LFS pointers)..."
MODEL_CHECK=$(head -1 src/models/ES_rf_model.pkl 2>/dev/null)
if echo "$MODEL_CHECK" | grep -q "version https://git-lfs"; then
    echo "✗ ERROR: ML models are still LFS pointers!"
    echo "  Run: git lfs pull"
    exit 1
else
    echo "✓ ML models are actual binary files"
fi

# Count model files (should have pairs: _best.json and _rf_model.pkl)
MODEL_COUNT=$(ls -1 src/models/*_rf_model.pkl 2>/dev/null | wc -l)
JSON_COUNT=$(ls -1 src/models/*_best.json 2>/dev/null | wc -l)
echo "✓ Found $MODEL_COUNT model files (.pkl)"
echo "✓ Found $JSON_COUNT metadata files (.json)"

# Show sample of available models
echo ""
echo "Sample ML models available:"
ls -1 src/models/*.pkl | head -5 | xargs -n 1 basename

echo ""
echo "✓ Repository cloned successfully"
```

### 1.5 Create Python Virtual Environment

```bash
cd /opt/pine/rooney-capital-v1

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 1.6 Install Python Dependencies

```bash
# Still in virtual environment
pip install -r requirements.txt

# This installs:
# - databento (market data)
# - backtrader (trading engine)
# - scikit-learn (ML models)
# - pandas (data processing)
# - requests (HTTP client)
# - pyyaml (config parsing)
# - pytest (testing)
# - and other dependencies

# Verify key packages installed
python -c "import backtrader; import databento; import sklearn; import yaml; print('✓ All dependencies installed successfully')"

# Check versions
echo ""
echo "Key package versions:"
python -c "import databento; print(f'  databento: {databento.__version__}')"
python -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"
python -c "import backtrader; print(f'  backtrader: {backtrader.__version__}')"
```

---

## Sprint 2: Configuration & Credentials

### 2.1 Create Environment File

```bash
cd /opt/pine/runtime

# Create .env file with placeholders
cat > .env << 'EOF'
# Databento Configuration
DATABENTO_API_KEY=REPLACE_WITH_YOUR_DATABENTO_KEY

# TradersPost Configuration (webhook only - no API available)
TRADERSPOST_WEBHOOK_URL=REPLACE_WITH_YOUR_TRADERSPOST_WEBHOOK

# Emergency Kill Switch (set to true to halt all trading)
POLICY_KILLSWITCH=true

# Optional: Commission override
# COMMISSION_PER_SIDE=4.0
EOF

# Secure the file
chmod 600 .env

echo ""
echo "⚠️  IMPORTANT: Now edit .env with your actual credentials"
echo "Run: nano /opt/pine/runtime/.env"
echo ""
echo "Replace these values:"
echo "  - DATABENTO_API_KEY"
echo "  - TRADERSPOST_WEBHOOK_URL"
echo ""
echo "Note: POLICY_KILLSWITCH starts as 'true' for safety (paper trading mode)"
echo ""
read -p "Press Enter after you've updated the credentials..."

# Verify credentials were updated
echo ""
echo "Verifying credentials..."
if grep -q "REPLACE_WITH_YOUR" .env; then
    echo "⚠️  WARNING: Placeholder values still detected in .env"
    echo "   Please update all REPLACE_WITH_YOUR_* values"
else
    echo "✓ Credentials appear to be updated"
fi
```

### 2.2 Create Runtime Configuration

```bash
cd /opt/pine/runtime

# Create minimal runtime configuration file
# (Most defaults come from src/config.py and contract_specs.py)
cat > config.yml << 'EOF'
# Contract metadata location (JSON file, not YAML despite extension)
contract_map: /opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml

# ML models directory
models_path: /opt/pine/rooney-capital-v1/src/models

# Symbols to trade (start small for testing)
symbols:
  - ES
  - NQ

# Databento configuration
databento_api_key: ${DATABENTO_API_KEY}

# TradersPost configuration (webhook only - no REST API available)
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}

# Account settings
starting_cash: 250000

# Data settings
backfill: true
queue_maxsize: 4096

# Heartbeat monitoring
heartbeat_interval: 30
heartbeat_file: /var/run/pine/worker_heartbeat.json
heartbeat_write_interval: 30
poll_interval: 1.0

# Pre-flight validation
preflight:
  enabled: true
  skip_ml_validation: false
  skip_connection_checks: false
  fail_fast: true

# Contract-specific overrides (optional)
# Default commission: 1.25 per side (from src/config.py)
# Default pair mappings: already defined in src/config.py
# Default tick sizes/values: already defined in contract_specs.py
contracts:
  ES:
    size: 1
    # commission: 1.25  # Uncomment to override default
  NQ:
    size: 1
  RTY:
    size: 1  # Change to 2 if you want 2 contracts
  YM:
    size: 1
  GC:
    size: 1
  SI:
    size: 1
  HG:
    size: 1
  CL:
    size: 1
  NG:
    size: 1
  6A:
    size: 1
  6B:
    size: 1
  6E:
    size: 1
EOF

# Secure the file
chmod 600 config.yml

echo "✓ Runtime configuration created"
echo ""
echo "Configuration notes:"
echo "  - Default commission: 1.25 per side (from src/config.py)"
echo "  - Can override with PINE_COMMISSION_PER_SIDE env var"
echo "  - Pair mappings already defined in src/config.py (ES↔NQ, GC↔SI, etc.)"
echo "  - Contract specs (tick size/value) already in contract_specs.py"
echo "  - Reference feeds (TLT, VIX, etc.) already in contract map"
echo ""
echo "Portfolio reconciliation must be done manually via TradersPost console"
echo "or broker statements (TradersPost does not expose a REST API)"
```

### 2.3 Verify Critical Files

```bash
echo "Checking critical system files..."

# Check contract map (NOTE: It's JSON format despite .yml extension!)
if [ -f "/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml" ]; then
    echo "✓ Contract map exists"
    echo "  Location: /opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml"
    echo "  Note: File is JSON format despite .yml extension"
    
    # Verify it's valid JSON
    if python3 -m json.tool /opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml > /dev/null 2>&1; then
        echo "✓ Contract map is valid JSON"
    else
        echo "⚠️  Warning: Contract map may not be valid JSON"
    fi
    
    # Count contracts and reference feeds
    CONTRACT_COUNT=$(python3 -c "import json; data=json.load(open('/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml')); print(len(data.get('contracts', [])))")
    REF_COUNT=$(python3 -c "import json; data=json.load(open('/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml')); print(len(data.get('reference_feeds', [])))")
    echo "  Contains: $CONTRACT_COUNT tradable contracts, $REF_COUNT reference feeds"
    
    # Show some examples
    echo "  Tradable symbols:"
    python3 -c "import json; data=json.load(open('/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml')); print('    ' + ', '.join([c['symbol'] for c in data.get('contracts', [])[:6]]))"
    echo "  Reference symbols:"
    python3 -c "import json; data=json.load(open('/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml')); print('    ' + ', '.join([r['symbol'] for r in data.get('reference_feeds', [])[:6]]))"
else
    echo "✗ Contract map MISSING - cannot proceed"
    exit 1
fi

# Check ML models
MODEL_COUNT=$(ls -1 /opt/pine/rooney-capital-v1/src/models/*_rf_model.pkl 2>/dev/null | wc -l)
if [ $MODEL_COUNT -gt 0 ]; then
    echo "✓ Found $MODEL_COUNT ML model(s)"
    echo "  Models:"
    ls -1 /opt/pine/rooney-capital-v1/src/models/*_rf_model.pkl | xargs -n 1 basename | head -5
    if [ $MODEL_COUNT -gt 5 ]; then
        echo "  ... and $(($MODEL_COUNT - 5)) more"
    fi
else
    echo "✗ No ML models found - did git lfs pull succeed?"
    exit 1
fi

# Check requirements.txt
if [ -f "/opt/pine/rooney-capital-v1/requirements.txt" ]; then
    echo "✓ requirements.txt exists"
else
    echo "✗ requirements.txt MISSING"
    exit 1
fi

# Check src/config.py (contains defaults)
if [ -f "/opt/pine/rooney-capital-v1/src/config.py" ]; then
    echo "✓ src/config.py exists (contains commission and pair map defaults)"
else
    echo "✗ src/config.py MISSING"
    exit 1
fi

# Check contract_specs.py (contains tick sizes/values)
if [ -f "/opt/pine/rooney-capital-v1/src/strategy/contract_specs.py" ]; then
    echo "✓ contract_specs.py exists (contains tick sizes and point values)"
else
    echo "✗ contract_specs.py MISSING"
    exit 1
fi

# Check runner module
if [ -f "/opt/pine/rooney-capital-v1/src/runner/__init__.py" ]; then
    echo "✓ runner module exists"
else
    echo "✗ runner module MISSING"
    exit 1
fi

echo ""
echo "All critical files verified ✓"
echo ""
echo "Configuration defaults are sourced from:"
echo "  - Commission: src/config.py (DEFAULT_COMMISSION_PER_SIDE = 1.25)"
echo "  - Pair mappings: src/config.py (ES↔NQ, GC↔SI, etc.)"
echo "  - Contract specs: src/strategy/contract_specs.py"
echo "  - Contract map: Data/Databento_contract_map.yml (JSON format)"
echo "  - Reference feeds: TLT, VIX, PL, 6C, 6J, 6M, 6N, 6S"
```

---

## Sprint 3: Testing & Validation

### 3.1 Load Environment Variables

```bash
cd /opt/pine/runtime

# Source environment file
set -a
source .env
set +a

# Verify variables loaded
echo "Databento key: ${DATABENTO_API_KEY:0:10}..."
echo "TradersPost webhook: ${TRADERSPOST_WEBHOOK_URL}"
```

### 3.2 Create Test Script

```bash
cd /opt/pine/rooney-capital-v1

cat > test_worker.py << 'EOF'
#!/usr/bin/env python3
"""Test worker launch with dry-run validation."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from runner import LiveWorker, load_runtime_config

def main():
    print("=== Rooney Capital Worker Test ===\n")
    
    # Load configuration
    print("Loading runtime configuration...")
    try:
        config = load_runtime_config()
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1
    
    print(f"✓ Configuration loaded")
    print(f"  Symbols: {config.symbols}")
    print(f"  Starting cash: ${config.starting_cash:,.0f}")
    print(f"  Models path: {config.models_path}")
    print(f"  Killswitch: {config.killswitch}\n")
    
    if config.killswitch:
        print("✓ Killswitch is ENABLED (paper trading mode)")
    else:
        print("⚠️  WARNING: Killswitch is DISABLED (live trading mode)")
    
    # Create worker instance
    print("\nInitializing worker...")
    try:
        worker = LiveWorker(config)
    except Exception as e:
        print(f"✗ Failed to initialize worker: {e}")
        return 1
    
    print("✓ Worker created")
    print(f"  Trading symbols: {', '.join(worker.symbols)}")
    print(f"  Reference feeds: {', '.join(worker.reference_symbols)}")
    print(f"  Total data feeds: {len(worker.data_symbols)}\n")
    
    # Run preflight checks
    print("Running preflight checks...")
    try:
        passed = worker.run_preflight_checks()
    except Exception as e:
        print(f"\n✗ Preflight checks failed with error: {e}")
        return 1
    
    if not passed:
        print("\n✗ Preflight checks failed - see errors above")
        return 1
    
    print("\n=== Test Complete ===")
    print("✓ Worker is ready for launch.")
    print("\nTo start live trading:")
    print("  1. Disable killswitch in /opt/pine/runtime/.env")
    print("  2. Run: python launch_worker.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_worker.py
```

### 3.3 Run Test with Paper Trading Configuration

```bash
cd /opt/pine/rooney-capital-v1

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
set -a
source /opt/pine/runtime/.env
set +a

# Enable kill switch for testing (no real orders)
export POLICY_KILLSWITCH=true

# Run test
python test_worker.py
```

**Expected output:**
- Configuration loads successfully
- Worker initializes
- Preflight checks pass (ML models, connections, reference data)

### 3.4 Create Launch Script

```bash
cd /opt/pine/rooney-capital-v1

cat > launch_worker.py << 'EOF'
#!/usr/bin/env python3
"""Launch the live trading worker."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from runner import LiveWorker, load_runtime_config

def main():
    print("=== Rooney Capital Worker Starting ===\n")
    
    # Load configuration
    config = load_runtime_config()
    
    # Check for kill switch
    if config.killswitch:
        print("⚠️  POLICY KILLSWITCH IS ENABLED")
        print("    No orders will be sent - paper trading mode")
        print("    Set POLICY_KILLSWITCH=false in .env to enable live trading\n")
    else:
        print("🚨 LIVE TRADING MODE - Real orders will be sent")
        print("    Set POLICY_KILLSWITCH=true in .env to enable paper trading\n")
    
    print(f"Trading symbols: {', '.join(config.symbols)}")
    print(f"Starting cash: ${config.starting_cash:,.0f}")
    if config.heartbeat_file:
        print(f"Heartbeat file: {config.heartbeat_file}")
    print()
    
    # Create and run worker
    # Note: run() handles signal handling internally via run_async()
    worker = LiveWorker(config)
    
    print("Starting worker (Ctrl+C to stop gracefully)...\n")
    worker.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

chmod +x launch_worker.py
```

### 3.5 Test Manual Launch (Paper Mode)

```bash
cd /opt/pine/rooney-capital-v1

# Activate virtual environment
source venv/bin/activate

# Set environment
export PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
set -a
source /opt/pine/runtime/.env
set +a

# Keep kill switch enabled for testing
export POLICY_KILLSWITCH=true

# Launch worker (will run until Ctrl+C)
python launch_worker.py

# Let it run for 2-3 minutes, watch for:
# - Successful Databento connection
# - Market data flowing
# - Heartbeat updates
# - No errors in logs

# Press Ctrl+C to stop gracefully
```

---

## Sprint 4: Production Deployment

### 4.1 Create systemd Service File

```bash
# Create service file with automatic username detection
sudo tee /etc/systemd/system/pine-runner.service > /dev/null << EOF
[Unit]
Description=Rooney Capital Trading Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=/opt/pine/rooney-capital-v1

# Environment configuration
EnvironmentFile=/opt/pine/runtime/.env
Environment="PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml"
Environment="PYTHONUNBUFFERED=1"

# Virtual environment
ExecStart=/opt/pine/rooney-capital-v1/venv/bin/python /opt/pine/rooney-capital-v1/launch_worker.py

# Restart configuration
Restart=on-failure
RestartSec=30
StartLimitInterval=350
StartLimitBurst=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=pine-runner

# Process limits
LimitNOFILE=65536

# Security hardening (optional)
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Service file created for user: $USER"
```

### 4.2 Enable and Configure Service

```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable pine-runner.service

# Check service status (should be inactive/dead)
sudo systemctl status pine-runner.service
```

### 4.3 Create Log Rotation

```bash
# Create logrotate config with automatic username
sudo tee /etc/logrotate.d/pine-runner > /dev/null << EOF
/opt/pine/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 $USER $USER
    sharedscripts
    postrotate
        systemctl reload pine-runner.service > /dev/null 2>&1 || true
    endscript
}
EOF

echo "✓ Log rotation configured for user: $USER"
```

### 4.4 Create Monitoring Script

```bash
cat > /opt/pine/monitor_worker.sh << 'EOF'
#!/bin/bash
# Monitor worker health and heartbeat

HEARTBEAT_FILE="/var/run/pine/worker_heartbeat.json"
MAX_AGE_SECONDS=120

echo "=== Pine Worker Monitor ==="
echo

# Check service status
echo "Service Status:"
systemctl is-active pine-runner.service
echo

# Check heartbeat
if [ -f "$HEARTBEAT_FILE" ]; then
    echo "Heartbeat file exists: $HEARTBEAT_FILE"
    
    # Get file age
    FILE_AGE=$(($(date +%s) - $(stat -c %Y "$HEARTBEAT_FILE")))
    echo "Heartbeat age: ${FILE_AGE}s"
    
    if [ $FILE_AGE -gt $MAX_AGE_SECONDS ]; then
        echo "⚠️  WARNING: Heartbeat is stale (>${MAX_AGE_SECONDS}s)"
    else
        echo "✓ Heartbeat is fresh"
    fi
    
    echo
    echo "Latest heartbeat:"
    cat "$HEARTBEAT_FILE" | python3 -m json.tool 2>/dev/null || cat "$HEARTBEAT_FILE"
else
    echo "✗ Heartbeat file not found"
fi

echo
echo "Recent logs (last 20 lines):"
sudo journalctl -u pine-runner.service -n 20 --no-pager
EOF

chmod +x /opt/pine/monitor_worker.sh
```

### 4.5 Pre-Production Checklist

Before starting live trading, verify:

```bash
cd /opt/pine/rooney-capital-v1

# Checklist script
cat > pre_launch_checklist.sh << 'EOF'
#!/bin/bash

echo "=== Pre-Launch Checklist ==="
echo

# Check 1: ML Models
echo "[1/7] Checking ML models..."
MODEL_COUNT=$(ls -1 src/models/*_rf_model.pkl 2>/dev/null | wc -l)
if [ $MODEL_COUNT -gt 0 ]; then
    echo "✓ Found $MODEL_COUNT ML model(s)"
else
    echo "✗ No ML models found"
fi

# Check 2: Configuration
echo "[2/7] Checking configuration..."
if [ -f "/opt/pine/runtime/config.yml" ]; then
    echo "✓ Runtime config exists"
else
    echo "✗ Runtime config missing"
fi

# Check 3: Environment file
echo "[3/7] Checking environment file..."
if [ -f "/opt/pine/runtime/.env" ]; then
    echo "✓ Environment file exists"
else
    echo "✗ Environment file missing"
fi

# Check 4: Contract map
echo "[4/7] Checking contract map..."
if [ -f "Data/Databento_contract_map.yml" ]; then
    echo "✓ Contract map exists"
else
    echo "✗ Contract map missing"
fi

# Check 5: Virtual environment
echo "[5/7] Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "✗ Virtual environment missing"
fi

# Check 6: Service file
echo "[6/7] Checking systemd service..."
if systemctl list-unit-files | grep -q pine-runner.service; then
    echo "✓ Service file installed"
else
    echo "✗ Service file not found"
fi

# Check 7: Kill switch status
echo "[7/7] Checking kill switch..."
source /opt/pine/runtime/.env
if [ "$POLICY_KILLSWITCH" == "true" ]; then
    echo "⚠️  Kill switch is ENABLED (safe mode)"
else
    echo "✓ Kill switch is disabled (live trading enabled)"
fi

echo
echo "=== Checklist Complete ==="
EOF

chmod +x pre_launch_checklist.sh
./pre_launch_checklist.sh
```

### 4.6 Start Production Service (Paper Trading Mode)

```bash
# Verify kill switch is enabled (should already be true from step 2.1)
grep POLICY_KILLSWITCH /opt/pine/runtime/.env

# Should show: POLICY_KILLSWITCH=true

# Start the service
sudo systemctl start pine-runner.service

# Check status
sudo systemctl status pine-runner.service

# Monitor logs in real-time
sudo journalctl -u pine-runner.service -f

# Let it run for several hours, monitoring:
# - Market data connectivity
# - ML model predictions
# - Strategy signals (no actual orders due to kill switch)
# - Heartbeat updates
# - No crashes or errors
```

### 4.7 Enable Live Trading (When Ready)

**⚠️ CRITICAL: Only proceed after thorough paper trading validation**

```bash
# Stop the service
sudo systemctl stop pine-runner.service

# Disable kill switch
sudo sed -i 's/POLICY_KILLSWITCH=true/POLICY_KILLSWITCH=false/' /opt/pine/runtime/.env

# Verify change
grep POLICY_KILLSWITCH /opt/pine/runtime/.env

# Start service with live trading enabled
sudo systemctl start pine-runner.service

# Monitor closely
sudo journalctl -u pine-runner.service -f
```

### 4.8 Common Operations

```bash
# View live logs
sudo journalctl -u pine-runner.service -f

# View logs from last hour
sudo journalctl -u pine-runner.service --since "1 hour ago"

# Restart service (after config changes)
sudo systemctl restart pine-runner.service

# Stop service
sudo systemctl stop pine-runner.service

# Check service status
sudo systemctl status pine-runner.service

# View heartbeat
cat /var/run/pine/worker_heartbeat.json | python3 -m json.tool

# Emergency stop (enable kill switch without restart)
echo "POLICY_KILLSWITCH=true" | sudo tee /opt/pine/runtime/.env
sudo systemctl restart pine-runner.service

# Update code
cd /opt/pine/rooney-capital-v1
git pull
git lfs pull
sudo systemctl restart pine-runner.service
```

---

## Emergency Procedures

### Emergency Stop

```bash
# Method 1: Kill switch (graceful, preferred)
sudo sed -i 's/POLICY_KILLSWITCH=false/POLICY_KILLSWITCH=true/' /opt/pine/runtime/.env
sudo systemctl restart pine-runner.service

# Method 2: Stop service immediately
sudo systemctl stop pine-runner.service

# Method 3: Kill process (last resort)
sudo pkill -f launch_worker.py
```

### Troubleshooting

```bash
# Check recent errors
sudo journalctl -u pine-runner.service -p err -n 50

# Check service won't start
sudo systemctl status pine-runner.service -l
sudo journalctl -xe

# Check file permissions
ls -la /opt/pine/runtime/
ls -la /var/run/pine/

# Test configuration manually
cd /opt/pine/rooney-capital-v1
source venv/bin/activate
export PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
python test_worker.py

# Check Databento connectivity
ping api.databento.com

# Verify TradersPost webhook (basic connectivity test)
# Note: Only webhook is available - no REST API
curl -X POST "${TRADERSPOST_WEBHOOK_URL}" \
  -H "Content-Type: application/json" \
  -d '{"test": true}'
```

#### Common Issues

**Issue: ML models not found**
```bash
# Verify Git LFS is installed
git lfs version

# Re-pull LFS files
cd /opt/pine/rooney-capital-v1
git lfs pull

# Check models exist
ls -lh src/models/*.pkl
```

**Issue: Import errors (ModuleNotFoundError)**
```bash
# Verify virtual environment is activated
which python  # Should show /opt/pine/rooney-capital-v1/venv/bin/python

# Reinstall dependencies
cd /opt/pine/rooney-capital-v1
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Issue: Contract map not found**
```bash
# Verify file exists
ls -lh /opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml

# Check it's readable
cat /opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml | head -20
```

**Issue: Permission denied on heartbeat file**
```bash
# Fix permissions on /var/run/pine
sudo chown -R $USER:$USER /var/run/pine
sudo chmod 755 /var/run/pine
```

**Issue: Environment variables not loading**
```bash
# Check .env file exists and is readable
cat /opt/pine/runtime/.env

# Verify no placeholder values remain
grep "REPLACE_WITH_YOUR" /opt/pine/runtime/.env

# Check systemd service has EnvironmentFile directive
sudo cat /etc/systemd/system/pine-runner.service | grep EnvironmentFile
```

---

## Post-Launch Monitoring

### Daily Checks

```bash
# Run daily monitoring
/opt/pine/monitor_worker.sh

# Check for any errors
sudo journalctl -u pine-runner.service --since today -p err

# Verify positions in TradersPost dashboard
# (manual check via web interface)
```

### Weekly Tasks

- Review trade logs and performance metrics
- Check disk space: `df -h /opt/pine`
- Update system packages: `sudo apt update && sudo apt upgrade`
- Verify ML model performance against expectations
- Review and rotate logs if needed

---

## Success Indicators

Your system is running correctly when you see:

✓ Service status shows "active (running)"  
✓ Heartbeat file updates every 30 seconds  
✓ Market data flowing without gaps  
✓ ML model predictions being generated  
✓ Strategy signals appearing in logs  
✓ Orders posting to TradersPost (when kill switch disabled)  
✓ No connection errors or timeouts  
✓ Memory usage stable over time  

---

## Configuration Tuning

### Configuration Tuning

### Adjust Trading Universe

```bash
# Edit runtime config
nano /opt/pine/runtime/config.yml

# Modify symbols list to trade more instruments
symbols:
  - ES
  - NQ
  - RTY
  - YM
  - GC

# Save and restart
sudo systemctl restart pine-runner.service
```

### Adjust Position Sizes

```bash
# Edit contract settings in config.yml
nano /opt/pine/runtime/config.yml

# Find contracts section and adjust size:
contracts:
  ES:
    size: 2  # Changed from 1 to 2
  NQ:
    size: 2  # Trade 2 contracts instead of 1

# Save and restart
sudo systemctl restart pine-runner.service
```

### Override Default Commission

```bash
# Method 1: Environment variable (recommended)
echo "PINE_COMMISSION_PER_SIDE=1.50" >> /opt/pine/runtime/.env
sudo systemctl restart pine-runner.service

# Method 2: Per-contract override in config.yml
nano /opt/pine/runtime/config.yml
# Add under specific contract:
# contracts:
#   ES:
#     size: 1
#     commission: 1.50  # Override default 1.25

# Default commission (from src/config.py): 1.25 per side
```

### Verify Configuration Changes

```bash
# Test configuration loads without errors
cd /opt/pine/rooney-capital-v1
source venv/bin/activate
export PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
set -a && source /opt/pine/runtime/.env && set +a
python test_worker.py

# If successful, restart live service
sudo systemctl restart pine-runner.service
```

---

## Notes

- Always test configuration changes in paper mode first (kill switch enabled)
- Keep at least 30 days of logs for compliance and debugging
- Monitor disk space in `/opt/pine/logs` and `/var/run/pine`
- Backup your configuration files regularly
- Document any custom modifications to the strategy
- Review TradersPost execution reports daily
- Set up external monitoring (PagerDuty, etc.) for production alerts

---

## Pre-Go-Live Checklist

Before enabling live trading (disabling kill switch), verify ALL items:

### Configuration
- [ ] `.env` file has real credentials (no REPLACE_WITH_YOUR placeholders)
- [ ] `config.yml` references correct paths
- [ ] Contract map exists at `/opt/pine/rooney-capital-v1/Data/Databento_contract_map.yml`
  - Note: File is JSON format despite .yml extension
- [ ] All ML models present (check with `ls /opt/pine/rooney-capital-v1/src/models/*.pkl`)
- [ ] Default commission rate (1.25 per side from `src/config.py`) is acceptable for your broker
  - Override with `PINE_COMMISSION_PER_SIDE` env var if needed
- [ ] Position sizes in `config.yml` are appropriate for account size
- [ ] Pair mappings in `src/config.py` match your trading strategy (ES↔NQ, GC↔SI, etc.)
- [ ] Contract map contains all needed symbols and reference feeds
  - Verify with: `python3 -m json.tool Data/Databento_contract_map.yml`

### Testing
- [ ] Paper trading ran for at least 24 hours without crashes
- [ ] Heartbeat file updates every 30 seconds
- [ ] No connection errors in logs
- [ ] ML models generating predictions successfully
- [ ] Strategy signals appearing in logs
- [ ] Webhook sends reaching TradersPost (check TradersPost console)
- [ ] Test script passes all preflight checks

### Infrastructure
- [ ] Service starts automatically: `systemctl is-enabled pine-runner.service`
- [ ] Sufficient disk space (at least 5GB free)
- [ ] Log rotation configured
- [ ] Monitoring script works: `/opt/pine/monitor_worker.sh`
- [ ] Emergency stop procedure tested

### TradersPost Configuration
- [ ] TradersPost strategies configured for correct contracts
- [ ] Contract expiry dates are current (manual check in TradersPost)
- [ ] Broker account connected and funded
- [ ] Webhook URL is correct and active
- [ ] Test orders appeared correctly during paper trading

### Safety
- [ ] Emergency stop procedure documented and accessible
- [ ] Kill switch tested and working
- [ ] Alerts configured for system failures
- [ ] Contact information for Databento/TradersPost support saved
- [ ] Backup plan for server failure

### Final Steps
1. Run full test suite: `cd /opt/pine/rooney-capital-v1 && python -m pytest tests/`
2. Monitor paper trading for 1-2 days during market hours
3. Review all logs for warnings or errors
4. Double-check position sizing won't over-leverage account
5. Verify TradersPost shows test orders from paper trading
6. Have emergency contact numbers ready

**Only after ALL boxes are checked, proceed to step 4.7 to enable live trading.**

---

**Last Updated:** October 2025  
**System Version:** v1.0  
**Repository:** https://github.com/jamesrooney7/rooney-capital-v1.git

---

## Quick Reference Card

### Key Paths
```bash
# Installation directory
/opt/pine/rooney-capital-v1/

# Configuration files
/opt/pine/runtime/config.yml
/opt/pine/runtime/.env

# Heartbeat file
/var/run/pine/worker_heartbeat.json

# Logs
/opt/pine/logs/
sudo journalctl -u pine-runner.service -f

# Virtual environment
/opt/pine/rooney-capital-v1/venv/
```

### Key Commands
```bash
# Start service
sudo systemctl start pine-runner.service

# Stop service
sudo systemctl stop pine-runner.service

# Restart service
sudo systemctl restart pine-runner.service

# View logs
sudo journalctl -u pine-runner.service -f

# Check status
sudo systemctl status pine-runner.service

# Monitor health
/opt/pine/monitor_worker.sh

# Emergency stop
sudo sed -i 's/POLICY_KILLSWITCH=false/POLICY_KILLSWITCH=true/' /opt/pine/runtime/.env
sudo systemctl restart pine-runner.service
```

### Environment Variables
```bash
# Required in .env file:
DATABENTO_API_KEY=your_key_here
TRADERSPOST_WEBHOOK_URL=your_webhook_here
POLICY_KILLSWITCH=true  # true for paper trading, false for live

# Auto-set by system:
PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml

# Optional overrides (defaults come from src/config.py):
# PINE_COMMISSION_PER_SIDE=1.25  # Default is 1.25 per side
# COMMISSION_PER_SIDE=1.25       # Alternative name

# Note: TradersPost does not have a REST API
# Portfolio reconciliation must be done manually via:
#   - TradersPost web console
#   - Tradovate broker statements
```

### Default Configuration Sources
```bash
# Most settings come from repository files:
# - Commission rate: src/config.py (DEFAULT_COMMISSION_PER_SIDE = 1.25)
# - Pair mappings: src/config.py (DEFAULT_PAIR_MAP: ES↔NQ, GC↔SI, CL↔NG, etc.)
# - Contract specs: src/strategy/contract_specs.py (tick sizes, tick values, point values)
# - Contract map: Data/Databento_contract_map.yml (JSON format despite extension!)
#   * Databento dataset mappings (GLBX.MDP3)
#   * Tradovate symbol mappings (MES, MNQ, etc.)
#   * Product IDs for Databento feeds
# - Reference feeds: Data/Databento_contract_map.yml
#   * TLT (bonds), VIX (volatility), PL (platinum)
#   * Currency pairs: 6C, 6J, 6M, 6N, 6S

# config.yml only needs to specify:
# - Which symbols to trade (subset of available contracts)
# - Position sizes per contract
# - Credential references (databento_api_key, traderspost_webhook)
# - Monitoring settings (heartbeat_file, etc.)
# - Optional per-contract overrides
```
