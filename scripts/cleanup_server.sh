#!/bin/bash
# Server Cleanup Script - Configuration B Migration
# This script consolidates duplicate configs and cleans up old files

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "🧹 SERVER CLEANUP - Configuration B Migration"
echo "================================================================================"
echo ""

# Confirm before proceeding
echo -e "${YELLOW}This script will:${NC}"
echo "  1. Archive old config files from /opt/pine/runtime/"
echo "  2. Remove duplicate rooney-trading.service"
echo "  3. Remove bash version of reset_dashboard.sh (keep Python version)"
echo "  4. Update pine-runner.service to use correct config paths"
echo "  5. Create .gitignore entries for ML models and analysis files"
echo ""
read -p "Continue? (yes/no) " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cleanup cancelled"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 1: Archive old runtime configs"
echo "================================================================================"

ARCHIVE_DIR="/opt/pine/runtime/archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

if [ -f "/opt/pine/runtime/config.yml" ]; then
    echo "Moving /opt/pine/runtime/config.yml to archive..."
    mv /opt/pine/runtime/config.yml "$ARCHIVE_DIR/"
    echo -e "${GREEN}✓${NC} Archived old config.yml"
fi

if [ -f "/opt/pine/runtime/.env" ]; then
    echo "Moving /opt/pine/runtime/.env to archive..."
    mv /opt/pine/runtime/.env "$ARCHIVE_DIR/"
    echo -e "${GREEN}✓${NC} Archived old .env"
fi

echo -e "${GREEN}✓${NC} Old configs archived to: $ARCHIVE_DIR"
echo ""

echo "================================================================================"
echo "Step 2: Remove duplicate systemd service"
echo "================================================================================"

if systemctl list-unit-files | grep -q "rooney-trading.service"; then
    echo "Removing rooney-trading.service..."
    sudo systemctl stop rooney-trading.service 2>/dev/null || true
    sudo systemctl disable rooney-trading.service 2>/dev/null || true
    sudo rm -f /etc/systemd/system/rooney-trading.service
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓${NC} Removed rooney-trading.service"
else
    echo "rooney-trading.service already removed"
fi

echo ""

echo "================================================================================"
echo "Step 3: Remove duplicate dashboard reset script"
echo "================================================================================"

if [ -f "/opt/pine/rooney-capital-v1/scripts/reset_dashboard.sh" ]; then
    echo "Removing bash version of reset_dashboard.sh (keeping Python version)..."
    rm /opt/pine/rooney-capital-v1/scripts/reset_dashboard.sh
    echo -e "${GREEN}✓${NC} Removed reset_dashboard.sh (bash version)"
fi

echo ""

echo "================================================================================"
echo "Step 4: Update pine-runner.service paths"
echo "================================================================================"

echo "Creating backup of pine-runner.service..."
sudo cp /etc/systemd/system/pine-runner.service /etc/systemd/system/pine-runner.service.backup

echo "Updating pine-runner.service to use correct config paths..."
sudo sed -i 's|EnvironmentFile=/opt/pine/runtime/.env|EnvironmentFile=/opt/pine/rooney-capital-v1/.env|g' /etc/systemd/system/pine-runner.service
sudo sed -i 's|PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml|PINE_RUNTIME_CONFIG=/opt/pine/rooney-capital-v1/config.yml|g' /etc/systemd/system/pine-runner.service

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo -e "${GREEN}✓${NC} Updated pine-runner.service"
echo ""

echo "================================================================================"
echo "Step 5: Update .gitignore for ML models and analysis"
echo "================================================================================"

cd /opt/pine/rooney-capital-v1

# Create/update .gitignore
cat >> .gitignore << 'EOF'

# Analysis results
analysis/
greedy_optimizer_log.txt

# ML Models (too large for git, should be deployed separately)
src/models/*_test_results.json
src/models/*_best.json
src/models/*.pkl
src/models_OLD_*/

# Archives
*.tar.gz

# Runtime files
live_worker.pid
*.pid

# Database files (runtime data, not code)
*.db
*.db-journal
EOF

echo -e "${GREEN}✓${NC} Updated .gitignore"
echo ""

echo "================================================================================"
echo "✅ CLEANUP COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ Old configs archived to: $ARCHIVE_DIR"
echo "  ✓ Removed rooney-trading.service"
echo "  ✓ Removed duplicate reset_dashboard.sh"
echo "  ✓ Updated pine-runner.service config paths"
echo "  ✓ Updated .gitignore"
echo ""
echo "Next steps:"
echo "  1. Review git status: git status"
echo "  2. Start pine-runner: sudo systemctl start pine-runner"
echo "  3. Verify all 16 instruments load: sudo journalctl -u pine-runner -f | grep 'Model loaded'"
echo ""
echo "Single source of truth for configs:"
echo "  • Config: /opt/pine/rooney-capital-v1/config.yml (16 instruments)"
echo "  • Env: /opt/pine/rooney-capital-v1/.env"
echo "  • Service: pine-runner.service"
echo ""
