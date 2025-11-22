#!/bin/bash
################################################################################
# Remote Progress Monitor
# Check status of multi-instrument run
################################################################################

BASE_DIR="/opt/pine/rooney-capital-v1"
LOG_DIR="${BASE_DIR}/logs/strategy_factory"

echo "================================================================================"
echo "STRATEGY FACTORY PROGRESS MONITOR"
echo "================================================================================"
echo "Time: $(date)"
echo ""

# Check if process is running
if pgrep -f "run_all_instruments.sh" > /dev/null; then
    echo "Status: 🟢 RUNNING"
    PID=$(pgrep -f "run_all_instruments.sh")
    echo "PID: ${PID}"
else
    echo "Status: 🔴 NOT RUNNING (either completed or not started)"
fi

echo ""
echo "================================================================================"
echo "COMPLETED INSTRUMENTS"
echo "================================================================================"

# Count completed instruments by checking for success markers in logs
for symbol in ES NQ YM RTY GC SI HG CL NG 6A 6B 6E 6J 6N 6S; do
    latest_log=$(ls -t ${LOG_DIR}/${symbol}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        if grep -q "PHASE 1 COMPLETE" "$latest_log" 2>/dev/null; then
            winners=$(grep "FINAL WINNERS:" "$latest_log" | tail -1)
            echo "✅ ${symbol}: COMPLETE - ${winners}"
        elif grep -q "Failed to optimize" "$latest_log" 2>/dev/null; then
            echo "❌ ${symbol}: FAILED"
        else
            echo "🔄 ${symbol}: IN PROGRESS"
            # Show current strategy being processed
            current_strategy=$(grep "Optimizing Strategy" "$latest_log" | tail -1)
            if [ -n "$current_strategy" ]; then
                echo "   ${current_strategy}"
            fi
        fi
    fi
done

echo ""
echo "================================================================================"
echo "OVERALL LOG"
echo "================================================================================"

# Show last 20 lines of overall log
latest_overall=$(ls -t ${LOG_DIR}/all_instruments_*.log 2>/dev/null | head -1)
if [ -n "$latest_overall" ]; then
    echo "File: ${latest_overall}"
    echo ""
    tail -20 "$latest_overall"
else
    echo "No overall log found yet."
fi

echo ""
echo "================================================================================"
echo "QUICK STATS"
echo "================================================================================"

# Database stats
if [ -f "${BASE_DIR}/research/strategy_factory/results/strategy_factory.db" ]; then
    db_size=$(du -h "${BASE_DIR}/research/strategy_factory/results/strategy_factory.db" | cut -f1)
    echo "Database size: ${db_size}"
fi

# Disk space
echo "Disk space remaining:"
df -h "${BASE_DIR}" | tail -1

echo ""
