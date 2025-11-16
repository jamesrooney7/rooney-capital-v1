#!/bin/bash
# Summary script for three-way split optimization results

MODELS_DIR="/opt/pine/rooney-capital-v1/src/models"

echo "========================================="
echo "Three-Way Split Optimization Results"
echo "========================================="
echo ""

# Check if any results exist
if ! ls "$MODELS_DIR"/*_test_results.json 1> /dev/null 2>&1; then
    echo "No test results found in $MODELS_DIR"
    exit 1
fi

# Print header
printf "%-6s %12s %13s %14s %10s %8s\n" \
    "Symbol" "Test_Sharpe" "Test_Sortino" "Profit_Factor" "Win_Rate" "Trades"
echo "--------------------------------------------------------------------------------"

# Create temporary file for sorting
temp_file=$(mktemp)

# Collect all results
for file in "$MODELS_DIR"/*_test_results.json; do
    symbol=$(basename "$file" | sed 's/_test_results.json//')

    # Extract metrics from nested test_metrics object
    python3 << EOF >> "$temp_file"
import json
try:
    with open('$file') as f:
        d = json.load(f)
        metrics = d.get('test_metrics', {})
        sharpe = metrics.get('sharpe', 0)
        sortino = metrics.get('sortino', 0)
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate', 0) * 100  # Convert to percentage
        trades = metrics.get('trades', 0)
        # Format: sharpe|symbol|sortino|pf|wr|trades
        print(f'{sharpe:.3f}|$symbol|{sortino:.3f}|{pf:.2f}|{wr:.1f}|{trades}')
except Exception as e:
    print(f'0.000|$symbol|0.000|0.00|0.0|0')
EOF
done

# Sort by sharpe (first field) in reverse numeric order and display
sort -t'|' -k1 -rn "$temp_file" | while IFS='|' read sharpe symbol sortino pf wr trades; do
    printf "%-6s %12s %13s %14s %9s%% %8s\n" \
        "$symbol" "$sharpe" "$sortino" "$pf" "$wr" "$trades"
done

# Cleanup
rm -f "$temp_file"

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Summary statistics
echo "Summary Statistics:"
python3 << 'PYEOF'
import json
import glob

files = glob.glob('/opt/pine/rooney-capital-v1/src/models/*_test_results.json')
sharpes = []
total_trades = 0

for file in files:
    with open(file) as f:
        d = json.load(f)
        metrics = d.get('test_metrics', {})
        sharpes.append(metrics.get('sharpe', 0))
        total_trades += metrics.get('trades', 0)

if sharpes:
    print(f"  Total Symbols: {len(sharpes)}")
    print(f"  Average Test Sharpe: {sum(sharpes)/len(sharpes):.3f}")
    print(f"  Best Test Sharpe: {max(sharpes):.3f}")
    print(f"  Worst Test Sharpe: {min(sharpes):.3f}")
    print(f"  Total Trades (all symbols, test period): {total_trades:,}")
PYEOF

echo ""
echo "To view detailed results for a symbol:"
echo "  cat $MODELS_DIR/ES_test_results.json | python3 -m json.tool"
echo ""
