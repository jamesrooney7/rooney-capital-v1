#!/bin/bash
# Verify that data extraction is complete and files are ready for optimization

set -e

DATA_DIR="data/training"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Verifying Extraction Completion${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check 1: Are extraction processes still running?
echo "1. Checking for running extraction processes..."
running_count=$(ps aux | grep -E "extract_training_data.py|FeatureLoggingStrategy" | grep -v grep | wc -l)

if [ $running_count -gt 0 ]; then
    echo -e "${YELLOW}   ⚠ WARNING: $running_count extraction process(es) still running${NC}"
    echo ""
    ps aux | grep -E "extract_training_data.py" | grep -v grep
    echo ""
    echo -e "${YELLOW}   Wait for these to complete before running optimization.${NC}"
else
    echo -e "${GREEN}   ✓ No extraction processes running${NC}"
fi
echo ""

# Check 2: Do transformed_features.csv files exist?
echo "2. Checking for transformed_features.csv files..."
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}   ✗ ERROR: $DATA_DIR directory does not exist!${NC}"
    echo "   Create it with: mkdir -p $DATA_DIR"
    exit 1
fi

csv_files=($(find "$DATA_DIR" -name "*_transformed_features.csv" -type f 2>/dev/null))
csv_count=${#csv_files[@]}

if [ $csv_count -eq 0 ]; then
    echo -e "${RED}   ✗ ERROR: No *_transformed_features.csv files found in $DATA_DIR${NC}"
    echo "   Run extract_training_data.py first."
    exit 1
else
    echo -e "${GREEN}   ✓ Found $csv_count CSV file(s)${NC}"
fi
echo ""

# Check 3: File details
echo "3. File Summary:"
echo ""
printf "%-10s %-15s %-15s %-15s %s\n" "Symbol" "Rows" "Columns" "Size" "Status"
printf "%-10s %-15s %-15s %-15s %s\n" "------" "----" "-------" "----" "------"

total_rows=0
min_rows=999999999
max_rows=0
issues=0

for csv_file in "${csv_files[@]}"; do
    symbol=$(basename "$csv_file" | sed 's/_transformed_features.csv//')

    # Check if file is empty or incomplete
    if [ ! -s "$csv_file" ]; then
        printf "%-10s %-15s %-15s %-15s %s\n" "$symbol" "0" "0" "0 B" "❌ EMPTY"
        issues=$((issues + 1))
        continue
    fi

    # Get file size
    size=$(du -h "$csv_file" | cut -f1)

    # Get row count (subtract 1 for header)
    rows=$(wc -l < "$csv_file" | xargs)
    rows=$((rows - 1))

    # Get column count
    cols=$(head -1 "$csv_file" | awk -F',' '{print NF}')

    # Track statistics
    total_rows=$((total_rows + rows))
    if [ $rows -lt $min_rows ]; then min_rows=$rows; fi
    if [ $rows -gt $max_rows ]; then max_rows=$rows; fi

    # Status check
    status="✅"
    if [ $rows -lt 1000 ]; then
        status="⚠️  LOW"
        issues=$((issues + 1))
    fi
    if [ $cols -lt 50 ]; then
        status="⚠️  FEW COLS"
        issues=$((issues + 1))
    fi

    printf "%-10s %-15s %-15s %-15s %s\n" "$symbol" "$rows" "$cols" "$size" "$status"
done

echo ""
echo "Statistics:"
echo "  Total CSV files: $csv_count"
echo "  Total trades: $total_rows"
echo "  Min trades per symbol: $min_rows"
echo "  Max trades per symbol: $max_rows"
if [ $csv_count -gt 0 ]; then
    avg_rows=$((total_rows / csv_count))
    echo "  Avg trades per symbol: $avg_rows"
fi
echo ""

# Check 4: Inspect a sample file
if [ $csv_count -gt 0 ]; then
    echo "4. Sample File Inspection (first file):"
    sample_file="${csv_files[0]}"
    sample_symbol=$(basename "$sample_file" | sed 's/_transformed_features.csv//')

    echo "   File: $sample_symbol"
    echo ""
    echo "   First 5 columns:"
    head -1 "$sample_file" | cut -d',' -f1-5
    echo ""
    echo "   Required columns check:"

    # Check for critical columns
    header=$(head -1 "$sample_file")

    check_column() {
        local col=$1
        if echo "$header" | grep -q "$col"; then
            echo -e "      ${GREEN}✓${NC} $col"
        else
            echo -e "      ${RED}✗${NC} $col (MISSING!)"
            issues=$((issues + 1))
        fi
    }

    check_column "Date/Time"
    check_column "Symbol"
    check_column "Return"
    check_column "Binary_Label"

    echo ""
fi

# Check 5: Final verdict
echo "========================================${NC}"
if [ $running_count -gt 0 ]; then
    echo -e "${YELLOW}STATUS: EXTRACTION STILL RUNNING${NC}"
    echo ""
    echo "Wait for extraction to complete, then run this script again."
    exit 1
elif [ $csv_count -eq 0 ]; then
    echo -e "${RED}STATUS: NO DATA EXTRACTED${NC}"
    echo ""
    echo "Run extract_training_data.py to generate CSV files first."
    exit 1
elif [ $issues -gt 0 ]; then
    echo -e "${YELLOW}STATUS: READY WITH WARNINGS${NC}"
    echo ""
    echo "Found $issues potential issue(s) (check LOW or FEW COLS warnings above)."
    echo "You can still proceed with optimization, but review the warnings."
    echo ""
    echo "To run optimization:"
    echo "  ./scripts/parallel_optimization.sh"
else
    echo -e "${GREEN}STATUS: ✅ READY FOR OPTIMIZATION!${NC}"
    echo ""
    echo "All checks passed. $csv_count symbol(s) ready."
    echo ""
    echo "Next step:"
    echo "  ./scripts/parallel_optimization.sh"
fi
echo "========================================${NC}"
