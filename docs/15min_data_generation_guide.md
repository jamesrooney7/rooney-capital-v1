# 15-Minute Bar Data Generation Guide

## Overview

The data resampling script has been updated to generate 15-minute bars alongside the existing hourly and daily bars. This follows the same process and architecture used for the 60m and 1D bar generation.

## Updated Script

**Location**: `research/utils/resample_data.py`

**New Functionality**:
- `resample_to_15min()` - Resamples 1m bars to 15-minute OHLCV bars
- Output files: `{SYMBOL}_15min.csv` in `data/resampled/`
- Follows same pattern as hourly/daily with `label='left', closed='left'`

## Usage

### Process All Symbols

```bash
# Generate 15min, hourly, and daily bars for all symbols
python research/utils/resample_data.py --all

# Generate ONLY 15min bars (skip hourly and daily if you already have them)
python research/utils/resample_data.py --all --only-15min

# With date range filter
python research/utils/resample_data.py --all --start-date 2010-01-01 --end-date 2024-12-31 --only-15min
```

### Process Single Symbol

```bash
# Generate bars for ES only
python research/utils/resample_data.py --symbol ES --input data/historical/ES_bt.csv

# Generate ONLY 15min bars for ES
python research/utils/resample_data.py --symbol ES --input data/historical/ES_bt.csv --only-15min

# With custom output directory
python research/utils/resample_data.py --symbol ES --input data/historical/ES_bt.csv --output-dir data/resampled
```

## Output Files

After running the script, you'll have three files per symbol in `data/resampled/`:

```
data/resampled/
├── ES_15min.csv    # NEW: 15-minute bars
├── ES_hourly.csv   # Existing: Hourly bars
├── ES_daily.csv    # Existing: Daily bars
├── NQ_15min.csv
├── NQ_hourly.csv
├── NQ_daily.csv
└── ...
```

## Expected Bar Counts (2010-2024, ~14 years)

Assuming 1-minute input data:

| Timeframe | Bars per Day | Approximate Total Bars (14 years) |
|-----------|--------------|-----------------------------------|
| 1-minute  | ~1,440       | ~7.3M                             |
| 15-minute | ~96          | ~490K                             |
| 1-hour    | ~24          | ~123K                             |
| 1-day     | 1            | ~5,110                            |

## Data Structure

All output CSV files have the same format:

```csv
datetime,Open,High,Low,Close,volume
2010-01-04 18:00:00,1132.25,1133.00,1131.75,1132.50,12345
2010-01-04 18:15:00,1132.50,1133.25,1132.00,1133.00,23456
...
```

- **datetime**: Pandas datetime index (UTC or market timezone)
- **OHLCV**: Standard open, high, low, close, volume
- **15min bars**: Each bar starts at :00, :15, :30, :45 minutes
- **Session alignment**: Respects 24-hour futures sessions (same as hourly/daily)

## Running on Your Server

### Prerequisites

```bash
# Ensure you're in the project directory
cd /path/to/rooney-capital-v1

# Activate virtual environment
source venv/bin/activate

# Verify pandas is installed
pip list | grep pandas
```

### Execution

```bash
# Navigate to project root
cd /path/to/rooney-capital-v1

# If you already have hourly and daily data, use --only-15min flag
python research/utils/resample_data.py --all --only-15min

# Or process all timeframes (if starting fresh)
python research/utils/resample_data.py --all
```

### Expected Output

**With `--only-15min` flag** (recommended if you already have hourly/daily):
```
2025-01-20 14:30:22 - INFO - Processing ES...
2025-01-20 14:30:22 - INFO - Reading data/historical/ES_bt.csv...
2025-01-20 14:30:45 - INFO - Loaded 7,234,891 tick/minute bars from 2010-01-04 to 2024-12-31
2025-01-20 14:30:45 - INFO - Resampling to 15-minute bars...
2025-01-20 14:30:52 - INFO - Created 489,234 15-minute bars
2025-01-20 14:30:52 - INFO - Saving 15-minute bars to data/resampled/ES_15min.csv...
2025-01-20 14:30:54 - INFO - ✅ ES complete: 489,234 15min bars only

[Repeats for NQ, YM, RTY, etc.]
```

**Without flag** (generates all timeframes):
```
2025-01-20 14:30:22 - INFO - Processing ES...
2025-01-20 14:30:22 - INFO - Reading data/historical/ES_bt.csv...
2025-01-20 14:30:45 - INFO - Loaded 7,234,891 tick/minute bars from 2010-01-04 to 2024-12-31
2025-01-20 14:30:45 - INFO - Resampling to 15-minute bars...
2025-01-20 14:30:52 - INFO - Created 489,234 15-minute bars
2025-01-20 14:30:52 - INFO - Resampling to hourly bars...
2025-01-20 14:30:58 - INFO - Created 122,308 hourly bars
2025-01-20 14:30:58 - INFO - Resampling to daily bars...
2025-01-20 14:31:03 - INFO - Created 5,110 daily bars
2025-01-20 14:31:03 - INFO - Saving 15-minute bars to data/resampled/ES_15min.csv...
2025-01-20 14:31:05 - INFO - Saving hourly bars to data/resampled/ES_hourly.csv...
2025-01-20 14:31:06 - INFO - Saving daily bars to data/resampled/ES_daily.csv...
2025-01-20 14:31:06 - INFO - ✅ ES complete: 489,234 15min, 122,308 hourly, 5,110 daily bars

[Repeats for NQ, YM, RTY, etc.]
```

## Verification

After generation, verify the output:

```bash
# Check files exist
ls -lh data/resampled/*_15min.csv

# Verify bar counts
wc -l data/resampled/ES_15min.csv

# Check first few rows
head -10 data/resampled/ES_15min.csv

# Check date range
head -2 data/resampled/ES_15min.csv
tail -2 data/resampled/ES_15min.csv
```

## Performance Considerations

- **Memory**: Script loads entire symbol history into memory (pandas DataFrame)
- **For large files**: Use `--start-date` and `--end-date` to process in chunks
- **Disk space**: 15min bars will be ~2x the size of hourly bars

**Estimated processing time** (per symbol, 14 years of 1m data):
- ES: ~30-60 seconds
- All symbols (20+): ~15-30 minutes total

## Integration with Strategy Optimizer

Once generated, the 15-minute bars can be used for:

1. **Higher-frequency strategy testing** (opening range breakouts, intraday patterns)
2. **More granular entry/exit timing** (vs hourly bars)
3. **Multi-timeframe strategies** (15m signals, hourly filters, daily regime)

The strategy optimizer tool will be able to load these bars using the existing data loader infrastructure.

## Troubleshooting

### "Input file not found"
- Verify 1-minute data exists: `ls data/historical/*_bt.csv`
- Check file naming convention matches `{SYMBOL}_bt.csv`

### "Out of memory"
- Process symbols one at a time instead of `--all`
- Use date range filters to process in smaller chunks
- Increase available memory on server

### "Empty output file"
- Check date range filters (start/end date)
- Verify input data has valid datetime index
- Check for timezone issues

## Next Steps

After generating 15-minute bars:

1. ✅ Verify output files exist and have correct bar counts
2. Update `research/utils/data_loader.py` to support loading 15min bars (if needed)
3. Build strategies that can use 15min timeframe
4. Test strategies on 15min bars before running full optimizer

---

**Created**: 2025-01-20
**Script Version**: Updated with 15min support
**Documentation**: research/README.md
