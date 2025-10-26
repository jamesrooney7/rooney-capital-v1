# Research & Backtesting

This directory contains tools for backtesting using the **production IbsStrategy code**, ensuring parity between research and live trading.

## Overview

The research framework has three main components:

1. **Data Resampling** (`utils/resample_data.py`) - Convert tick data to hourly + daily bars
2. **Data Loading** (`utils/data_loader.py`) - Load resampled data into Backtrader
3. **Backtest Runner** (`backtest_runner.py`) - Run production strategy on historical data

## Quick Start

### Step 1: Resample Your Data

First, convert tick data to the hourly and daily timeframes needed by IbsStrategy:

```bash
# Resample all symbols
python research/utils/resample_data.py --all

# Or resample a single symbol
python research/utils/resample_data.py --symbol ES --input data/historical/ES_bt.csv

# Resample with date range filter
python research/utils/resample_data.py --all --start-date 2020-01-01 --end-date 2024-12-31
```

This creates:
- `data/resampled/ES_hourly.csv`
- `data/resampled/ES_daily.csv`
- (Same for all other symbols)

### Step 2: Run Backtest

Run the production IbsStrategy on your resampled data:

```bash
# Backtest ES from 2023-2024 (auto-loads ES_rf_model.pkl + ES_best.json)
python research/backtest_runner.py --symbol ES --start 2023-01-01 --end 2024-12-31

# Backtest multiple symbols (each loads its own ML model)
python research/backtest_runner.py --symbols ES NQ YM RTY --start 2023-01-01

# Backtest with custom initial capital
python research/backtest_runner.py --symbol ES --start 2023-01-01 --cash 250000

# Backtest without ML model (use default params)
python research/backtest_runner.py --symbol ES --start 2023-01-01 --no-ml
```

**Important:** The backtest runner **automatically loads your trained ML models** from `src/models/`:
- `ES_rf_model.pkl` - Your trained Random Forest classifier
- `ES_best.json` - 30 features + 0.55 probability threshold

This ensures the backtest uses the **exact same filters and ML model as production**!

### Step 3: Compare to Notebook Results

The backtest results should match your notebook backtests because:
- ✅ Uses the **exact** production `IbsStrategy` class
- ✅ Same features, same logic, same position sizing
- ✅ Same commission and slippage settings

If results differ, investigate why - this is critical for parity!

## Directory Structure

```
research/
├── README.md                      # This file
├── backtest_runner.py             # Main backtest script
├── utils/
│   ├── resample_data.py          # Tick → hourly/daily resampling
│   └── data_loader.py            # Load data into Backtrader
└── notebooks/                     # (Optional) Jupyter notebooks for analysis
```

## Data Requirements

### IbsStrategy needs TWO timeframes per symbol:

1. **Hourly bars** (`{SYMBOL}_hour`) - For intraday signals, IBS, stops
2. **Daily bars** (`{SYMBOL}_day`) - For trend filters, daily IBS, SMA200

### Session Handling

- **24-hour continuous futures** trading
- **Daily break:** 5-6pm ET (session close to next open)
- **Daily bars:** 6pm ET (one day) → 5pm ET (next day)

### Required Symbols

- **Execution symbols:** ES, NQ, YM, RTY, etc. (you trade these)
- **Reference symbols:** TLT (for regime filters)
- **Pair symbols:** Based on PAIR_MAP (e.g., NQ pairs with ES)

## Workflow: Research → Production

### Phase 2 (Current): Backtest Validation

1. ✅ Resample historical data to hourly + daily
2. ✅ Run production strategy on historical data
3. 🔲 Compare backtest results to notebook results
4. 🔲 Debug any discrepancies
5. 🔲 Validate on all symbols (ES, NQ, YM, RTY, etc.)

### Phase 3 (Next): ML Training Pipeline

1. Extract features from production code
2. Train models using same feature calculations
3. Save model bundles with metadata
4. Version control models

### Phase 4: Portfolio Optimization

1. Load trained models for all symbols
2. Run portfolio-level backtests
3. Optimize position sizing, correlation filters, risk limits

### Phase 5: Validation & Deployment

1. Validate new models
2. Paper trade for 1-2 weeks
3. Deploy to production
4. Monitor performance

## Common Commands

### Resample specific date range
```bash
python research/utils/resample_data.py --all --start-date 2020-01-01
```

### Backtest recent period (2023-2024)
```bash
python research/backtest_runner.py --symbol ES --start 2023-01-01
```

### Backtest full history (after resampling all data)
```bash
python research/backtest_runner.py --symbol ES --start 2010-01-01
```

### Backtest with ML models (automatic!)
```bash
# ML models are loaded automatically - no need to specify!
python research/backtest_runner.py --symbol ES --start 2023-01-01

# This automatically loads:
# - src/models/ES_rf_model.pkl (your trained model)
# - src/models/ES_best.json (features + threshold)
```

## Troubleshooting

### "Missing data feed ES_hour"
- Make sure you ran the resampling script first
- Check that `data/resampled/ES_hourly.csv` and `ES_daily.csv` exist

### "Input file not found"
- Verify your tick data is in `data/historical/`
- File should be named like `ES_bt.csv`, `NQ_bt.csv`, etc.

### Results don't match notebook
- Check if notebook uses different date range
- Verify commission/slippage settings match
- Ensure resampling logic handles sessions correctly
- Debug feature calculations (are they identical?)

### Out of memory
- Process one symbol at a time
- Use date range filters (`--start-date`, `--end-date`)
- Resample to smaller date ranges

## Next Steps

See the full ML pipeline roadmap in `docs/ml_pipeline_roadmap.md`.
