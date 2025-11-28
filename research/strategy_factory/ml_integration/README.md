# ML Integration Layer for Strategy Factory

**Purpose**: Extract training data from vectorized strategies and run the ML pipeline.

## Overview

The Strategy Factory uses **vectorized pandas-based strategies** for fast research and backtesting. This ML integration layer bridges the gap between these vectorized strategies and the existing ML training pipeline (`train_rf_cpcv_bo.py`).

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Strategy Factory                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Vectorized       │    │ Backtester       │                   │
│  │ Strategies       │───▶│ (engine/)        │                   │
│  │ (strategies/)    │    │                  │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │             ML Integration Layer                        │     │
│  │  ┌──────────────────┐    ┌──────────────────────────┐  │     │
│  │  │ FeatureExtractor │───▶│ Training Data CSV        │  │     │
│  │  │                  │    │ (y_binary, y_pnl, etc.)  │  │     │
│  │  └──────────────────┘    └────────────┬─────────────┘  │     │
│  └───────────────────────────────────────┼────────────────┘     │
│                                          │                       │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ train_rf_cpcv_bo.py    │
                              │ (CPCV + Bayesian Opt)  │
                              └────────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ Trained ML Model       │
                              │ (models/*.pkl)         │
                              └────────────────────────┘
```

## Quick Start

### 1. Extract Training Data for a Single Strategy

```bash
# Extract training data with specific parameters
python -m research.strategy_factory.ml_integration.extract_training_data \
    --strategy RSI2MeanReversion \
    --symbol ES \
    --params '{"rsi_length": 2, "rsi_oversold": 10, "rsi_overbought": 65}' \
    --start 2010-01-01 \
    --end 2024-12-31
```

### 2. Extract Training Data for All Winners

```bash
# First, generate winners.json using extract_winners.py
python -m research.strategy_factory.extract_winners \
    --db results/strategy_factory.db \
    --output winners.json \
    --top-n 3

# Then extract training data for all winners
python -m research.strategy_factory.ml_integration.extract_training_data \
    --from-winners winners.json \
    --output-dir data/training/factory
```

### 3. Run Full ML Pipeline

```bash
# Run extraction + ML training for all winners
python -m research.strategy_factory.ml_integration.run_ml_pipeline \
    --winners winners.json \
    --output-dir models/factory \
    --n-trials 50 \
    --n-folds 5

# Run for specific symbol only
python -m research.strategy_factory.ml_integration.run_ml_pipeline \
    --winners winners.json \
    --symbol ES \
    --output-dir models/factory
```

## Components

### 1. `feature_extractor.py`

Core class that extracts features from vectorized strategy backtests.

**FeatureExtractor**: Runs backtests and captures indicator values at each entry point.

```python
from research.strategy_factory.ml_integration import FeatureExtractor
from research.strategy_factory.strategies.rsi2_mean_reversion import RSI2MeanReversion

# Create strategy
strategy = RSI2MeanReversion(params={
    'rsi_length': 2,
    'rsi_oversold': 10,
    'rsi_overbought': 65
})

# Extract training data
extractor = FeatureExtractor(commission_per_side=1.00, slippage_ticks=1.0)
df = extractor.extract(strategy, data, symbol='ES')

# Output columns:
# - Date/Time, Exit Date/Time: Trade timestamps
# - Entry_Price, Exit_Price: Prices
# - y_return: Price return
# - y_binary: 1 if profitable, 0 if loss
# - y_pnl_usd: Net PnL after commissions
# - y_pnl_gross: Gross PnL before commissions
# - [indicator columns]: RSI, ATR, etc. at entry time
```

### 2. `extract_training_data.py`

CLI tool for extracting training data.

**Key Features**:
- Auto-registers strategy classes from the strategies/ directory
- Supports JSON or Python dict format for parameters
- Can process individual strategies or batch from winners.json
- Adds additional ML features (calendar, volatility, etc.)

### 3. `run_ml_pipeline.py`

Orchestrates the full ML workflow.

**Pipeline Steps**:
1. Load winners from JSON file
2. Extract training data using vectorized strategies
3. Run `train_rf_cpcv_bo.py` for each strategy
4. Save models and results summary

## Output Format

Training data CSV matches the format expected by `train_rf_cpcv_bo.py`:

| Column | Description |
|--------|-------------|
| `Date/Time` | Entry datetime |
| `Exit Date/Time` | Exit datetime |
| `Entry_Price` | Entry price (after slippage) |
| `Exit_Price` | Exit price (after slippage) |
| `y_return` | Price return (exit-entry)/entry |
| `y_binary` | 1 if net PnL > 0, else 0 |
| `y_pnl_usd` | Net PnL after commissions |
| `y_pnl_gross` | Gross PnL before commissions |
| `rsi` | RSI value at entry (strategy-specific) |
| `atr` | ATR value at entry |
| `returns_1` | 1-bar return |
| `volatility_5` | 5-bar volatility |
| ... | Additional features |

## Features Extracted

### Strategy-Specific Features
Each strategy calculates its own indicators (RSI, Bollinger Bands, MACD, etc.) which are captured at entry time.

### Additional ML Features
The extractor adds:
- **Price features**: Returns over 1, 5, 20 bars
- **Volatility features**: Rolling std of returns
- **Range features**: High-Low range, position in range
- **IBS**: Internal Bar Strength
- **Volume features**: Volume ratio, z-score (if volume data available)
- **Calendar features**: Day of week, hour, day of month, month-end flags

## Comparison: Vectorized vs Backtrader

| Aspect | Vectorized (Strategy Factory) | Backtrader (IbsStrategy) |
|--------|-------------------------------|--------------------------|
| **Speed** | Fast (pandas operations) | Slower (event-driven) |
| **Use Case** | Research, optimization | Live trading |
| **Feature Access** | DataFrame columns | collect_filter_values() |
| **ML Integration** | This layer | extract_training_data.py |

## Best Practices

### 1. Use Sufficient Training Data
- Minimum 500+ trades for reliable ML training
- 2010-2024 period provides ~14 years of diverse market conditions

### 2. Feature Selection
- Start with strategy-specific indicators
- Add cross-asset features for diversification
- Remove highly correlated features

### 3. Walk-Forward Validation
- Training data should use walk-forward splits
- Don't train on data that will be used for live trading

## Troubleshooting

### "Unknown strategy" Error
```bash
# Check available strategies
python -c "from research.strategy_factory.ml_integration.extract_training_data import register_strategies; register_strategies()"
```

### No Trades Extracted
- Check that data file exists in `data/resampled/`
- Verify parameter values allow entries (e.g., RSI oversold threshold)
- Check warmup period isn't consuming all data

### ML Training Fails
- Ensure training data has sufficient samples (500+)
- Check for NaN values in features
- Verify target column (y_binary) has both classes

## Files

```
ml_integration/
├── __init__.py              # Module exports
├── README.md                # This file
├── feature_extractor.py     # Core FeatureExtractor class
├── extract_training_data.py # CLI for training data extraction
└── run_ml_pipeline.py       # Full ML pipeline runner
```
