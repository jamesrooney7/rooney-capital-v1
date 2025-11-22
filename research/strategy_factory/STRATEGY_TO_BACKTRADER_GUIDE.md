# Strategy Factory → Backtrader Porting Guide

## Overview

After Phase 1 identifies winning strategies, we need to port them to Backtrader format to enable:
1. Rich feature collection (`collect_filter_values()`)
2. ML meta-labeling
3. Integration with existing production system

---

## Architecture Comparison

### Strategy Factory (Phase 1)
```python
class RSI2MeanReversion(BaseStrategy):
    def entry_logic(self, data, params):
        return data['rsi'] < params['rsi_oversold']

    def exit_logic(self, data, params, entry_idx, entry_price, current_idx):
        if data.iloc[current_idx]['rsi'] > params['rsi_overbought']:
            return TradeExit(exit=True, exit_type='signal')
        return TradeExit(exit=False)
```

### Backtrader (Phase 3 - ML)
```python
class RSI2MeanReversionBT(IbsStrategy):  # Inherit from IbsStrategy for features
    params = (
        ('rsi_length', 2),
        ('rsi_oversold', 10),
        ('rsi_overbought', 65),
        ('enable_ml_filter', True),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(period=self.params.rsi_length)

    def next(self):
        # Entry logic
        if self.rsi < self.params.rsi_oversold:
            features = self.collect_filter_values()  # Get 50+ features
            if self.ml_filter_allows_trade(features):
                self.buy()

        # Exit logic
        elif self.position and self.rsi > self.params.rsi_overbought:
            self.close()
```

---

## Step-by-Step Porting Process

### 1. Create Backtrader Strategy File

**Location:** `src/strategy/strategy_factory/{strategy_name}_bt.py`

**Template:**
```python
#!/usr/bin/env python3
"""
{StrategyName} - Backtrader Implementation

Ported from Strategy Factory winner for ML enhancement.

Original Strategy: research/strategy_factory/strategies/{strategy_name}.py
Optimized Params: From Phase 1 winner
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class {StrategyName}BT(IbsStrategy):
    """
    {StrategyName} implemented in Backtrader with ML meta-labeling.

    Inherits from IbsStrategy to get:
    - collect_filter_values() for 50+ features
    - ML filter integration
    - Risk management (ATR stops, time stops, etc.)
    """

    params = (
        # Strategy-specific parameters (from Phase 1 winner)
        ('param1', default_value),
        ('param2', default_value),

        # ML parameters
        ('enable_ml_filter', True),
        ('ml_model_path', None),

        # Risk management (inherited from IbsStrategy)
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),
    )

    def __init__(self):
        super().__init__()

        # Initialize strategy-specific indicators
        # Example: self.rsi = bt.indicators.RSI(period=self.params.rsi_length)

    def entry_conditions_met(self):
        """
        Override from IbsStrategy to implement strategy-specific entry logic.

        Returns:
            bool: True if entry conditions are met
        """
        # Implement your strategy's entry logic here
        # Example:
        # return self.rsi[0] < self.params.rsi_oversold

        raise NotImplementedError("Implement entry logic")

    def exit_conditions_met(self):
        """
        Override from IbsStrategy to implement strategy-specific exit logic.

        Returns:
            bool: True if exit conditions are met
        """
        # Implement your strategy's exit logic here
        # Example:
        # return self.rsi[0] > self.params.rsi_overbought

        return False  # Let base class exits handle (ATR stops, time, etc.)
```

### 2. Map Strategy Factory Logic to Backtrader

Common patterns:

#### **RSI-Based Strategies**
```python
# Strategy Factory
data['rsi'] = calculate_rsi(data['Close'], period=rsi_length)
entry = data['rsi'] < rsi_oversold

# Backtrader
self.rsi = bt.indicators.RSI(period=self.params.rsi_length)
entry_met = self.rsi[0] < self.params.rsi_oversold
```

#### **Moving Average Strategies**
```python
# Strategy Factory
data['sma'] = data['Close'].rolling(window=sma_period).mean()
entry = data['Close'] > data['sma']

# Backtrader
self.sma = bt.indicators.SMA(period=self.params.sma_period)
entry_met = self.data.close[0] > self.sma[0]
```

#### **Bollinger Bands**
```python
# Strategy Factory
data['bb_upper'] = data['Close'].rolling(20).mean() + 2*data['Close'].rolling(20).std()

# Backtrader
self.bbands = bt.indicators.BollingerBands(period=20, devfactor=2)
entry_met = self.data.close[0] < self.bbands.lines.bot[0]
```

### 3. Configure for Feature Collection

The key advantage of Backtrader implementation is access to rich features:

```python
def next(self):
    if not self.position:
        # Check entry conditions
        if self.entry_conditions_met():
            # Collect 50+ features at entry time
            features = self.collect_filter_values()

            # ML filter (if enabled)
            if self.params.enable_ml_filter:
                if not self.ml_filter_allows_trade(features):
                    return  # ML says skip this trade

            # Place order
            self.buy()
    else:
        # Check exit conditions
        if self.exit_conditions_met():
            self.close()
```

**Features automatically collected:**
- Calendar: DOW, month, day of month, beginning of week
- Price: prev_day_pct, prev_bar_pct
- IBS: ibs, daily_ibs, prev_ibs, prev_daily_ibs
- Volume: volz, dvolz, vol_return, dvol_return
- ATR: atrz, datrz, atr_return, datr_return
- RSI: rsi, daily_rsi, prev_rsi
- Cross-asset: ES/NQ/YM z-scores and returns (if configured)
- 50+ total features!

---

## 4. Extract Training Data

Once ported to Backtrader:

```bash
# Run backtest and extract features + outcomes
python research/extract_training_data.py \
    --symbol ES \
    --start 2010-01-01 \
    --end 2021-12-31 \
    --strategy-class RSI2MeanReversionBT \
    --output training_data/ES_RSI2_training.csv
```

This generates a CSV with:
- Entry timestamp
- 50+ feature values at entry
- Exit timestamp
- Trade outcome (PnL, return, binary win/loss)

---

## 5. Train ML Model

```bash
# Run ML optimization (420 trials: 120 random + 300 Bayesian)
python research/train_rf_cpcv_bo.py \
    --input training_data/ES_RSI2_training.csv \
    --output models/ES_RSI2_model.pkl \
    --train-end 2021-12-31
```

This produces:
- Optimized Random Forest model
- Hyperparameters (n_estimators, max_depth, etc.)
- Optimal probability threshold
- CPCV validation results
- Deflated Sharpe Ratio

---

## 6. Validate on 2022-2024

```bash
# Run backtest with ML filter on validation period
python research/backtest_runner.py \
    --symbol ES \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --strategy RSI2MeanReversionBT \
    --ml-model models/ES_RSI2_model.pkl \
    --output results/ES_RSI2_validation.csv
```

Compare:
- Baseline (no ML): Sharpe = 1.5
- With ML: Sharpe = 3.2 (target: 2x improvement)

---

## Example: Complete RSI2 Porting

### Original (Strategy Factory)
```python
class RSI2MeanReversion(BaseStrategy):
    @property
    def param_grid(self):
        return {
            'rsi_length': [2, 3, 4],
            'rsi_oversold': [5, 10, 15],
            'rsi_overbought': [60, 65, 70, 75]
        }

    def entry_logic(self, data, params):
        return data['rsi'] < params['rsi_oversold']

    def exit_logic(self, data, params, entry_idx, entry_price, current_idx):
        if data.iloc[current_idx]['rsi'] > params['rsi_overbought']:
            return TradeExit(exit=True, exit_type='signal')
        return TradeExit(exit=False)
```

### Ported (Backtrader)
```python
class RSI2MeanReversionBT(IbsStrategy):
    params = (
        ('rsi_length', 2),  # From Phase 1 winner
        ('rsi_oversold', 10),
        ('rsi_overbought', 65),
        ('enable_ml_filter', True),
        ('ml_model_path', 'models/ES_RSI2_model.pkl'),
        ('stop_loss_atr', 1.5),  # From Phase 1 winner
        ('take_profit_atr', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(period=self.params.rsi_length)

    def entry_conditions_met(self):
        return self.rsi[0] < self.params.rsi_oversold

    def exit_conditions_met(self):
        return self.rsi[0] > self.params.rsi_overbought
```

**That's it!** Now this strategy:
- ✅ Uses optimized params from Phase 1
- ✅ Collects 50+ features via `collect_filter_values()`
- ✅ Integrates ML filter
- ✅ Ready for ML optimization

---

## Automation Workflow

For **150 winners** (15 instruments × 10 strategies):

1. **Group by strategy type** (might only be 20-30 unique strategies)
2. **Port once per strategy type** (RSI2, BuyOn5BarLow, etc.)
3. **Parameterize per winner** (different params per instrument)
4. **Automate ML pipeline** (extract → train → validate)

See `run_ml_pipeline.py` for full automation.

---

## Common Gotchas

### 1. Data Alignment
- Strategy Factory uses pandas (forward-looking ok for research)
- Backtrader enforces bar-by-bar (no lookahead)
- Use `self.indicator[0]` for current bar, `self.indicator[-1]` for previous

### 2. Exit Logic
- Strategy Factory checks exits every bar after entry
- Backtrader needs position tracking: `if self.position:`

### 3. Parameter Names
- Keep same names as Strategy Factory for traceability
- Document which Phase 1 run provided these params

### 4. Feature Availability
- IbsStrategy's `collect_filter_values()` requires warmup period
- Ensure sufficient historical data loaded

---

## Questions?

This is a one-time porting effort per strategy type. Once ported:
- All instruments with that strategy use same code
- Just different parameters
- ML pipeline fully automated

Next: See `run_ml_pipeline.py` for end-to-end automation!
