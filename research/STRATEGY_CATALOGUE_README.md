# Strategy Catalogue

## Overview

This catalogue contains 39 trading strategies organized for systematic testing through the strategy research pipeline. Each strategy is designed to feed into the ML optimization process once it demonstrates a positive edge in raw form.

## File Location

- **Main Catalogue**: `research/strategy_catalogue.csv`
- **This Documentation**: `research/STRATEGY_CATALOGUE_README.md`

## Strategy Sources

| Source | Count | IDs | Focus |
|--------|-------|-----|-------|
| **TradingView Built-in** | 20 | 1-20 | Wide variety of technical indicators and approaches |
| **Botnet101** | 8 | 21-28 | Practical intraday and short-term patterns |
| **Ernie Chan** | 7 | 29-35 | Advanced quantitative approaches (pairs, regime detection) |
| **Larry Connors** | 4 | 36-39 | Mean reversion with trend filters |

## Strategy Categories

- **Mean Reversion** (15 strategies): 1, 2, 4, 8, 9, 10, 16, 21, 23, 24, 26, 30, 31, 34, 36-39
- **Trend Following** (7 strategies): 3, 4, 17, 18, 20, 35
- **Momentum** (8 strategies): 5, 6, 7, 11, 12, 19, 28
- **Breakout** (5 strategies): 7, 15, 22, 25, 27
- **Volume-Based** (3 strategies): 13, 14, 28
- **Pairs Trading** (2 strategies): 29, 33
- **Regime Detection** (1 strategy): 32

## CSV Structure

### Column Definitions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `strategy_id` | int | Unique identifier (1-39) | 1 |
| `strategy_name` | string | Descriptive name | "Overlay Bands (Bollinger Bands)" |
| `category` | string | Strategy type | "Mean Reversion" |
| `source` | string | Origin/author | "TradingView Built-in" |
| `entry_logic_summary` | string | High-level entry description | "Enter when price closes below lower BB" |
| `primary_exit_logic` | string | Strategy-specific exit (if any) | "Close when price closes above middle band" |
| `stop_loss_atr_multiple` | float | Stop loss distance (ATR multiplier) | 1.0 |
| `take_profit_atr_multiple` | float | Profit target (ATR multiplier) | 1.0 |
| `max_bars_held` | int | Maximum holding period (bars) | 20 |
| `auto_close_time` | string | Intraday auto-close time | "4pm EST" |
| `key_parameters` | string | Primary parameters to optimize | "bb_length: 20, bb_stddev: 2.0" |
| `warmup_period` | int | Minimum bars needed before trading | 50 |
| `complexity_rating` | string | Implementation difficulty | "Low" / "Medium" / "High" / "Very High" |
| `implementation_notes` | string | Important considerations | "Simple and well-known indicator..." |

### Standardized Exits

All strategies use **consistent exit rules** to enable fair comparison:

| Exit Type | Value | Purpose |
|-----------|-------|---------|
| **Stop Loss** | 1.0 × ATR(14) | Risk management |
| **Take Profit** | 1.0 × ATR(14) | Profit taking |
| **Max Bars Held** | 20 | Time-based exit |
| **Auto Close** | 4pm EST | End-of-day exit |

**Note**: Some strategies have `primary_exit_logic` that triggers *before* standardized exits. The standardized exits act as backstops.

## Complexity Ratings

| Rating | Count | Description | Examples |
|--------|-------|-------------|----------|
| **Low** | 24 | Single indicator, straightforward logic | RSI(2), MA Cross, Bollinger Bands |
| **Medium** | 9 | Multiple indicators or pattern recognition | CRSI, Support/Resistance, PVT |
| **High** | 3 | Advanced math or multi-step logic | Regime Detection, Cointegration |
| **Very High** | 3 | Requires specialized algorithms | Kalman Filter, O-U Process, HMM |

## Data Requirements by Strategy

### Timeframe Requirements

| Strategies | Minimum Timeframe | Notes |
|------------|-------------------|-------|
| 1-20, 29-39 | Daily or Hourly | Standard technical indicators |
| 21, 23, 25 | 15-minute | Intraday patterns (gap fill, opening range) |
| 34 | 1-minute or Tick | High-frequency mean reversion |

### Special Data Needs

| Requirement | Strategies | Notes |
|-------------|-----------|-------|
| **Volume Data** | 6, 13, 14, 22, 24, 25, 28 | Critical for strategy logic |
| **Multiple Symbols** | 29, 33 | Pairs trading (ES + TLT or similar) |
| **Intraday Open/Close** | 23, 25, 26 | Gap detection, pivots, opening range |

## Usage in Strategy Optimizer

### Phase 1: Raw Strategy Testing

```python
import pandas as pd

# Load catalogue
strategies = pd.read_csv('research/strategy_catalogue.csv')

# Filter by complexity
simple_strategies = strategies[strategies['complexity_rating'] == 'Low']

# Filter by category
mean_reversion = strategies[strategies['category'] == 'Mean Reversion']

# Filter by warmup period (for limited data)
quick_warmup = strategies[strategies['warmup_period'] <= 30]
```

### Phase 2: Parameter Grid Generation

Each strategy's `key_parameters` field defines the parameters to optimize:

**Example**: Strategy #1 (Bollinger Bands)
```
key_parameters: "bb_length: 20, bb_stddev: 2.0"
```

**Optimizer generates grid**:
```python
param_grid = {
    'bb_length': [10, 15, 20, 25, 30],
    'bb_stddev': [1.5, 2.0, 2.5, 3.0]
}
# = 5 × 4 = 20 combinations per strategy
```

### Phase 3: Trade Volume Filtering

**Requirement**: 10,000+ trades (2010-2024) for ML pipeline

Strategies likely to meet this:
- Mean reversion strategies (frequent signals)
- RSI-based strategies (#1, 9, 21, 36-39)
- Daily moving average strategies (#2, 17, 18)

Strategies that may struggle:
- High-frequency strategies (#34) - need tick data
- Pairs trading (#29, 33) - less frequent setups
- Complex patterns (#27) - selective entries

## Priority Recommendations

### Start With (High Volume + Low Complexity)

1. **RSI(2) Mean Reversion** (#21) - Very frequent signals, simple logic
2. **Bollinger Bands** (#1) - Well-studied, generates many trades
3. **MA Cross** (#17) - Classic, easy to implement and optimize
4. **Double 7s** (#37) - Connors strategy, high frequency
5. **Price Channel Breakout** (#15) - Simple breakout logic

### Advanced Testing (After Initial Success)

6. **Pairs Trading** (#29) - If multi-symbol data available
7. **Kalman Filter** (#30) - Advanced mean reversion
8. **Regime Detection** (#32) - Adaptive strategy switching

## Optimization Strategy

### Parameter Ranges (Suggested)

| Parameter Type | Range | Step | Example |
|----------------|-------|------|---------|
| **Lookback Periods** | [5, 10, 15, 20, 30, 50] | Variable | MA length, RSI period |
| **Thresholds** | [10, 20, 30, 40] | 10 | RSI oversold levels |
| **Multipliers** | [1.0, 1.5, 2.0, 2.5, 3.0] | 0.5 | Bollinger std dev, ATR multiplier |
| **Percentages** | [1, 2, 3, 5] | Variable | Envelope %, gap threshold % |

### Optimization Phases

1. **Coarse Grid**: Test wide parameter ranges, 3-4 values per parameter
2. **Walk-Forward**: Validate on out-of-sample period (2022-2024)
3. **Multi-Symbol**: Test on NQ, YM, RTY if ES shows promise
4. **Fine Tuning**: Narrow parameter ranges around best performers
5. **ML Integration**: Extract features, train models, portfolio optimization

## Expected Trade Volumes (Estimates)

Based on strategy characteristics (2010-2024, ES):

| Strategy | Est. Trades | Meets 10k Threshold? |
|----------|-------------|----------------------|
| RSI(2) strategies (#21, 36-39) | 15,000-20,000 | ✅ Yes |
| Bollinger Bands (#1) | 12,000-15,000 | ✅ Yes |
| MA Cross (#17, 18) | 800-2,000 | ❌ No (unless optimized for frequency) |
| Breakout strategies (#15, 22, 25) | 1,500-3,000 | ❌ Borderline |
| MACD (#19) | 3,000-5,000 | ❌ Borderline |
| Pairs Trading (#29, 33) | 500-1,500 | ❌ No |

**Strategy**: For lower-frequency strategies, optimize parameters to increase signal frequency without sacrificing edge quality.

## Implementation Notes

### Critical Considerations

1. **ATR Calculation**: All strategies use ATR(14) for stops/targets. Ensure consistent calculation.
2. **Time Zones**: `auto_close_time` assumes EST. Adjust for futures sessions (6pm start).
3. **Warmup Periods**: Strategies need minimum bars before generating signals. Plan data accordingly.
4. **Volume Data**: 7 strategies require volume - ensure data source includes this.
5. **Intraday vs Daily**: Some strategies explicitly need intraday data (15-min or better).

### Data Pipeline Integration

```
1. Load 15-min/hourly/daily bars (just generated!)
   ↓
2. Load strategy from catalogue (CSV)
   ↓
3. Generate parameter grid
   ↓
4. Run backtests (parallel processing)
   ↓
5. Filter: Sharpe > 0.2, Trades > 10k, PF > 1.15
   ↓
6. Extract features → ML pipeline
```

## Next Steps

1. ✅ **Data Generation Complete**: 15-min bars generated via `resample_data.py`
2. 🔄 **Build Optimizer Engine**: Process strategies from this catalogue
3. 📊 **Implement Top 10**: Start with low-complexity, high-volume strategies
4. 🧪 **Backtest & Filter**: Find strategies with positive Sharpe + 10k trades
5. 🤖 **ML Integration**: Feed winners into existing ML optimization pipeline

## Questions?

For questions about specific strategies or implementation details, refer to:
- **TradingView**: [https://www.tradingview.com/scripts/](https://www.tradingview.com/scripts/)
- **Ernie Chan Books**: "Quantitative Trading", "Algorithmic Trading"
- **Larry Connors**: "Short Term Trading Strategies That Work"

---

**Last Updated**: 2025-01-20
**Total Strategies**: 39
**Status**: Ready for implementation
