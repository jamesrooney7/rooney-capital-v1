# Strategy Parameter Grids Summary

## Overview

This document summarizes the parameter grids defined in `strategy_parameter_grids.yaml`. Each of the 39 strategies has a specific parameter grid that will be tested during the optimization process.

## Grid Design Philosophy

### Three Types of Parameters

1. **Indicator Parameters** (Strategy-Specific)
   - Core indicator settings (RSI period, MA length, etc.)
   - 3-4 values per parameter
   - Based on literature defaults and common variations

2. **Stop Loss Parameters** (Universal)
   - ATR multiples: [0.5, 1.0, 1.5, 2.0]
   - Tests both tight and loose risk management
   - Based on ATR(14)

3. **Take Profit Parameters** (Universal)
   - ATR multiples: [1.0, 1.5, 2.0, 3.0]
   - Tests different risk/reward ratios
   - Based on ATR(14)

### Why Test Stops and Targets?

- Different strategies have different optimal risk/reward profiles
- A mean reversion strategy might need tight stops (0.5 ATR)
- A trend-following strategy might need wider stops (2.0 ATR)
- Optimal stop/target ratio is crucial for profitability
- Testing combinations reveals which exits work best with which signals

## Combination Counts by Strategy

### Low Complexity (<200 combinations)

| Strategy ID | Name | Combinations | Priority |
|-------------|------|--------------|----------|
| 5 | Momentum Strategy | 64 | Medium |
| 24 | VWAP Reversion | 64 | High |
| 14 | Price Volume Trend | 81 | Medium |
| 26 | Pivot Point Reversal | 81 | Medium |
| 35 | Linear Regression Channel | 81 | Medium |
| 16 | Envelope Indicator | 108 | Medium |
| 23 | Gap Fill Strategy | 108 | High |
| 2 | Overlay Bands MA | 144 | High |
| 4 | Supertrend | 144 | Medium |
| 7 | Volatility Strategy | 144 | Medium |
| 13 | Volume Indicator | 144 | Medium |
| 25 | Opening Range Breakout | 144 | High |
| 27 | 3-Bar Pattern | 144 | Medium |
| 32 | Regime Change | 162 | Low |
| 15 | Price Channel | 192 | High |
| 22 | Support/Resistance Breakout | 192 | Medium |

### Medium Complexity (200-500 combinations)

| Strategy ID | Name | Combinations | Priority |
|-------------|------|--------------|----------|
| 3 | Parabolic SAR | 243 | Medium |
| 8 | CMO Strategy | 243 | Medium |
| 10 | Psychological Line | 243 | Medium |
| 18 | Triple MA | 243 | Medium |
| 20 | Parabolic SAR (dup) | 243 | Low |
| 6 | Volume Weighted MACD | 243 | Medium |
| 28 | Volume Price Confirmation | 243 | Medium |
| 29 | Pairs Trading | 243 | Medium |
| 30 | Kalman Filter | 243 | Low |
| 31 | O-U Mean Reversion | 243 | Low |
| 33 | Cointegration | 243 | Low |
| 34 | High Frequency MR | 243 | Low |
| 1 | Bollinger Bands | 256 | High |
| 17 | MA Cross | 256 | High |
| 19 | MACD | 432 | High |
| 37 | Double 7s | 432 | High |

### High Complexity (>500 combinations)

| Strategy ID | Name | Combinations | Priority |
|-------------|------|--------------|----------|
| 21 | RSI(2) Mean Reversion | 576 | **Very High** |
| 39 | Cumulative RSI | 864 | High |
| 9 | Connors RSI | 972 | Medium |
| 36 | RSI(2) + 200 SMA | 1,296 | **Very High** |
| 11 | WaveTrend | 729 | Medium |
| 12 | Know Sure Thing | 1,701 | Low |
| 38 | 2-Period RSI + MA | 1,728 | High |

## Estimated Total Combinations

**Across all 39 strategies**: ~14,000 parameter combinations

With concurrent execution (16-32 workers), this represents:
- **Low estimate** (30 sec/backtest): 117 hours single-threaded → **4-7 hours parallelized**
- **High estimate** (60 sec/backtest): 233 hours single-threaded → **7-15 hours parallelized**

## Prioritization for Phase 1

### Tier 1: Start Here (High Volume + Manageable Combinations)
1. **RSI(2) Mean Reversion** (#21) - 576 combinations, 15k+ trades expected
2. **Bollinger Bands** (#1) - 256 combinations, 12k+ trades expected
3. **RSI(2) + 200 SMA** (#36) - 1,296 combinations, 15k+ trades expected
4. **Double 7s** (#37) - 432 combinations, 10k+ trades expected
5. **MA Cross** (#17) - 256 combinations, varies by parameters

**Rationale**: These strategies are well-documented, likely to hit 10k trade threshold, and have reasonable combination counts.

### Tier 2: Quick Wins (Low Combinations)
6. **VWAP Reversion** (#24) - 64 combinations
7. **Gap Fill** (#23) - 108 combinations (needs intraday data)
8. **Opening Range Breakout** (#25) - 144 combinations (needs 15-min data)
9. **Price Channel** (#15) - 192 combinations

**Rationale**: Fast to test, can provide quick feedback on infrastructure.

### Tier 3: Advanced Testing (After Initial Success)
10. **Pairs Trading** (#29) - 243 combinations (multi-symbol)
11. **Kalman Filter** (#30) - 243 combinations (complex math)
12. **Regime Detection** (#32) - 162 combinations (adaptive)

**Rationale**: More complex implementation, test after infrastructure proven.

## Parameter Naming Conventions

All parameter names follow these patterns:

### Indicator Parameters
- `{indicator}_{property}`: e.g., `rsi_length`, `ma_fast`, `bb_stddev`
- Lowercase with underscores
- Descriptive of what they control

### Exit Parameters
- `stop_loss_atr`: Stop loss distance in ATR multiples
- `take_profit_atr`: Take profit distance in ATR multiples
- Always in ATR(14) multiples for consistency

## Usage in Optimizer

```python
import yaml

# Load parameter grids
with open('research/strategy_parameter_grids.yaml') as f:
    param_grids = yaml.safe_load(f)

# Get grid for strategy #21 (RSI(2))
rsi2_grid = param_grids[21]

# Generate all combinations
from itertools import product

keys = rsi2_grid.keys()
values = rsi2_grid.values()
combinations = [dict(zip(keys, v)) for v in product(*values)]

print(f"Total combinations: {len(combinations)}")
# Output: Total combinations: 576

# Example combination:
# {'rsi_length': 2, 'rsi_oversold': 10, 'rsi_overbought': 65,
#  'stop_loss_atr': 1.0, 'take_profit_atr': 1.5}
```

## Optimization Strategy

### Phase 1: Coarse Grid (Current)
- Test all defined parameter combinations
- Filter for Sharpe > 0.2, Trades > 10k, PF > 1.15
- Identify which strategies have promise

### Phase 2: Fine Grid (After Phase 1)
- For winning strategies, narrow parameter ranges
- Add intermediate values around best performers
- Test stop/target ratios more granularly

### Phase 3: Bayesian Optimization (After Phase 2)
- Use results from Phases 1-2 to inform priors
- Apply Bayesian optimization (like your ML pipeline uses)
- Find optimal parameters with fewer evaluations

## Risk/Reward Analysis

Different stop/target combinations imply different risk/reward ratios:

| Stop (ATR) | Target (ATR) | R:R Ratio | Win Rate Needed* |
|------------|--------------|-----------|------------------|
| 0.5 | 1.0 | 1:2 | 33% |
| 1.0 | 1.0 | 1:1 | 50% |
| 1.0 | 1.5 | 1:1.5 | 40% |
| 1.0 | 2.0 | 1:2 | 33% |
| 1.5 | 3.0 | 1:2 | 33% |
| 2.0 | 3.0 | 1:1.5 | 40% |

*Breakeven win rate for 1.0 profit factor

### Expected Patterns

- **Mean reversion strategies**: Likely prefer tighter stops, symmetric R:R (1:1)
- **Trend following strategies**: Likely prefer wider stops, asymmetric R:R (1:2)
- **Breakout strategies**: Likely prefer wider stops, higher R:R (1:2 or 1:3)

The optimizer will reveal which exit parameters work best with which signal parameters.

## Constraints and Considerations

### Maximum Bars Held
- All strategies: 20 bars max (can be parameterized later)
- Prevents holding losers too long
- Ensures portfolio turnover

### Auto-Close Time
- All strategies: 4pm EST
- Prevents overnight gaps (futures have 1-hour gap 5-6pm)
- Can be disabled for multi-day strategies

### ATR Period
- All exit parameters use ATR(14)
- Standard industry setting
- Could be parameterized in Phase 2 if needed

## Reducing Combination Counts

If computing resources are limited, reduce combinations by:

1. **Remove one stop/target value**
   - Use [0.5, 1.0, 2.0] instead of [0.5, 1.0, 1.5, 2.0]
   - Reduces combinations by 25%

2. **Fix stop or target**
   - Start with stop_loss_atr = 1.0 (fixed)
   - Only vary take_profit_atr
   - Reduces combinations by 75%

3. **Use factorial design**
   - Instead of full grid, use orthogonal arrays
   - Reduces combinations by 50-80% with minimal information loss

4. **Prioritize indicator parameters**
   - Test indicator parameters first
   - Then optimize stops/targets for winners only

## Next Steps

1. ✅ **Parameter grids defined** - This file
2. 🔄 **Implement strategy classes** - One per strategy from catalogue
3. 🔄 **Build optimizer engine** - Generate combinations, run backtests
4. 📊 **Execute Phase 1** - Test all combinations on ES (2010-2024)
5. 🎯 **Filter results** - Find strategies with positive edge + 10k trades
6. 🤖 **ML integration** - Feed winners into existing pipeline

---

**Last Updated**: 2025-01-20
**Total Strategies**: 39
**Total Combinations**: ~14,000
**Estimated Runtime**: 4-15 hours (parallelized)
