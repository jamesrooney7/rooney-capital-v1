# Portfolio Constructor & Optimizer

A comprehensive system for combining multiple optimized symbol models into a unified portfolio and optimizing the maximum number of positions to improve Sharpe ratio.

## Overview

This portfolio optimization system provides three components:

1. **`portfolio_constructor.py`**: Reusable class for portfolio construction with position constraints
2. **`optimize_portfolio_positions.py`**: Lightweight optimizer using simplified signal extraction
3. **`portfolio_optimizer_full.py`**: Full integration with Backtrader and production IbsStrategy

## Key Features

### Daily Stop Loss Protection
All portfolio optimizers include a **hard daily stop loss** of $2,500 (configurable). When this threshold is hit:
- All positions are immediately exited
- No new trades are entered for the remainder of the trading day
- Trading resumes normally on the next trading day
- The number of stop-outs is tracked in the metrics

### Enhanced Performance Metrics
The system reports comprehensive metrics for each optimization run:
- **Total Return**: Both percentage and dollar amount
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Both percentage and dollar amount
- **Profit Factor**: Gross profits / gross losses
- **Daily Stops Hit**: Number of times the daily stop loss was triggered

## Quick Start (Recommended)

**Use actual pre-computed trade data** from your optimization runs:

```bash
python research/portfolio_simulator.py \
    --results-dir results \
    --min-positions 1 \
    --max-positions 10
```

This loads your **actual ML-filtered trades** from `results/` and simulates the portfolio with:
- ✅ Real trade sequences from your optimized models
- ✅ $2,500 daily stop loss applied to combined portfolio
- ✅ Position sizing and max_positions constraints
- ✅ Exact P&L tracking with daily stop enforcement

**Alternative (Quick Statistical Estimate):**

```bash
python research/portfolio_optimizer_simple.py \
    --min-positions 1 \
    --max-positions 10
```

This uses aggregate metrics from `src/models/` for instant estimates (less accurate).

## Components

### 1. **Portfolio Simulator** (✅ RECOMMENDED - Uses Actual Trade Data)

`portfolio_simulator.py` loads your actual ML-filtered trade sequences and simulates portfolio performance.

**What it does:**
- ✅ Loads daily returns from `results/{SYMBOL}_optimization/{SYMBOL}_trades.csv`
- ✅ These are already filtered by your optimized ML models
- ✅ Combines trades across symbols with max_positions limit
- ✅ Applies $2,500 daily stop loss to the combined portfolio
- ✅ Ranks symbols by Sharpe or Profit Factor
- ✅ Provides exact results based on actual trade history

**When to use:**
- ✅ You have optimization results in `results/` directory
- ✅ You want accurate simulation using real trade sequences
- ✅ You want to test daily stop loss on actual trades
- ✅ This is the MOST ACCURATE method

**Usage:**
```bash
# Basic usage (auto-discovers all symbols)
python research/portfolio_simulator.py \
    --results-dir results \
    --min-positions 1 \
    --max-positions 10

# Specify symbols and ranking method
python research/portfolio_simulator.py \
    --symbols ES NQ YM RTY GC SI \
    --ranking-method sharpe \
    --output results/portfolio_sim.csv

# Custom capital and stop
python research/portfolio_simulator.py \
    --initial-cash 500000 \
    --daily-stop-loss 5000
```

**Arguments:**
- `--symbols`: Symbols to include (default: auto-discover from results/)
- `--results-dir`: Results directory (default: results)
- `--min-positions`: Minimum positions (default: 1)
- `--max-positions`: Maximum positions (default: all available)
- `--initial-cash`: Starting capital (default: 250,000)
- `--daily-stop-loss`: Daily stop loss (default: 2,500)
- `--ranking-method`: How to rank symbols (default: sharpe, options: sharpe, profit_factor)
- `--output`: Output CSV file

---

### 2. **Simple Optimizer** (⚡ Fast - Uses Aggregate Metrics)

`portfolio_optimizer_simple.py` uses your existing optimization results without re-running backtests.

**When to use:**
- ✅ Quick analysis and recommendations
- ✅ You already have optimized models in `src/models/`
- ✅ You want fast results (seconds vs minutes/hours)

**Usage:**
```bash
# Basic usage (auto-discovers all models)
python research/portfolio_optimizer_simple.py \
    --min-positions 1 \
    --max-positions 10

# Specify symbols
python research/portfolio_optimizer_simple.py \
    --symbols ES NQ YM RTY GC SI \
    --output results/portfolio_quick.csv

# Custom capital and stop
python research/portfolio_optimizer_simple.py \
    --initial-cash 500000 \
    --daily-stop-loss 5000
```

**Arguments:**
- `--symbols`: Symbols to include (default: auto-discover from src/models/)
- `--models-dir`: Models directory (default: src/models)
- `--min-positions`: Minimum positions (default: 1)
- `--max-positions`: Maximum positions (default: all available)
- `--initial-cash`: Starting capital (default: 250,000)
- `--daily-stop-loss`: Daily stop loss (default: 2,500)
- `--output`: Output CSV file

**Note:** This provides estimates based on individual symbol metrics. For precise results with actual trade-by-trade simulation, use the Full Optimizer.

---

### 2. Portfolio Constructor (Class-Based)

`portfolio_constructor.py` provides a `PortfolioConstructor` class for flexible portfolio management.

**Key Features:**
- Load multiple symbol models from optimization results
- Generate combined portfolio signals with position constraints
- Support multiple ranking methods (probability, Sharpe, profit factor)
- Calculate portfolio-level metrics
- Optimize max_positions parameter

**Example Usage:**

```python
from research.portfolio_constructor import PortfolioConstructor

# Initialize constructor
portfolio = PortfolioConstructor(
    results_dir='src/models',
    max_positions=6,
    ranking_method='probability'
)

# Load models
portfolio.load_models(symbols=['ES', 'NQ', 'YM', 'RTY', 'GC', 'SI'])

# Get summary
print(portfolio.get_summary())

# Generate signals (requires feature_data and returns_data)
signals, probas = portfolio.generate_signals(feature_data, returns_data)

# Backtest portfolio
equity, metrics = portfolio.backtest_portfolio(signals, returns_data)

# Optimize max_positions
results = portfolio.optimize_max_positions(
    feature_data,
    returns_data,
    position_range=(1, 10)
)
```

### 3. Full Portfolio Optimizer (🔬 Precise - Runs Full Backtests)

`portfolio_optimizer_full.py` provides the most accurate optimization by running full Backtrader backtests with the production IbsStrategy.

**When to use:**
- ✅ Need precise, trade-by-trade simulation
- ✅ Validating results from Simple Optimizer
- ✅ Final verification before live trading
- ⚠️  Slower - requires full backtest runs

**Note:** This re-runs complete backtests for each symbol, which can be time-consuming.

**Usage:**

```bash
# Run portfolio optimization
python research/portfolio_optimizer_full.py \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --min-positions 1 \
    --max-positions 10

# With specific symbols
python research/portfolio_optimizer_full.py \
    --symbols ES NQ YM RTY \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --output results/portfolio_optimization.csv

# Custom data directory
python research/portfolio_optimizer_full.py \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --data-dir /path/to/data/resampled \
    --models-dir src/models
```

**Arguments:**
- `--symbols`: Symbols to include (default: auto-discover from models)
- `--start`: Start date (YYYY-MM-DD) [required]
- `--end`: End date (YYYY-MM-DD, default: today)
- `--data-dir`: Directory with resampled data (default: data/resampled)
- `--models-dir`: Directory with trained models (default: src/models)
- `--min-positions`: Minimum positions to test (default: 1)
- `--max-positions`: Maximum positions to test (default: n_symbols)
- `--commission-pct`: Commission as percentage (default: 0.0001)
- `--initial-cash`: Starting capital (default: 250,000)
- `--daily-stop-loss`: Daily stop loss in dollars (default: 2,500)
- `--output`: Output CSV file
- `--no-ml`: Disable ML filtering

## Workflow

### Step 1: Train Models for Multiple Symbols

First, ensure you have trained models for your symbols:

```bash
# Run parallel optimization for all symbols
./scripts/parallel_optimization.sh

# Or train individual symbols
python research/train_rf_three_way_split.py --symbol ES --rs-trials 25 --bo-trials 65
python research/train_rf_three_way_split.py --symbol NQ --rs-trials 25 --bo-trials 65
# ... etc
```

This creates model files in `src/models/`:
- `{SYMBOL}_rf_model.pkl`
- `{SYMBOL}_best.json`

### Step 2: Run Portfolio Simulation

**Recommended: Use Actual Trade Data**

Simulate your portfolio using the actual ML-filtered trades:

```bash
python research/portfolio_simulator.py \
    --results-dir results \
    --min-positions 1 \
    --max-positions 10 \
    --output results/portfolio_simulation.csv
```

This uses your **real trade sequences** with:
- ✅ Actual ML-filtered trades from optimization
- ✅ Daily stop loss applied to portfolio
- ✅ Position sizing and ranking
- ✅ Exact P&L based on your optimized models

**Alternative: Quick Statistical Estimate**

For instant ballpark estimates (less accurate):

```bash
python research/portfolio_optimizer_simple.py \
    --min-positions 1 \
    --max-positions 10 \
    --output results/portfolio_quick.csv
```

This uses aggregate metrics only and completes in seconds.

### Step 3: Analyze Results

The optimizer will output:

```
==========================================================================================
OPTIMIZING MAX POSITIONS: 1 to 10
Initial Capital: $250,000 | Daily Stop Loss: $2,500
==========================================================================================

Testing max_positions = 1
  Sharpe:   0.856 | CAGR:  12.30% | Return:   $45,600 | MaxDD:   -$18,200 | PF:  1.45 | Stops:   3

Testing max_positions = 2
  Sharpe:   1.024 | CAGR:  15.80% | Return:   $58,300 | MaxDD:   -$15,400 | PF:  1.68 | Stops:   2
...

==========================================================================================
OPTIMIZATION RESULTS (sorted by Sharpe)
==========================================================================================

MaxPos    Sharpe    CAGR%     Return $       MaxDD $        PF        Stops
------------------------------------------------------------------------------------------
6         1.245     18.50      $67,200        -$12,800      1.85      1
5         1.189     17.20      $61,400        -$14,100      1.73      2
7         1.156     19.30      $72,100        -$16,300      1.79      1
8         1.098     20.10      $75,800        -$19,500      1.68      3
...
==========================================================================================

🏆 OPTIMAL CONFIGURATION:
   Max Positions: 6
   Sharpe Ratio: 1.245
   CAGR: 18.50%
   Total Return: $67,200.00
   Max Drawdown: -$12,800.00 (-5.12%)
   Profit Factor: 1.85
   Daily Stops Hit: 1
```

### Step 4: Implement Optimal Configuration

Update your live trading configuration to use the optimal max_positions:

```python
# In your live trading script
MAX_POSITIONS = 6  # Based on optimization results
```

## Position Ranking Methods

When limiting positions, the portfolio constructor supports different ranking methods:

### 1. Probability Ranking (Default)
Ranks symbols by their ML prediction probability. Higher probability = higher priority.

```python
portfolio = PortfolioConstructor(
    results_dir='src/models',
    max_positions=6,
    ranking_method='probability'  # Default
)
```

### 2. Sharpe Ranking
Ranks symbols by their historical Sharpe ratio from optimization.

```python
portfolio = PortfolioConstructor(
    results_dir='src/models',
    max_positions=6,
    ranking_method='sharpe'
)
```

### 3. Profit Factor Ranking
Ranks symbols by their historical profit factor.

```python
portfolio = PortfolioConstructor(
    results_dir='src/models',
    max_positions=6,
    ranking_method='profit_factor'
)
```

## Output Format

The optimizer saves results to CSV with these columns:

| Column | Description |
|--------|-------------|
| `max_positions` | Number of max positions tested |
| `total_return` | Total return over period |
| `annualized_return` | Annualized return |
| `annualized_volatility` | Annualized volatility |
| `sharpe_ratio` | Sharpe ratio (risk-free rate = 0) |
| `max_drawdown` | Maximum drawdown |
| `avg_positions` | Average number of positions held |
| `n_periods` | Number of trading periods |

## Performance Metrics

### Sharpe Ratio
Primary optimization metric. Measures risk-adjusted return.

**Formula:** `Sharpe = Annualized_Return / Annualized_Volatility`

**Interpretation:**
- `< 0.5`: Poor
- `0.5 - 1.0`: Good
- `1.0 - 2.0`: Very good
- `> 2.0`: Excellent

### Max Drawdown
Maximum peak-to-trough decline in portfolio value.

Lower (less negative) is better. Indicates risk management quality.

### Win Rate
Percentage of profitable trades.

Note: High win rate doesn't guarantee high Sharpe if avg loss > avg win.

## Advanced Usage

### Custom Feature Data

If you have pre-computed features:

```python
from research.portfolio_constructor import PortfolioConstructor

portfolio = PortfolioConstructor(results_dir='src/models')
portfolio.load_models()

# Your feature data (dict of DataFrames)
feature_data = {
    'ES': es_features_df,
    'NQ': nq_features_df,
    # ...
}

# Your returns data (dict of Series)
returns_data = {
    'ES': es_returns_series,
    'NQ': nq_returns_series,
    # ...
}

# Optimize
results = portfolio.optimize_max_positions(
    feature_data,
    returns_data,
    position_range=(1, 12)
)
```

### Backtesting Individual Configurations

```python
# Test specific max_positions value
portfolio.max_positions = 6
signals, probas = portfolio.generate_signals(feature_data, returns_data)
equity, metrics = portfolio.backtest_portfolio(signals, returns_data)

print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
print(f"Return: {metrics['annualized_return']*100:.2f}%")
```

## Data Requirements

### For Full Optimizer

Requires resampled OHLCV data in `data/resampled/`:
- `{SYMBOL}_hourly.csv`
- `{SYMBOL}_daily.csv`

Format:
```
datetime,Open,High,Low,Close,volume
2023-01-03 09:30:00,4100.0,4105.5,4098.0,4103.0,50000
...
```

### For Simplified Optimizer

Same as above, but uses simplified IBS calculation instead of full feature engineering.

## Troubleshooting

### "No models found"

Ensure models exist in `src/models/`:
```bash
ls src/models/*_best.json
```

If missing, run optimization first:
```bash
./scripts/parallel_optimization.sh
```

### "No data found for symbol"

Check data directory:
```bash
ls data/resampled/{SYMBOL}_*.csv
```

Verify data format and date ranges.

### "No signals extracted"

- Check that date range has sufficient data
- Verify ML models load correctly
- Check logs for feature calculation errors

### Memory Issues

For large portfolios or long date ranges:
- Reduce number of symbols
- Shorten date range
- Use simplified optimizer instead of full

## Best Practices

1. **Use Out-of-Sample Data**: Optimize on data NOT used for model training
2. **Test Multiple Periods**: Verify robustness across different market regimes
3. **Monitor Regime Changes**: Re-optimize periodically (e.g., quarterly)
4. **Consider Transaction Costs**: Higher max_positions = more rebalancing costs
5. **Diversify**: Don't just choose highest Sharpe - consider drawdown and volatility

## Future Enhancements

Potential improvements:

- [ ] Dynamic position sizing based on volatility
- [ ] Sector/asset class constraints
- [ ] Correlation-based diversification
- [ ] Kelly criterion position sizing
- [ ] Walk-forward optimization
- [ ] Multi-objective optimization (Sharpe + drawdown)
- [ ] Machine learning-based position allocation
- [ ] Real-time portfolio rebalancing signals

## References

- IbsStrategy: `src/strategy/ibs_strategy.py`
- Model Training: `research/train_rf_three_way_split.py`
- Backtest Runner: `research/backtest_runner.py`
- Model Loader: `src/models/loader.py`
