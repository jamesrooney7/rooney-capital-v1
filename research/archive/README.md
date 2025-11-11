# Research Scripts Archive

This directory contains old scripts that have been superseded by newer implementations.

## Archived: 2025-11-11

### Old Training Scripts
Replaced by `train_rf_three_way_split.py` (three-way time split methodology):
- `rf_cpcv_random_then_bo.py` - Old training script with double-dipping threshold optimization
- `train_rf_cpcv_bo.py` - Earlier training script version

**Why replaced:** The new three-way split approach eliminates threshold optimization bias by using separate temporal data periods for hyperparameter tuning, threshold optimization, and final evaluation.

### Old Portfolio Optimizers
Replaced by `portfolio_optimizer_greedy_train_test.py` (current):
- `portfolio_optimizer_full.py` - Full optimizer (older version)
- `portfolio_optimizer_train_test.py` - Train/test split optimizer (older version)
- `optimize_portfolio_positions.py` - Position optimizer (deprecated)
- `portfolio_constructor.py` - Portfolio constructor (deprecated)
- `portfolio_performance_analysis.py` - Performance analysis (deprecated)
- `portfolio_simulator.py` - Portfolio simulator (deprecated)

**Current optimizer:** `portfolio_optimizer_greedy_train_test.py`
- Greedy removal (removes worst performer iteratively)
- Max positions testing (finds optimal portfolio size)
- Train/test split (2023 train, 2024 test)
- $5k drawdown constraint on training period

### Utility Scripts
One-off diagnostic and utility scripts:
- `filter_trades_with_ml.py` - Fast ML filtering (doesn't work with current CSV format)
- `backtest_runner.py` - Backtest runner (deprecated)
- `check_trade_dates.py` - Trade date diagnostic
- `concatenate_chunks.py` - Data processing utility
- `diagnose_entries.py` - Entry diagnostic
- `export_portfolio_config.py` - Config export utility
- `extract_symbol_sharpes.py` - Sharpe extraction utility

## Current Research Scripts

**Training:**
- `extract_training_data.py` - Extract features from backtests
- `train_rf_three_way_split.py` - **Current training script** (three-way time split)

**Portfolio Optimization:**
- `portfolio_optimizer_greedy_train_test.py` - **Current optimizer** (greedy with train/test split)
- `portfolio_optimizer_simple.py` - Quick estimator using pre-computed results
- `generate_portfolio_backtest_data.py` - Generate detailed trade data for optimization

**Utilities:**
- `daily_utils.py` - Daily utility functions
- `feature_utils.py` - Feature calculation utilities
- `production_retraining.py` - Production ML retraining

## Need an Archived Script?

If you need to reference or restore any archived script, they remain available here. However, consider:
1. The current scripts implement newer, more robust methodologies
2. Archived scripts may not work with the current codebase structure
3. Check current documentation before using archived approaches
