-- Strategy Factory Results Database Schema
-- SQLite database for storing backtest results, filters, and meta-analysis

-- Phase 1: Backtest Results
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,  -- UUID for this optimization run
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Strategy info
    strategy_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    params TEXT NOT NULL,  -- JSON string of parameters

    -- Date range
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    total_bars INTEGER NOT NULL,

    -- Trade statistics
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    win_rate REAL NOT NULL,

    -- P&L statistics
    total_pnl REAL NOT NULL,
    total_pnl_pct REAL NOT NULL,
    avg_pnl_per_trade REAL NOT NULL,
    avg_win REAL NOT NULL,
    avg_loss REAL NOT NULL,
    largest_win REAL NOT NULL,
    largest_loss REAL NOT NULL,

    -- Risk metrics
    sharpe_ratio REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    profit_factor REAL NOT NULL,

    -- Holding statistics
    avg_bars_held REAL NOT NULL,
    max_bars_held INTEGER NOT NULL,
    min_bars_held INTEGER NOT NULL,

    -- Exit breakdown (JSON string)
    exit_counts TEXT NOT NULL,

    -- Filter results (Phase 1)
    passed_gate1 BOOLEAN DEFAULT 0,  -- Trade count, Sharpe, PF, DD, WR
    passed_walkforward BOOLEAN DEFAULT 0,
    passed_regime BOOLEAN DEFAULT 0,
    passed_stability BOOLEAN DEFAULT 0,
    passed_statistical BOOLEAN DEFAULT 0,
    passed_all_filters BOOLEAN DEFAULT 0,

    -- Composite score
    composite_score REAL,

    -- Indexes for fast queries
    UNIQUE(run_id, strategy_id, symbol, params)
);

CREATE INDEX IF NOT EXISTS idx_strategy ON backtest_results(strategy_name);
CREATE INDEX IF NOT EXISTS idx_symbol ON backtest_results(symbol);
CREATE INDEX IF NOT EXISTS idx_sharpe ON backtest_results(sharpe_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_trades ON backtest_results(total_trades DESC);
CREATE INDEX IF NOT EXISTS idx_passed ON backtest_results(passed_all_filters);
CREATE INDEX IF NOT EXISTS idx_run ON backtest_results(run_id);

-- Phase 2: Multi-Symbol Validation
CREATE TABLE IF NOT EXISTS multi_symbol_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Strategy info (from Phase 1)
    strategy_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    params TEXT NOT NULL,

    -- Multi-symbol metrics
    symbols_tested TEXT NOT NULL,  -- JSON list of symbols
    symbols_passed INTEGER NOT NULL,  -- How many symbols passed
    avg_sharpe REAL NOT NULL,  -- Average Sharpe across symbols
    min_sharpe REAL NOT NULL,  -- Worst Sharpe
    max_sharpe REAL NOT NULL,  -- Best Sharpe

    -- Portfolio metrics
    avg_correlation REAL,  -- With existing strategies
    max_correlation REAL,
    portfolio_sharpe_improvement REAL,  -- Delta when added to portfolio
    incremental_alpha REAL,  -- Alpha from regression
    incremental_alpha_pvalue REAL,  -- P-value for alpha

    -- Pass/Fail
    passed_phase2 BOOLEAN DEFAULT 0,

    FOREIGN KEY(run_id) REFERENCES backtest_results(run_id)
);

CREATE INDEX IF NOT EXISTS idx_phase2_passed ON multi_symbol_results(passed_phase2);

-- Phase 3: ML Integration Results
CREATE TABLE IF NOT EXISTS ml_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Strategy info
    strategy_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    params TEXT NOT NULL,

    -- Raw performance (from Phase 1)
    raw_sharpe REAL NOT NULL,
    raw_trades INTEGER NOT NULL,
    raw_profit_factor REAL NOT NULL,

    -- ML performance (after meta-labeling)
    ml_sharpe REAL NOT NULL,
    ml_trades INTEGER NOT NULL,  -- Trades kept after filtering
    ml_profit_factor REAL NOT NULL,
    ml_precision REAL NOT NULL,  -- Precision of ML filter
    ml_recall REAL NOT NULL,  -- Recall of ML filter

    -- Improvement metrics
    sharpe_improvement_ratio REAL NOT NULL,  -- ml_sharpe / raw_sharpe
    trade_efficiency REAL NOT NULL,  -- ml_trades / raw_trades

    -- Feature importance (JSON string)
    feature_importance TEXT,

    -- Model info
    model_path TEXT,  -- Path to saved model file
    training_date TIMESTAMP,

    -- Production status
    approved_for_production BOOLEAN DEFAULT 0,

    FOREIGN KEY(run_id) REFERENCES backtest_results(run_id)
);

CREATE INDEX IF NOT EXISTS idx_ml_improvement ON ml_results(sharpe_improvement_ratio DESC);

-- Meta-Learning: Track what works across cycles
CREATE TABLE IF NOT EXISTS meta_learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Strategy characteristics
    strategy_id INTEGER NOT NULL,
    archetype TEXT NOT NULL,  -- mean_reversion, momentum, etc.

    -- Raw metrics
    raw_sharpe REAL NOT NULL,
    raw_trades INTEGER NOT NULL,

    -- ML metrics
    ml_sharpe REAL NOT NULL,
    ml_boost REAL NOT NULL,  -- Improvement ratio

    -- Feature analysis
    feature_count INTEGER NOT NULL,
    feature_diversity_score REAL NOT NULL,
    top_feature_importance REAL NOT NULL,

    -- Robustness
    regime_consistency REAL NOT NULL,  -- Score 0-1
    param_stability REAL NOT NULL,  -- Score 0-1
    cross_symbol_avg_sharpe REAL NOT NULL,

    -- Notes
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_archetype ON meta_learning(archetype);
CREATE INDEX IF NOT EXISTS idx_ml_boost ON meta_learning(ml_boost DESC);

-- Execution Runs: Track each optimization run
CREATE TABLE IF NOT EXISTS execution_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phase INTEGER NOT NULL,  -- 1, 2, or 3

    -- Configuration
    symbols TEXT NOT NULL,  -- JSON list
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    workers INTEGER NOT NULL,

    -- Results summary
    strategies_tested INTEGER NOT NULL,
    total_backtests INTEGER NOT NULL,
    strategies_passed INTEGER NOT NULL,
    runtime_seconds REAL NOT NULL,

    -- Status
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    error_message TEXT,

    -- Output paths
    report_path TEXT,
    charts_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_phase ON execution_runs(phase);
CREATE INDEX IF NOT EXISTS idx_run_status ON execution_runs(status);
