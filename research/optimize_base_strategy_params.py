#!/usr/bin/env python3
"""
Base Strategy Parameter Optimization with Walk-Forward Analysis.

This script optimizes the core IBS strategy parameters (entry/exit thresholds,
ATR-based stops/targets, holding period) to maximize trade volume while
maintaining minimum quality thresholds.

Primary Goal: Generate MAXIMUM trade volume for downstream ML training
Success Metric: 3,000-5,000 OOS trades with win rate > 48% and Sharpe > 0.25

Usage:
    python research/optimize_base_strategy_params.py --symbol ES

Output:
    optimization_results/ES/
    ├── windows/           # Per-window results
    ├── analysis/          # Aggregate analysis
    ├── heldout/          # Held-out validation
    └── reports/          # Final reports
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research.utils.vectorized_backtest import load_data, run_backtest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# WALK-FORWARD WINDOWS
# =============================================================================

WINDOWS = [
    {
        'id': 1,
        'train_start': '2011-01-01',
        'train_end': '2015-12-31',
        'test_start': '2016-01-01',
        'test_end': '2016-12-31',
    },
    {
        'id': 2,
        'train_start': '2011-01-01',
        'train_end': '2016-12-31',
        'test_start': '2017-01-01',
        'test_end': '2017-12-31',
    },
    {
        'id': 3,
        'train_start': '2011-01-01',
        'train_end': '2017-12-31',
        'test_start': '2018-01-01',
        'test_end': '2018-12-31',
    },
    {
        'id': 4,
        'train_start': '2011-01-01',
        'train_end': '2018-12-31',
        'test_start': '2019-01-01',
        'test_end': '2019-12-31',
    },
    {
        'id': 5,
        'train_start': '2011-01-01',
        'train_end': '2019-12-31',
        'test_start': '2020-01-01',
        'test_end': '2020-12-31',
    },
]

HELDOUT_PERIOD = {
    'start': '2021-01-01',
    'end': '2024-12-31',
}


# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

FIXED_PARAMS = {
    'atr_period': 14,              # Industry standard
    'ibs_entry_low': 0.0,          # Capture all extreme oversold
    'ibs_exit_high': 1.0,          # Capture all overbought exits
    'auto_close_hour': 15,         # 3 PM ET close (unchanged from spec)
}

OPTIMIZATION_PARAMS = {
    # Entry threshold - how oversold before entry
    'ibs_entry_high': {
        'type': 'categorical',
        'values': [0.15, 0.20, 0.25, 0.30, 0.35],
        # Higher = more permissive = more trades
    },

    # Exit threshold - how overbought before exit
    'ibs_exit_low': {
        'type': 'categorical',
        'values': [0.65, 0.70, 0.75, 0.80],
        # Lower = exit later = more time for target
    },

    # Stop loss multiplier
    'stop_atr_mult': {
        'type': 'categorical',
        'values': [2.5, 3.0, 3.5],
        # Higher = wider stops = fewer stop-outs
    },

    # Take profit multiplier
    'target_atr_mult': {
        'type': 'categorical',
        'values': [1.0, 1.5, 2.0, 2.5],
        # Higher = let winners run more
    },

    # Maximum holding period (hours)
    'max_holding_bars': {
        'type': 'categorical',
        'values': [6, 8, 10, 12, 15],
        # Longer = fewer time-based exits
    },
}

# Total search space: 5 × 4 × 3 × 4 × 5 = 1,200 combinations
# Bayesian optimization will test ~100 of these per window


# =============================================================================
# THRESHOLDS
# =============================================================================

TRAINING_THRESHOLDS = {
    'absolute_minimum': 1500,    # Below this: STOP optimization
    'warning': 2000,              # Below this: Flag warning
    'target': 2500,               # Desired minimum
    'good': 3000,                 # Good volume
}

TESTING_THRESHOLDS = {
    'unreliable': 300,           # Below this: Flag as unreliable
    'marginal': 500,             # Below this: Marginal reliability
    'target': 700,               # Desired minimum per year
    'good': 900,                 # Good volume
}

TOTAL_OOS_THRESHOLDS = {
    'reject': 2000,              # Below this: Reject for ML
    'marginal': 3000,            # Below this: Simplified ML only
    'acceptable': 5000,          # Below this: Standard ML
    'good': 7000,                # Above this: Full ML with ensemble
}


# =============================================================================
# OPTUNA CONFIGURATION
# =============================================================================

OPTUNA_CONFIG = {
    'n_trials': 100,
    'n_startup_trials': 20,      # First 20 trials random
    'sampler': 'TPE',             # Tree-structured Parzen Estimator
    'pruner': 'MedianPruner',
    'pruner_config': {
        'n_startup_trials': 20,
        'n_warmup_steps': 25,     # Wait 25% of trials before pruning
        'interval_steps': 10,
    },
}


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def calculate_robust_performance(
    train_data: pd.DataFrame,
    params: Dict,
    symbol: str
) -> Dict:
    """
    Calculate robust performance metrics across yearly sub-periods.

    Instead of optimizing aggregate performance, this splits the training data
    into yearly chunks and calculates performance for each year. Returns the
    worst-case (minimum) profit factor, mean, and standard deviation.

    This ensures the strategy works consistently across different market regimes,
    not just in aggregate or during favorable periods.

    Args:
        train_data: Full training DataFrame
        params: Strategy parameters to test
        symbol: Symbol name for point_value lookup

    Returns:
        Dict with:
            - min_pf: Minimum profit factor across all years
            - mean_pf: Average profit factor across all years
            - std_pf: Standard deviation of profit factor across years
            - total_trades: Total number of trades across all years
            - year_results: List of per-year results
    """
    # Split data by year (using datetime index)
    train_data = train_data.copy()
    train_data['year'] = train_data.index.year
    years = sorted(train_data['year'].unique())

    year_results = []
    profit_factors = []
    total_trades = 0

    for year in years:
        year_data = train_data[train_data['year'] == year].copy()

        # Skip years with insufficient data
        if len(year_data) < 100:
            continue

        # Run backtest for this year
        results = run_backtest(year_data, params, symbol=symbol)

        # Skip if no trades
        if results['num_trades'] == 0:
            continue

        pf = results['profit_factor']

        # Handle None (no losses - treat as very profitable)
        if pf is None:
            pf = 2.0  # Conservative estimate for no-loss scenario

        year_results.append({
            'year': year,
            'profit_factor': pf,
            'num_trades': results['num_trades'],
            'win_rate': results['win_rate'],
            'sharpe_ratio': results['sharpe_ratio']
        })

        profit_factors.append(pf)
        total_trades += results['num_trades']

    # Calculate robust metrics
    if len(profit_factors) == 0:
        return {
            'min_pf': 0.0,
            'mean_pf': 0.0,
            'std_pf': 0.0,
            'total_trades': 0,
            'year_results': []
        }

    return {
        'min_pf': np.min(profit_factors),
        'mean_pf': np.mean(profit_factors),
        'std_pf': np.std(profit_factors),
        'total_trades': total_trades,
        'year_results': year_results
    }


def objective_function(trial: optuna.Trial, train_data: pd.DataFrame, symbol: str) -> float:
    """
    Objective function that prioritizes volume with light consistency preference.

    Philosophy:
    - Base strategy does NOT need to be profitable (ML filter adds profitability)
    - Primary goal: Maximum trade volume for ML training
    - Secondary goal: Light consistency across years (avoid boom-bust)

    This function:
    1. Splits training data into yearly sub-periods
    2. Calculates profit factor variance across years
    3. Optimizes for: (80% volume) + (20% consistency bonus)

    Returns score to maximize: 80% volume + 20% consistency

    Args:
        trial: Optuna trial object
        train_data: Training DataFrame
        symbol: Symbol name for point_value lookup

    Returns:
        Score to maximize (higher is better)
    """
    # 1. SUGGEST PARAMETERS
    params = {**FIXED_PARAMS}  # Start with fixed params

    params['ibs_entry_high'] = trial.suggest_categorical(
        'ibs_entry_high', OPTIMIZATION_PARAMS['ibs_entry_high']['values']
    )
    params['ibs_exit_low'] = trial.suggest_categorical(
        'ibs_exit_low', OPTIMIZATION_PARAMS['ibs_exit_low']['values']
    )
    params['stop_atr_mult'] = trial.suggest_categorical(
        'stop_atr_mult', OPTIMIZATION_PARAMS['stop_atr_mult']['values']
    )
    params['target_atr_mult'] = trial.suggest_categorical(
        'target_atr_mult', OPTIMIZATION_PARAMS['target_atr_mult']['values']
    )
    params['max_holding_bars'] = trial.suggest_categorical(
        'max_holding_bars', OPTIMIZATION_PARAMS['max_holding_bars']['values']
    )

    # 2. RUN ROBUST BACKTEST (calculate performance for each year)
    robust_results = calculate_robust_performance(train_data, params, symbol=symbol)

    # 3. APPLY HARD CONSTRAINTS (MUST PASS)

    # Constraint 1: Minimum trade volume (total across all years)
    if robust_results['total_trades'] < 1000:
        return -999999.0  # Severe penalty - reject trial

    # NOTE: We do NOT require profitability at this stage!
    # Base strategy was slightly unprofitable (mean -$0.13/trade)
    # ML filter is what makes it profitable
    # We just want: (1) volume, (2) light consistency

    # 4. CALCULATE OBJECTIVE SCORE
    # Weighted combination: 80% volume, 20% consistency bonus

    # Volume component (normalize to ~1.0 at 2500 trades)
    volume_score = robust_results['total_trades'] / 2500.0

    # Consistency bonus: Reward low variance across years (very light)
    # We don't want boom-bust strategies, but we're not strict about it
    # std_pf = 0.0 → bonus of +0.20 (perfectly consistent)
    # std_pf = 0.5 → bonus of +0.10 (moderate variance)
    # std_pf = 1.0 → bonus of 0.00 (high variance, no bonus)
    consistency_bonus = max(0.0, 0.20 - robust_results['std_pf'] * 0.20)

    # Combined score: 80% volume, 20% consistency
    # This prioritizes volume generation while gently favoring consistency
    score = 0.80 * volume_score + 0.20 * consistency_bonus

    # 5. BONUS FOR HIGH VOLUME
    if robust_results['total_trades'] >= 3000:
        score *= 1.3  # 30% bonus
    elif robust_results['total_trades'] >= 2500:
        score *= 1.15  # 15% bonus
    elif robust_results['total_trades'] >= 2000:
        score *= 1.05  # 5% bonus

    return score


# =============================================================================
# VIABILITY CHECKS
# =============================================================================

def check_window_1_training_viability(train_results: Dict) -> Dict:
    """
    Check if Window 1 training results are viable.
    If not, STOP and flag for parameter adjustment.

    Returns: dict with 'proceed' (bool), 'status', and 'message' (str)
    """
    num_trades = train_results['num_trades']

    # Rule 1: Absolute minimum
    if num_trades < TRAINING_THRESHOLDS['absolute_minimum']:
        return {
            'proceed': False,
            'status': 'STOP',
            'message': f'CRITICAL: Only {num_trades} training trades. '
                      f'Parameters too restrictive. Adjust to more permissive ranges and re-run.'
        }

    # Rule 2: Warning threshold
    elif num_trades < TRAINING_THRESHOLDS['warning']:
        return {
            'proceed': True,
            'status': 'WARNING',
            'message': f'WARNING: Only {num_trades} training trades. '
                      f'At edge of viability. Proceeding with caution.'
        }

    # Rule 3: Acceptable
    else:
        return {
            'proceed': True,
            'status': 'OK',
            'message': f'OK: {num_trades} training trades. Sufficient volume.'
        }


def check_window_test_viability(test_results: Dict) -> Dict:
    """
    Check if window test results are viable.
    Flag unreliable windows.

    Returns: dict with 'reliable' (bool), 'status', and 'message' (str)
    """
    num_trades = test_results['num_trades']

    # Rule 1: Unreliable
    if num_trades < TESTING_THRESHOLDS['unreliable']:
        return {
            'reliable': False,
            'status': 'UNRELIABLE',
            'message': f'WARNING: Only {num_trades} test trades. '
                      f'Metrics unreliable. Flag window for exclusion.'
        }

    # Rule 2: Marginal
    elif num_trades < TESTING_THRESHOLDS['marginal']:
        return {
            'reliable': True,
            'status': 'MARGINAL',
            'message': f'WARNING: Only {num_trades} test trades. '
                      f'Marginal reliability. Include but note uncertainty.'
        }

    # Rule 3: Good
    else:
        return {
            'reliable': True,
            'status': 'OK',
            'message': f'OK: {num_trades} test trades. Reliable metrics.'
        }


def check_total_oos_viability(total_oos_trades: int) -> Dict:
    """
    Check if total out-of-sample volume is sufficient for ML.
    This determines if we proceed to ML layer.

    Returns: dict with 'ml_viable' (bool), 'ml_config' (dict), 'status', 'message' (str)
    """
    # Rule 1: Insufficient - REJECT
    if total_oos_trades < TOTAL_OOS_THRESHOLDS['reject']:
        return {
            'ml_viable': False,
            'ml_config': None,
            'status': 'REJECT',
            'message': f'REJECT: Only {total_oos_trades} total OOS trades. '
                      f'Insufficient for ML training. Do not proceed to ML layer.'
        }

    # Rule 2: Marginal - Simplified ML
    elif total_oos_trades < TOTAL_OOS_THRESHOLDS['marginal']:
        return {
            'ml_viable': True,
            'ml_config': {
                'max_features': 20,      # Reduce from 30
                'use_ensemble': False,   # Skip ensemble
                'conservative': True,    # More regularization
            },
            'status': 'MARGINAL',
            'message': f'MARGINAL: {total_oos_trades} total OOS trades. '
                      f'Proceed with simplified ML: 20 features, no ensemble.'
        }

    # Rule 3: Acceptable - Standard ML
    elif total_oos_trades < TOTAL_OOS_THRESHOLDS['acceptable']:
        return {
            'ml_viable': True,
            'ml_config': {
                'max_features': 25,      # Slightly reduced
                'use_ensemble': False,   # Skip ensemble
                'conservative': False,   # Normal regularization
            },
            'status': 'ACCEPTABLE',
            'message': f'ACCEPTABLE: {total_oos_trades} total OOS trades. '
                      f'Proceed with standard ML: 25 features, single model.'
        }

    # Rule 4: Good - Full ML
    else:
        return {
            'ml_viable': True,
            'ml_config': {
                'max_features': 30,      # Full feature set
                'use_ensemble': True,    # Can use ensemble
                'conservative': False,   # Normal regularization
            },
            'status': 'GOOD',
            'message': f'GOOD: {total_oos_trades} total OOS trades. '
                      f'Proceed with full ML: 30 features, ensemble optional.'
        }


# =============================================================================
# WALK-FORWARD EFFICIENCY (WFE)
# =============================================================================

def calculate_wfe(train_sharpe: float, test_sharpe: float) -> Optional[float]:
    """
    Calculate Walk-Forward Efficiency.

    WFE = Test Sharpe / Train Sharpe

    Interpretation:
    - WFE > 0.7: Excellent generalization
    - WFE 0.5-0.7: Good generalization
    - WFE 0.4-0.5: Acceptable
    - WFE < 0.4: Poor (likely overfitting)
    """
    if train_sharpe <= 0:
        return None  # Can't calculate meaningful WFE

    wfe = test_sharpe / train_sharpe
    return wfe


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def optimize_window(
    window: Dict,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: Path,
    symbol: str
) -> Dict:
    """
    Optimize parameters for a single walk-forward window.

    Args:
        window: Window configuration dict
        train_data: Training DataFrame
        test_data: Testing DataFrame
        output_dir: Output directory for this window
        symbol: Symbol name for point_value lookup

    Returns:
        Dict with window results
    """
    window_id = window['id']
    logger.info(f"\n{'='*80}")
    logger.info(f"WINDOW {window_id}: Training {window['train_start']} to {window['train_end']}")
    logger.info(f"           Testing {window['test_start']} to {window['test_end']}")
    logger.info(f"{'='*80}\n")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            n_startup_trials=OPTUNA_CONFIG['n_startup_trials']
        ),
        pruner=MedianPruner(
            n_startup_trials=OPTUNA_CONFIG['pruner_config']['n_startup_trials'],
            n_warmup_steps=OPTUNA_CONFIG['pruner_config']['n_warmup_steps'],
            interval_steps=OPTUNA_CONFIG['pruner_config']['interval_steps'],
        )
    )

    # Run optimization
    logger.info(f"Starting Bayesian optimization: {OPTUNA_CONFIG['n_trials']} trials")

    study.optimize(
        lambda trial: objective_function(trial, train_data, symbol),
        n_trials=OPTUNA_CONFIG['n_trials'],
        show_progress_bar=True
    )

    # Get best parameters
    best_params = {**FIXED_PARAMS, **study.best_params}
    logger.info(f"\nBest parameters found:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Train final model on all training data with best params
    logger.info(f"\nRunning final backtest on training data...")
    train_metrics = run_backtest(train_data, best_params, symbol=symbol)

    logger.info(f"Training Results:")
    logger.info(f"  Trades: {train_metrics['num_trades']:,}")
    logger.info(f"  Win Rate: {train_metrics['win_rate']:.2%}")
    logger.info(f"  Sharpe: {train_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Avg P&L: {train_metrics['avg_pnl_per_trade']:.2f}")

    # CHECKPOINT: Check training viability (Window 1 only)
    if window_id == 1:
        viability = check_window_1_training_viability(train_metrics)
        logger.info(f"\nWindow 1 Training Viability Check:")
        logger.info(f"  Status: {viability['status']}")
        logger.info(f"  Message: {viability['message']}")

        if not viability['proceed']:
            logger.error(f"\n{'='*80}")
            logger.error(f"STOPPING: {viability['message']}")
            logger.error(f"{'='*80}\n")
            sys.exit(1)

    # Test on out-of-sample data
    logger.info(f"\nRunning backtest on test data...")
    test_metrics = run_backtest(test_data, best_params, symbol=symbol)

    logger.info(f"Test Results:")
    logger.info(f"  Trades: {test_metrics['num_trades']:,}")
    logger.info(f"  Win Rate: {test_metrics['win_rate']:.2%}")
    logger.info(f"  Sharpe: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Avg P&L: {test_metrics['avg_pnl_per_trade']:.2f}")

    # Check test viability
    test_viability = check_window_test_viability(test_metrics)
    logger.info(f"\nTest Viability Check:")
    logger.info(f"  Status: {test_viability['status']}")
    logger.info(f"  Message: {test_viability['message']}")

    # Calculate WFE
    wfe = calculate_wfe(train_metrics['sharpe_ratio'], test_metrics['sharpe_ratio'])
    if wfe is not None:
        logger.info(f"\nWalk-Forward Efficiency: {wfe:.3f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimal parameters
    params_file = output_dir / f"window_{window_id}_optimal_params.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"\nSaved optimal parameters to: {params_file}")

    # Save train metrics
    train_metrics_file = output_dir / f"window_{window_id}_train_metrics.json"
    with open(train_metrics_file, 'w') as f:
        json.dump(train_metrics, f, indent=2)

    # Save test metrics
    test_metrics_file = output_dir / f"window_{window_id}_test_metrics.json"
    with open(test_metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Save optimization history
    history_df = study.trials_dataframe()
    history_file = output_dir / f"window_{window_id}_optimization_history.csv"
    history_df.to_csv(history_file, index=False)

    logger.info(f"Window {window_id} complete!\n")

    return {
        'window_id': window_id,
        'best_params': best_params,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'wfe': wfe,
        'test_viability': test_viability,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimize base strategy parameters with walk-forward analysis'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Symbol to optimize (e.g., ES)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/resampled',
        help='Data directory (default: data/resampled)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: optimization_results/{SYMBOL})'
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path('optimization_results') / args.symbol

    windows_dir = output_root / 'windows'
    windows_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"BASE STRATEGY PARAMETER OPTIMIZATION")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Output: {output_root}")
    logger.info(f"{'='*80}\n")

    # Load full dataset
    logger.info(f"Loading data from {args.data_dir}...")
    full_data = load_data(args.symbol, data_dir=args.data_dir)

    # Run optimization for each window
    all_results = []

    for window in WINDOWS:
        # Load window data
        train_data = load_data(
            args.symbol,
            data_dir=args.data_dir,
            start_date=window['train_start'],
            end_date=window['train_end']
        )

        test_data = load_data(
            args.symbol,
            data_dir=args.data_dir,
            start_date=window['test_start'],
            end_date=window['test_end']
        )

        # Optimize window
        window_results = optimize_window(
            window,
            train_data,
            test_data,
            windows_dir,
            args.symbol
        )

        all_results.append(window_results)

    # CHECKPOINT: Check total OOS viability
    total_oos_trades = sum(r['test_metrics']['num_trades'] for r in all_results)
    oos_viability = check_total_oos_viability(total_oos_trades)

    logger.info(f"\n{'='*80}")
    logger.info(f"TOTAL OUT-OF-SAMPLE VIABILITY CHECK")
    logger.info(f"{'='*80}")
    logger.info(f"Total OOS Trades: {total_oos_trades:,}")
    logger.info(f"Status: {oos_viability['status']}")
    logger.info(f"Message: {oos_viability['message']}")

    if oos_viability['ml_config']:
        logger.info(f"\nRecommended ML Config:")
        for k, v in oos_viability['ml_config'].items():
            logger.info(f"  {k}: {v}")

    logger.info(f"{'='*80}\n")

    # Save aggregate results
    aggregate_file = output_root / 'analysis' / 'aggregate_walkforward_metrics.json'
    aggregate_file.parent.mkdir(parents=True, exist_ok=True)

    aggregate_data = {
        'symbol': args.symbol,
        'optimization_date': datetime.now().isoformat(),
        'total_oos_trades': total_oos_trades,
        'oos_viability': oos_viability,
        'windows': all_results,
    }

    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_data, f, indent=2, default=str)

    logger.info(f"Saved aggregate results to: {aggregate_file}")

    logger.info(f"\n{'='*80}")
    logger.info(f"WALK-FORWARD OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Next steps:")
    logger.info(f"  1. Run: python research/analyze_strategy_stability.py --symbol {args.symbol}")
    logger.info(f"  2. Run: python research/validate_strategy_heldout.py --symbol {args.symbol}")
    logger.info(f"  3. Run: python research/finalize_strategy_decision.py --symbol {args.symbol}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
