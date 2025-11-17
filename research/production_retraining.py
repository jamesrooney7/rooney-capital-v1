#!/usr/bin/env python3
"""
Production Model Retraining Framework with Anchored Walk-Forward

This script implements a production retraining policy that avoids forward-looking bias:

1. **Monthly Weight Retraining**: Fast model updates with fixed hyperparameters
2. **Annual Hyperparameter Re-optimization**: Full 420-trial optimization on anchored window
3. **Performance-Triggered Re-optimization**: Ad-hoc optimization if performance degrades

Key Design Principles:
- **Anchored Windows**: Optimization windows end BEFORE deployment period
- **No Look-Ahead**: Never optimize on data from the deployment period
- **Walk-Forward**: Each retraining uses expanding or rolling windows
- **Performance Monitoring**: Track live Sharpe vs expected, trigger retraining if degradation

Usage:
    # Monthly weight retraining (fast, keeps hyperparameters fixed)
    python research/production_retraining.py \
        --symbol SPY \
        --mode monthly \
        --existing-model models/SPY_rf_model.pkl

    # Annual hyperparameter re-optimization (full 420 trials)
    python research/production_retraining.py \
        --symbol SPY \
        --mode annual \
        --anchor-end 2024-12-31

    # Performance-triggered re-optimization
    python research/production_retraining.py \
        --symbol SPY \
        --mode performance \
        --current-sharpe 0.25 \
        --expected-sharpe 0.45
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import optimization utility functions
from research.rf_optimization_utils import (
    sample_rf_params,
    evaluate_rf_cpcv,
    screen_features,
    build_core_features,
    add_engineered,
    deflated_sharpe_ratio,
    sharpe_ratio_from_daily,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_model(model_path: str) -> Dict[str, Any]:
    """Load existing production model with metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = joblib.load(model_path)
    logger.info(f"Loaded existing model from {model_path}")
    logger.info(f"  Training date: {model_data.get('training_date', 'unknown')}")
    logger.info(f"  Hyperparameters: {model_data.get('hyperparameters', {})}")
    logger.info(f"  Features: {len(model_data.get('features', []))} features")

    return model_data


def monthly_weight_retraining(
    symbol: str,
    existing_model_path: str,
    data_dir: str = "training_data",
    window_type: str = "expanding",  # "expanding" or "rolling"
    rolling_years: int = 10,
) -> Dict[str, Any]:
    """
    Monthly retraining: Update Random Forest weights only (fast).

    Keeps hyperparameters FIXED from original optimization to avoid forward-looking bias.
    Only retrains model weights on new data.

    Args:
        symbol: Trading symbol (e.g., "SPY")
        existing_model_path: Path to current production model
        data_dir: Directory containing training data
        window_type: "expanding" (all historical data) or "rolling" (last N years)
        rolling_years: If window_type="rolling", use last N years

    Returns:
        Dict with new model, metadata, and performance metrics
    """
    logger.info("="*80)
    logger.info("MONTHLY WEIGHT RETRAINING (Hyperparameters Fixed)")
    logger.info("="*80)

    # Load existing model and hyperparameters
    model_data = load_existing_model(existing_model_path)
    hyperparams = model_data.get('hyperparameters', {})
    feature_list = model_data.get('features', [])
    threshold = model_data.get('threshold', 0.50)

    logger.info(f"\nUsing FIXED hyperparameters from original optimization:")
    for k, v in hyperparams.items():
        logger.info(f"  {k}: {v}")

    # Load training data
    data_path = os.path.join(data_dir, f"{symbol}_training_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    Xy = pd.read_csv(data_path)
    logger.info(f"\nLoaded {len(Xy)} samples from {data_path}")

    # Determine training window
    Xy['Date'] = pd.to_datetime(Xy['Date'])
    anchor_date = datetime.now().date()

    if window_type == "rolling":
        start_date = anchor_date - timedelta(days=rolling_years * 365)
        Xy = Xy[Xy['Date'].dt.date >= start_date].copy()
        logger.info(f"Using rolling {rolling_years}-year window: {start_date} to {anchor_date}")
    else:
        logger.info(f"Using expanding window: all data up to {anchor_date}")

    # Build features
    X = build_core_features(Xy)
    X = add_engineered(X)

    # Select features (same as original model)
    if not all(f in X.columns for f in feature_list):
        missing = [f for f in feature_list if f not in X.columns]
        raise ValueError(f"Missing features in new data: {missing}")

    X = X[feature_list].copy()
    y = Xy['y_binary'].values

    logger.info(f"\nRetraining on {len(X)} samples with {len(feature_list)} features")
    logger.info(f"Class balance: {(y==1).sum()} wins / {(y==0).sum()} losses")

    # Retrain Random Forest with FIXED hyperparameters
    model = RandomForestClassifier(**hyperparams)
    model.fit(X, y)

    logger.info("✅ Model weights retrained successfully")

    # Create new model bundle
    new_model_data = {
        'model': model,
        'hyperparameters': hyperparams,
        'features': feature_list,
        'threshold': threshold,
        'symbol': symbol,
        'retraining_type': 'monthly_weights',
        'training_date': datetime.now().isoformat(),
        'training_window': {
            'start': Xy['Date'].min().isoformat() if not Xy.empty else None,
            'end': Xy['Date'].max().isoformat() if not Xy.empty else None,
            'window_type': window_type,
        },
        'training_samples': len(X),
        'previous_model': existing_model_path,
    }

    # Save new model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("models", f"{symbol}_monthly_retrain_{timestamp}.pkl")
    os.makedirs("models", exist_ok=True)
    joblib.dump(new_model_data, output_path)
    logger.info(f"\n✅ New model saved: {output_path}")

    return new_model_data


def annual_hyperparameter_reoptimization(
    symbol: str,
    anchor_end: str,
    data_dir: str = "training_data",
    rs_trials: int = 120,
    bo_trials: int = 300,
    folds: int = 5,
    k_test: int = 2,
    embargo_days: int = 2,
    k_features: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Annual re-optimization: Full 420-trial hyperparameter search on anchored window.

    CRITICAL: anchor_end MUST be BEFORE the intended deployment period to avoid look-ahead bias.

    Example:
        - Optimize on data up to 2024-12-31
        - Deploy model for 2025 onwards
        - This ensures optimization never "sees" 2025 data

    Args:
        symbol: Trading symbol
        anchor_end: End date for optimization window (YYYY-MM-DD)
        data_dir: Training data directory
        rs_trials: Random search trials (default: 120)
        bo_trials: Bayesian optimization trials (default: 300)
        folds: CPCV folds (default: 5)
        k_test: Test folds per combination (default: 2)
        embargo_days: Embargo period (default: 2, reduced from 5 for meta-labeling)
        k_features: Number of features to select (default: 30)
        seed: Random seed

    Returns:
        Dict with optimized model, best hyperparameters, and metrics
    """
    logger.info("="*80)
    logger.info("ANNUAL HYPERPARAMETER RE-OPTIMIZATION (Full 420-Trial Search)")
    logger.info("="*80)

    anchor_date = pd.to_datetime(anchor_end).date()
    logger.info(f"\n⚠️  CRITICAL: Anchored optimization window")
    logger.info(f"   Optimization ends: {anchor_date}")
    logger.info(f"   Deploy for period: AFTER {anchor_date}")
    logger.info(f"   This ensures NO forward-looking bias\n")

    # Load training data
    data_path = os.path.join(data_dir, f"{symbol}_training_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    Xy = pd.read_csv(data_path)
    Xy['Date'] = pd.to_datetime(Xy['Date'])

    # Filter to anchored window (all data UP TO anchor_end)
    Xy = Xy[Xy['Date'].dt.date <= anchor_date].copy()
    logger.info(f"Loaded {len(Xy)} samples up to {anchor_date}")

    if len(Xy) < 500:
        raise ValueError(f"Insufficient data for optimization: {len(Xy)} samples (need 500+)")

    # Build features
    X = build_core_features(Xy)
    X = add_engineered(X)

    # Feature screening
    logger.info(f"\nPerforming feature screening (top {k_features} features)...")
    feats = screen_features(
        Xy, X, seed,
        method='importance',
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        top_n=k_features,
    )

    X_selected = X[feats].copy()
    logger.info(f"Selected {len(feats)} features: {', '.join(feats[:10])}{'...' if len(feats) > 10 else ''}")

    # Random Search (120 trials)
    logger.info(f"\nStarting Random Search ({rs_trials} trials)...")
    import random
    rng = random.Random(seed)
    n_trials_total = rs_trials + bo_trials

    rs_rows = []
    for t in range(1, rs_trials + 1):
        params = sample_rf_params(rng)
        res = evaluate_rf_cpcv(
            Xy, X_selected, params,
            folds, k_test, embargo_days,
            n_trials_total=n_trials_total
        )
        rs_rows.append({**res, **params, "Trial": t, "Phase": "Random"})
        logger.info(
            f"[RS {t:03d}/{rs_trials}] "
            f"Sharpe={res['Sharpe']:.3f} DSR={res['DSR']:.3f} "
            f"Trades={res['Trades']}"
        )

    rs_df = pd.DataFrame(rs_rows).sort_values("Sharpe", ascending=False)
    best_rs_sharpe = rs_df.iloc[0]['Sharpe']
    logger.info(f"\n✅ Random Search complete. Best Sharpe: {best_rs_sharpe:.3f}")

    # Bayesian Optimization (300 trials)
    logger.info(f"\nStarting Bayesian Optimization ({bo_trials} trials)...")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Use FULL hyperparameter space (no hard bounds)
        est_range = [300, 600, 900, 1200]
        depth_opts = [3, 5, 7, None]
        leaf_range = [50, 100, 200]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", est_range),
                "max_depth": trial.suggest_categorical("max_depth", depth_opts),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", leaf_range),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
                "n_jobs": -1,
                "random_state": seed + trial.number,
            }

            if params["bootstrap"]:
                params["max_samples"] = trial.suggest_float("max_samples", 0.6, 0.95)
            else:
                params["max_samples"] = None

            res = evaluate_rf_cpcv(
                Xy, X_selected, params,
                folds, k_test, embargo_days,
                n_trials_total=n_trials_total
            )

            trial.set_user_attr("DSR", res.get("DSR", np.nan))
            trial.set_user_attr("Trades", res["Trades"])

            return res["Sharpe"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=bo_trials, show_progress_bar=True)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_sharpe = best_trial.value
        best_dsr = best_trial.user_attrs.get("DSR", np.nan)

        logger.info(f"\n✅ Bayesian Optimization complete. Best Sharpe: {best_sharpe:.3f}, DSR: {best_dsr:.3f}")
        logger.info(f"Best hyperparameters: {best_params}")

    except ImportError:
        logger.warning("Optuna not available, using best Random Search result")
        best_params = rs_df.iloc[0].to_dict()
        best_sharpe = best_rs_sharpe

    # Train final model on all anchored data
    logger.info("\nTraining final model on full anchored dataset...")
    final_model = RandomForestClassifier(**best_params)
    y = Xy['y_binary'].values
    final_model.fit(X_selected, y)

    # Optimize threshold (placeholder - can use CPCV if desired)
    threshold = 0.50  # Default threshold

    # Create model bundle
    model_data = {
        'model': final_model,
        'hyperparameters': best_params,
        'features': feats,
        'threshold': threshold,
        'symbol': symbol,
        'retraining_type': 'annual_hyperparameters',
        'training_date': datetime.now().isoformat(),
        'anchor_end': anchor_end,
        'training_window': {
            'start': Xy['Date'].min().isoformat(),
            'end': Xy['Date'].max().isoformat(),
        },
        'training_samples': len(X_selected),
        'optimization_metrics': {
            'best_sharpe': float(best_sharpe),
            'best_dsr': float(best_dsr) if not np.isnan(best_dsr) else None,
            'rs_trials': rs_trials,
            'bo_trials': bo_trials,
        },
    }

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join("models", f"{symbol}_annual_reopt_{timestamp}.pkl")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_data, output_path)
    logger.info(f"\n✅ Optimized model saved: {output_path}")

    return model_data


def performance_triggered_reoptimization(
    symbol: str,
    current_sharpe: float,
    expected_sharpe: float,
    degradation_threshold: float = 0.30,
    anchor_end: Optional[str] = None,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Performance-triggered re-optimization: Run full optimization if performance degrades.

    Args:
        symbol: Trading symbol
        current_sharpe: Current live Sharpe ratio
        expected_sharpe: Expected Sharpe from backtest
        degradation_threshold: Trigger re-optimization if current < expected * (1 - threshold)
        anchor_end: End date for optimization window (defaults to yesterday)
        **kwargs: Additional args passed to annual_hyperparameter_reoptimization

    Returns:
        New model if triggered, else None
    """
    logger.info("="*80)
    logger.info("PERFORMANCE-TRIGGERED RE-OPTIMIZATION CHECK")
    logger.info("="*80)

    logger.info(f"\nCurrent Sharpe: {current_sharpe:.3f}")
    logger.info(f"Expected Sharpe: {expected_sharpe:.3f}")

    degradation_factor = 1 - degradation_threshold
    trigger_level = expected_sharpe * degradation_factor

    logger.info(f"Degradation threshold: {degradation_threshold*100:.0f}%")
    logger.info(f"Trigger level: {trigger_level:.3f}")

    if current_sharpe >= trigger_level:
        logger.info(f"\n✅ Performance acceptable ({current_sharpe:.3f} >= {trigger_level:.3f})")
        logger.info("No re-optimization needed.")
        return None

    logger.info(f"\n⚠️  Performance degraded: {current_sharpe:.3f} < {trigger_level:.3f}")
    logger.info(f"Triggering full hyperparameter re-optimization...\n")

    # Default anchor_end to yesterday if not specified
    if anchor_end is None:
        anchor_end = (datetime.now().date() - timedelta(days=1)).isoformat()

    # Run full re-optimization
    return annual_hyperparameter_reoptimization(
        symbol=symbol,
        anchor_end=anchor_end,
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Production Model Retraining with Anchored Walk-Forward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., SPY)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['monthly', 'annual', 'performance'],
                       help='Retraining mode: monthly (weights only), annual (full optimization), performance (triggered)')

    # Monthly mode args
    parser.add_argument('--existing-model', type=str, help='Path to existing model (for monthly mode)')
    parser.add_argument('--window-type', type=str, default='expanding',
                       choices=['expanding', 'rolling'],
                       help='Training window: expanding (all data) or rolling (last N years)')
    parser.add_argument('--rolling-years', type=int, default=10,
                       help='If window_type=rolling, use last N years (default: 10)')

    # Annual mode args
    parser.add_argument('--anchor-end', type=str, help='End date for anchored optimization (YYYY-MM-DD)')
    parser.add_argument('--rs-trials', type=int, default=25, help='Random search trials (reduced from 120 for meta-labeling)')
    parser.add_argument('--bo-trials', type=int, default=65, help='Bayesian optimization trials (reduced from 300 for meta-labeling)')
    parser.add_argument('--folds', type=int, default=5, help='CPCV folds')
    parser.add_argument('--k-test', type=int, default=2, help='Test folds per combination')
    parser.add_argument('--embargo-days', type=int, default=2, help='Embargo period (reduced from 5 to 2 for meta-labeling)')
    parser.add_argument('--k-features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Performance mode args
    parser.add_argument('--current-sharpe', type=float, help='Current live Sharpe ratio')
    parser.add_argument('--expected-sharpe', type=float, help='Expected Sharpe from backtest')
    parser.add_argument('--degradation-threshold', type=float, default=0.30,
                       help='Trigger re-optimization if current < expected * (1 - threshold) (default: 0.30)')

    # Common args
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Training data directory')

    args = parser.parse_args()

    # Validate mode-specific args
    if args.mode == 'monthly':
        if not args.existing_model:
            parser.error("--existing-model required for monthly mode")

        result = monthly_weight_retraining(
            symbol=args.symbol,
            existing_model_path=args.existing_model,
            data_dir=args.data_dir,
            window_type=args.window_type,
            rolling_years=args.rolling_years,
        )

    elif args.mode == 'annual':
        if not args.anchor_end:
            # Default to yesterday
            args.anchor_end = (datetime.now().date() - timedelta(days=1)).isoformat()
            logger.info(f"No --anchor-end specified, using yesterday: {args.anchor_end}")

        result = annual_hyperparameter_reoptimization(
            symbol=args.symbol,
            anchor_end=args.anchor_end,
            data_dir=args.data_dir,
            rs_trials=args.rs_trials,
            bo_trials=args.bo_trials,
            folds=args.folds,
            k_test=args.k_test,
            embargo_days=args.embargo_days,
            k_features=args.k_features,
            seed=args.seed,
        )

    elif args.mode == 'performance':
        if args.current_sharpe is None or args.expected_sharpe is None:
            parser.error("--current-sharpe and --expected-sharpe required for performance mode")

        result = performance_triggered_reoptimization(
            symbol=args.symbol,
            current_sharpe=args.current_sharpe,
            expected_sharpe=args.expected_sharpe,
            degradation_threshold=args.degradation_threshold,
            anchor_end=args.anchor_end,
            data_dir=args.data_dir,
            rs_trials=args.rs_trials,
            bo_trials=args.bo_trials,
            folds=args.folds,
            k_test=args.k_test,
            embargo_days=args.embargo_days,
            k_features=args.k_features,
            seed=args.seed,
        )

        if result is None:
            logger.info("\n✅ No retraining triggered (performance acceptable)")
            return

    logger.info("\n" + "="*80)
    logger.info("RETRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbol: {args.symbol}")
    if result:
        logger.info(f"Output: {result.get('model', 'N/A')}")


if __name__ == '__main__':
    main()
