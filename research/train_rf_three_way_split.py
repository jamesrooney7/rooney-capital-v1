#!/usr/bin/env python3
"""
Three-Way Time Split Training for Random Forest with Bayesian Optimization.

This script eliminates threshold optimization bias by using separate temporal data splits:
    1. Training Period (2010-2018): Hyperparameter tuning (Random Search + Bayesian Opt)
    2. Threshold Period (2019-2020): Threshold optimization on separate data
    3. Test Period (2021-2024): Final evaluation on completely untouched data

This prevents "double-dipping" where both hyperparameters and threshold are optimized
on the same validation folds, which causes adaptive overfitting.

Usage:
    python research/train_rf_three_way_split.py \
        --symbol ES \
        --train-end 2018-12-31 \
        --threshold-end 2020-12-31 \
        --rs-trials 120 \
        --bo-trials 300 \
        --embargo-days 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

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
    profit_factor,
    portfolio_metrics_from_daily,
    ensure_daily_index,
)

import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_split_data(
    symbol: str,
    data_dir: str,
    train_end: str,
    threshold_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training data and split into three temporal periods.

    Args:
        symbol: Trading symbol (e.g., 'ES')
        data_dir: Directory containing training data
        train_end: End date for training period (e.g., '2018-12-31')
        threshold_end: End date for threshold period (e.g., '2020-12-31')

    Returns:
        Tuple of (Xy_train, X_train, Xy_threshold, X_threshold, Xy_test, X_test)
    """
    csv_path = Path(data_dir) / f"{symbol}_transformed_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}\n"
            f"Run: python research/extract_training_data.py --symbol {symbol}"
        )

    logger.info(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} trades with {len(df.columns)} columns")

    # Ensure we have date column
    if "Date/Time" in df.columns:
        df['Date'] = pd.to_datetime(df['Date/Time'])
    elif "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No date column found (expected 'Date/Time' or 'Date')")

    # Convert split dates
    train_end_dt = pd.Timestamp(train_end)
    threshold_end_dt = pd.Timestamp(threshold_end)

    # Split into three periods
    train_mask = df['Date'] <= train_end_dt
    threshold_mask = (df['Date'] > train_end_dt) & (df['Date'] <= threshold_end_dt)
    test_mask = df['Date'] > threshold_end_dt

    df_train = df[train_mask].copy()
    df_threshold = df[threshold_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"\n{'='*60}")
    logger.info("Data Split Summary:")
    logger.info(f"{'='*60}")
    logger.info(f"Training Period:   {df_train['Date'].min()} to {df_train['Date'].max()}")
    logger.info(f"  Trades: {len(df_train)}")
    logger.info(f"\nThreshold Period:  {df_threshold['Date'].min()} to {df_threshold['Date'].max()}")
    logger.info(f"  Trades: {len(df_threshold)}")
    logger.info(f"\nTest Period:       {df_test['Date'].min()} to {df_test['Date'].max()}")
    logger.info(f"  Trades: {len(df_test)} (HELD OUT - never touched until final eval)")
    logger.info(f"{'='*60}\n")

    # Prepare feature matrices (normalizes ALL column names to snake_case)
    # CRITICAL: Fit scaler on TRAINING data only, then transform all periods with same scaler
    # This prevents lookahead bias from using future data statistics
    logger.info("Normalizing features (fitting scaler on training data only)...")

    # FIT scaler on training data
    Xy_train, fitted_scaler = build_core_features(df_train, scaler=None, fit_scaler=True)

    # TRANSFORM threshold and test data with the SAME fitted scaler
    Xy_threshold, _ = build_core_features(df_threshold, scaler=fitted_scaler, fit_scaler=False)
    Xy_test, _ = build_core_features(df_test, scaler=fitted_scaler, fit_scaler=False)

    logger.info(f"✅ All periods normalized with same scaler (fit on training data only)")

    # Add back the date column and targets (normalized to lowercase)
    # Keep as datetime (not .dt.date) for pd.Grouper compatibility
    Xy_train["date"] = pd.to_datetime(df_train["Date/Time"])
    Xy_train["y_binary"] = df_train["y_binary"]
    Xy_train["y_return"] = df_train["y_return"]
    Xy_train["y_pnl_usd"] = df_train["y_pnl_usd"]

    Xy_threshold["date"] = pd.to_datetime(df_threshold["Date/Time"])
    Xy_threshold["y_binary"] = df_threshold["y_binary"]
    Xy_threshold["y_return"] = df_threshold["y_return"]
    Xy_threshold["y_pnl_usd"] = df_threshold["y_pnl_usd"]

    Xy_test["date"] = pd.to_datetime(df_test["Date/Time"])
    Xy_test["y_binary"] = df_test["y_binary"]
    Xy_test["y_return"] = df_test["y_return"]
    Xy_test["y_pnl_usd"] = df_test["y_pnl_usd"]

    # Extract feature columns (exclude metadata and targets)
    exclude_cols = {
        "date", "date_time", "exit_date_time", "entry_price", "exit_price",
        "y_return", "y_binary", "y_pnl_usd", "y_pnl_gross", "pnl_usd",
        "unnamed_0",
    }

    feature_cols = [col for col in Xy_train.columns if col not in exclude_cols]

    # Add engineered features
    X_train_raw = add_engineered(Xy_train[feature_cols])
    X_threshold_raw = add_engineered(Xy_threshold[feature_cols])
    X_test_raw = add_engineered(Xy_test[feature_cols])

    # Ensure all splits have same features (intersection)
    common_features = set(X_train_raw.columns) & set(X_threshold_raw.columns) & set(X_test_raw.columns)
    common_features = sorted(list(common_features))

    X_train = X_train_raw[common_features].copy()
    X_threshold = X_threshold_raw[common_features].copy()
    X_test = X_test_raw[common_features].copy()

    logger.info(f"Using {len(common_features)} features (intersection across all periods)")

    return Xy_train, X_train, Xy_threshold, X_threshold, Xy_test, X_test


def phase1_hyperparameter_tuning(
    Xy_train: pd.DataFrame,
    X_train: pd.DataFrame,
    symbol: str,
    seed: int,
    rs_trials: int,
    bo_trials: int,
    folds: int,
    k_test: int,
    embargo_days: int,
    k_features: int,
    screen_method: str,
) -> Tuple[Dict[str, Any], list, list]:
    """Phase 1: Hyperparameter tuning on training period (2010-2018).

    Uses Random Search + Bayesian Optimization with CPCV.

    Returns:
        Tuple of (best_params, feature_list, trial_history)
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: HYPERPARAMETER TUNING (Training Period)")
    logger.info(f"{'='*60}\n")

    rng = random.Random(seed)
    n_trials_total = rs_trials + bo_trials

    # Filter out problematic features BEFORE feature screening
    # This ensures we select k_features valid features (e.g., 30) after filtering
    logger.info("Filtering features before screening...")
    X_train_full = X_train.copy()

    # Remove duplicate cross-instrument feature columns
    # The training data has 3-4 versions of each cross-instrument feature:
    #   1. enableESReturnHour (boolean parameter - NOT the actual value)
    #   2. ES Hourly Return (Title Case - won't match runtime keys)
    #   3. es_hourly_return (snake_case - KEEP THIS)
    #   4. es_hourly_return_pipeline (snake_case with suffix - KEEP THIS)
    # We only want to keep #3 and #4 (snake_case versions with actual values)

    valid_cols = []
    filtered_title_case = 0
    filtered_enable_params = 0
    filtered_vix = 0

    cross_symbols = ["ES", "NQ", "RTY", "YM", "GC", "SI", "HG", "CL", "NG", "PL",
                     "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S", "TLT", "VIX"]

    for col in X_train_full.columns:
        # Skip Title Case cross-instrument features (e.g., "ES Hourly Return")
        if " " in col and any(col.startswith(sym + " ") for sym in cross_symbols):
            logger.debug(f"Filtering out Title Case duplicate: {col}")
            filtered_title_case += 1
            continue

        # Skip enable parameter columns for cross-instrument features
        # These are boolean/int parameters, not the actual feature values
        # Pattern: enable{SYMBOL}Return{Hour|Day} or enable{SYMBOL}ZScore{Hour|Day}
        if col.startswith("enable") and any(sym in col for sym in cross_symbols):
            if ("Return" in col or "ZScore" in col) and ("Hour" in col or "Day" in col):
                logger.debug(f"Filtering out enable parameter column: {col}")
                filtered_enable_params += 1
                continue

        # Skip VIX-related features (all variations: vix, VIX, enable_vix, etc.)
        if "vix" in col.lower():
            logger.debug(f"Filtering out VIX feature: {col}")
            filtered_vix += 1
            continue

        valid_cols.append(col)

    logger.info(f"Filtered out {filtered_title_case} Title Case duplicates, {filtered_enable_params} enable parameter columns, and {filtered_vix} VIX features")
    logger.info(f"Screening from {len(valid_cols)} valid candidate features")

    X_train_filtered = X_train_full[valid_cols].copy()

    # Feature screening on filtered feature set
    logger.info("Performing feature screening...")
    feats = screen_features(
        Xy_train,
        X_train_filtered,
        seed,
        method=screen_method,
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        top_n=k_features,
    )

    X_train_selected = X_train[feats].copy()
    logger.info(f"Selected {len(feats)} features: {', '.join(feats[:10])}{'...' if len(feats) > 10 else ''}")

    # Random Search
    logger.info(f"\nStarting Random Search ({rs_trials} trials)...")
    rs_rows = []
    for t in range(1, rs_trials + 1):
        params = sample_rf_params(rng)
        res = evaluate_rf_cpcv(
            Xy_train, X_train_selected, params,
            folds, k_test, embargo_days,
            n_trials_total=n_trials_total
        )
        rs_rows.append({**res, **params, "Trial": t, "Phase": "Random"})

        logger.info(
            f"[RS {t:03d}/{rs_trials}] "
            f"Sharpe={res['Sharpe']:.3f} DSR={res['DSR']:.3f} "
            f"PF={res['PF']:.3f} Trades={res['Trades']}"
        )

    rs_df = pd.DataFrame(rs_rows).sort_values("Sharpe", ascending=False)
    logger.info(f"\nRandom Search complete. Best Sharpe: {rs_df.iloc[0]['Sharpe']:.3f}")

    # Bayesian Optimization
    logger.info(f"\nStarting Bayesian Optimization ({bo_trials} trials)...")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # FIXED: Use FULL hyperparameter space for Bayesian optimization
        # Previous implementation constrained to top Random Search results,
        # which could miss the global optimum if Random Search found a local maximum.
        # Now using the same full space as Random Search (from sample_rf_params).
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

            if params["max_depth"] in [None, "None", np.nan]:
                params["max_depth"] = None
            else:
                params["max_depth"] = int(params["max_depth"])

            res = evaluate_rf_cpcv(
                Xy_train, X_train_selected, params,
                folds, k_test, embargo_days,
                n_trials_total=n_trials_total
            )

            trial.set_user_attr("PF", float(res["PF"]))
            trial.set_user_attr("Trades", int(res["Trades"]))
            trial.set_user_attr("DSR", float(res.get("DSR", np.nan)))

            # Log progress (trial.number is 0-indexed)
            logger.info(
                f"[BO {trial.number + 1:03d}/{bo_trials}] "
                f"Sharpe={res['Sharpe']:.3f} DSR={res.get('DSR', 0.0):.3f} "
                f"PF={res['PF']:.3f} Trades={res['Trades']}"
            )

            return float(res["Sharpe"])

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=bo_trials, show_progress_bar=False)

        bo_rows = []
        for tr in study.get_trials():
            p = tr.params.copy()
            p["max_depth"] = None if p.get("max_depth") in [None, "None", np.nan] else p.get("max_depth")
            bo_rows.append({
                "Trial": tr.number,
                "Phase": "BO",
                "Sharpe": tr.value,
                "PF": tr.user_attrs.get("PF", np.nan),
                "Trades": tr.user_attrs.get("Trades", 0),
                "DSR": tr.user_attrs.get("DSR", np.nan),
                **p
            })

        bo_df = pd.DataFrame(bo_rows).sort_values("Sharpe", ascending=False)
        logger.info(f"\nBayesian Optimization complete. Best Sharpe: {bo_df.iloc[0]['Sharpe']:.3f}")

    except ImportError:
        logger.warning("Optuna not available, skipping Bayesian Optimization")
        bo_df = pd.DataFrame()

    # Select best overall configuration
    all_trials = pd.concat([rs_df, bo_df], ignore_index=True).sort_values("Sharpe", ascending=False)
    best_row = all_trials.iloc[0]

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) or str(best_row["max_depth"]).lower() == "none" else int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_features": best_row["max_features"],
        "bootstrap": bool(best_row["bootstrap"]),
        "class_weight": None if pd.isna(best_row["class_weight"]) or str(best_row["class_weight"]).lower() == "none" else str(best_row["class_weight"]),
        "max_samples": best_row["max_samples"] if best_row["bootstrap"] else None,
        "n_jobs": -1,
        "random_state": seed,
    }

    logger.info(f"\n{'='*60}")
    logger.info("Phase 1 Complete - Best Hyperparameters:")
    logger.info(f"{'='*60}")
    logger.info(f"Sharpe: {best_row['Sharpe']:.3f}")
    logger.info(f"DSR: {best_row['DSR']:.3f}")
    logger.info(f"Profit Factor: {best_row['PF']:.3f}")
    logger.info(f"Trades: {int(best_row['Trades'])}")
    logger.info(f"\nParameters: {best_params}")
    logger.info(f"{'='*60}\n")

    return best_params, feats, all_trials.to_dict('records')


def phase2_threshold_optimization(
    Xy_train: pd.DataFrame,
    X_train: pd.DataFrame,
    Xy_threshold: pd.DataFrame,
    X_threshold: pd.DataFrame,
    best_params: Dict[str, Any],
    feature_list: list,
    min_trades: int = 100,
) -> Tuple[float, Dict[str, Any]]:
    """Phase 2: Fixed threshold validation on separate threshold period (2019-2020).

    Trains model on full training set with best hyperparameters,
    then validates performance using FIXED threshold = 0.50 (no optimization).

    NOTE: Threshold optimization removed to eliminate 5-15% optimistic bias from
    testing 31 thresholds on validation data. Using natural decision boundary (0.50)
    for calibrated Random Forest probabilities.

    Returns:
        Tuple of (fixed_threshold=0.50, threshold_metrics)
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: FIXED THRESHOLD VALIDATION (Threshold Period)")
    logger.info(f"{'='*60}\n")

    # Train final model on full training set with best hyperparameters
    from sklearn.ensemble import RandomForestClassifier

    logger.info("Training model on full training period (2010-2018)...")
    final_model = RandomForestClassifier(**best_params)

    X_train_selected = X_train[feature_list]
    y_train = Xy_train["y_binary"].values

    final_model.fit(X_train_selected, y_train)
    logger.info(f"Model trained on {len(X_train_selected)} samples")

    # Get predictions on SEPARATE threshold period (2019-2020)
    logger.info("\nPredicting on threshold period (2019-2020) - model has never seen this data...")
    X_threshold_selected = X_threshold[feature_list]
    y_threshold_proba = final_model.predict_proba(X_threshold_selected)[:, 1]

    returns_threshold = Xy_threshold["y_return"].values

    # Use fixed threshold = 0.50 (no optimization to avoid bias)
    logger.info("Using FIXED threshold = 0.50 (natural decision boundary for calibrated probabilities)")
    logger.info("Threshold optimization removed to eliminate 5-15% optimistic bias from multiple testing\n")

    fixed_threshold = 0.50

    # Calculate metrics at fixed threshold
    passed = y_threshold_proba >= fixed_threshold
    final_returns = returns_threshold[passed]
    final_dates = Xy_threshold.loc[passed, "date"]

    daily = pd.DataFrame({
        "d": final_dates,
        "r": final_returns
    }).groupby(pd.Grouper(key="d", freq="D"))["r"].sum()

    daily = ensure_daily_index(daily, final_dates)

    threshold_metrics = {
        "threshold": float(fixed_threshold),
        "trades": int(passed.sum()),
        "sharpe": sharpe_ratio_from_daily(daily),
        "profit_factor": profit_factor(pd.Series(Xy_threshold.loc[passed, "y_pnl_usd"])),
        "win_rate": float((Xy_threshold.loc[passed, "y_binary"] == 1).mean()) if passed.sum() > 0 else 0.0,
        "note": "Fixed threshold (0.50) used - no optimization to avoid bias",
    }

    logger.info(f"\n{'='*60}")
    logger.info("Phase 2 Complete - Fixed Threshold Validation:")
    logger.info(f"{'='*60}")
    logger.info(f"Threshold: {fixed_threshold:.2f} (FIXED - not optimized)")
    logger.info(f"Trades: {threshold_metrics['trades']}")
    logger.info(f"Sharpe: {threshold_metrics['sharpe']:.3f}")
    logger.info(f"Profit Factor: {threshold_metrics['profit_factor']:.3f}")
    logger.info(f"Win Rate: {threshold_metrics['win_rate']*100:.1f}%")
    logger.info(f"{'='*60}\n")

    return fixed_threshold, threshold_metrics


def phase3_final_evaluation(
    Xy_train: pd.DataFrame,
    X_train: pd.DataFrame,
    Xy_threshold: pd.DataFrame,
    X_threshold: pd.DataFrame,
    Xy_test: pd.DataFrame,
    X_test: pd.DataFrame,
    best_params: Dict[str, Any],
    best_threshold: float,
    feature_list: list,
) -> Dict[str, Any]:
    """Phase 3: Final evaluation on completely untouched test period (2021-2024).

    Retrains model on combined train+threshold periods (2010-2020),
    then evaluates on never-before-seen test period (2021-2024).

    Returns:
        Dictionary of final test set metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: FINAL EVALUATION (Test Period - UNTOUCHED DATA)")
    logger.info(f"{'='*60}\n")

    from sklearn.ensemble import RandomForestClassifier

    # Combine train + threshold periods for final model training
    logger.info("Training production model on combined train+threshold periods (2010-2020)...")

    Xy_combined = pd.concat([Xy_train, Xy_threshold], ignore_index=True)
    X_combined = pd.concat([X_train, X_threshold], ignore_index=True)

    X_combined_selected = X_combined[feature_list]
    y_combined = Xy_combined["y_binary"].values

    production_model = RandomForestClassifier(**best_params)
    production_model.fit(X_combined_selected, y_combined)

    logger.info(f"Production model trained on {len(X_combined_selected)} samples (2010-2020)")

    # Evaluate on COMPLETELY UNTOUCHED test period (2021-2024)
    logger.info("\nEvaluating on test period (2021-2024) - model has NEVER seen this data...")

    X_test_selected = X_test[feature_list]
    y_test_proba = production_model.predict_proba(X_test_selected)[:, 1]

    # Apply best threshold from phase 2
    passed = y_test_proba >= best_threshold

    logger.info(f"Threshold {best_threshold:.2f} passes {passed.sum()} / {len(passed)} trades")

    # Calculate metrics on test set
    test_returns = Xy_test.loc[passed, "y_return"].values
    test_pnl = Xy_test.loc[passed, "y_pnl_usd"].values
    test_dates = Xy_test.loc[passed, "date"]
    test_binary = Xy_test.loc[passed, "y_binary"].values

    # Daily returns for Sharpe
    daily = pd.DataFrame({
        "d": test_dates,
        "r": test_returns
    }).groupby(pd.Grouper(key="d", freq="D"))["r"].sum()

    daily = ensure_daily_index(daily, test_dates)

    # Portfolio metrics
    portfolio = portfolio_metrics_from_daily(daily)

    test_metrics = {
        "trades": int(passed.sum()),
        "sharpe": float(portfolio.get("Sharpe", 0.0)),
        "sortino": float(portfolio.get("Sortino", 0.0)),
        "profit_factor": profit_factor(pd.Series(test_pnl)),
        "win_rate": float((test_binary == 1).mean()) if len(test_binary) > 0 else 0.0,
        "total_pnl_usd": float(test_pnl.sum()),
        "max_drawdown_pct": float(portfolio.get("Max_Drawdown_Pct", 0.0)),
        "max_drawdown_usd": float(portfolio.get("Max_Drawdown_USD", 0.0)),
        "cagr": float(portfolio.get("CAGR", 0.0)),
        "start_date": str(Xy_test["date"].min()),
        "end_date": str(Xy_test["date"].max()),
    }

    logger.info(f"\n{'='*60}")
    logger.info("Phase 3 Complete - TRUE OUT-OF-SAMPLE PERFORMANCE:")
    logger.info(f"{'='*60}")
    logger.info(f"Test Period: {test_metrics['start_date']} to {test_metrics['end_date']}")
    logger.info(f"Trades: {test_metrics['trades']}")
    logger.info(f"Sharpe Ratio: {test_metrics['sharpe']:.3f}")
    logger.info(f"Sortino Ratio: {test_metrics['sortino']:.3f}")
    logger.info(f"Profit Factor: {test_metrics['profit_factor']:.3f}")
    logger.info(f"Win Rate: {test_metrics['win_rate']*100:.1f}%")
    logger.info(f"Total PnL: ${test_metrics['total_pnl_usd']:,.2f}")
    logger.info(f"CAGR: {test_metrics['cagr']*100:.1f}%")
    logger.info(f"Max Drawdown: {test_metrics['max_drawdown_pct']*100:.1f}%")
    logger.info(f"{'='*60}\n")

    return test_metrics, production_model


def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest with three-way time split to eliminate threshold optimization bias'
    )
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to train (e.g., ES)')
    parser.add_argument('--data-dir', type=str, default='data/training', help='Training data directory')
    parser.add_argument('--output-dir', type=str, default='src/models', help='Output directory for model')
    parser.add_argument('--train-end', type=str, default='2018-12-31',
                       help='End date for training period (hyperparameter tuning)')
    parser.add_argument('--threshold-end', type=str, default='2020-12-31',
                       help='End date for threshold period (threshold optimization)')
    parser.add_argument('--rs-trials', type=int, default=25, help='Random search trials (reduced from 120 for meta-labeling)')
    parser.add_argument('--bo-trials', type=int, default=65, help='Bayesian optimization trials (reduced from 300 for meta-labeling)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CPCV folds')
    parser.add_argument('--k-test', type=int, default=2, help='Number of test folds in CPCV')
    parser.add_argument('--embargo-days', type=int, default=2, help='Embargo days for CPCV (reduced from 5 to 2 for meta-labeling, 1-day hold + 1-day buffer)')
    parser.add_argument('--k-features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--screen-method', type=str, default='importance',
                       choices=['importance', 'permutation', 'l1', 'none'],
                       help='Feature screening method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--min-trades-threshold', type=int, default=100,
                       help='Minimum trades for threshold optimization')

    args = parser.parse_args()

    logger.info(f"\n{'='*60}")
    logger.info("THREE-WAY TIME SPLIT TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Training Period: [start] to {args.train_end}")
    logger.info(f"Threshold Period: {args.train_end} to {args.threshold_end}")
    logger.info(f"Test Period: {args.threshold_end} to [end]")
    logger.info(f"Random Search: {args.rs_trials} trials")
    logger.info(f"Bayesian Opt: {args.bo_trials} trials")
    logger.info(f"{'='*60}\n")

    # Load and split data
    Xy_train, X_train, Xy_threshold, X_threshold, Xy_test, X_test = load_and_split_data(
        args.symbol,
        args.data_dir,
        args.train_end,
        args.threshold_end,
    )

    # Phase 1: Hyperparameter tuning
    best_params, feature_list, trial_history = phase1_hyperparameter_tuning(
        Xy_train, X_train,
        args.symbol,
        args.seed,
        args.rs_trials,
        args.bo_trials,
        args.folds,
        args.k_test,
        args.embargo_days,
        args.k_features,
        args.screen_method,
    )

    # Phase 2: Threshold optimization
    best_threshold, threshold_metrics = phase2_threshold_optimization(
        Xy_train, X_train,
        Xy_threshold, X_threshold,
        best_params,
        feature_list,
        args.min_trades_threshold,
    )

    # Phase 3: Final evaluation
    test_metrics, production_model = phase3_final_evaluation(
        Xy_train, X_train,
        Xy_threshold, X_threshold,
        Xy_test, X_test,
        best_params,
        best_threshold,
        feature_list,
    )

    # Save model and metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{args.symbol}_rf_model.pkl"
    metadata_path = output_dir / f"{args.symbol}_best.json"
    test_results_path = output_dir / f"{args.symbol}_test_results.json"

    logger.info(f"\nSaving production model to: {model_path}")
    joblib.dump({
        'model': production_model,
        'features': feature_list,
    }, model_path)

    metadata = {
        "symbol": args.symbol,
        "threshold": best_threshold,
        "params": best_params,
        "features": feature_list,
        "train_period": {
            "start": str(Xy_train["date"].min()),
            "end": str(Xy_train["date"].max()),
            "trades": len(Xy_train),
        },
        "threshold_period": {
            "start": str(Xy_threshold["date"].min()),
            "end": str(Xy_threshold["date"].max()),
            "trades": len(Xy_threshold),
        },
        "threshold_optimization": threshold_metrics,
        "model_metadata": {
            "rs_trials": args.rs_trials,
            "bo_trials": args.bo_trials,
            "embargo_days": args.embargo_days,
            "screen_method": args.screen_method,
            "seed": args.seed,
        }
    }

    logger.info(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saving test results to: {test_results_path}")
    with open(test_results_path, 'w') as f:
        json.dump({
            "symbol": args.symbol,
            "test_metrics": test_metrics,
            "note": "This is TRUE out-of-sample performance on data never seen during training or threshold optimization",
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"\nModel saved: {model_path}")
    logger.info(f"Metadata saved: {metadata_path}")
    logger.info(f"Test results saved: {test_results_path}")
    logger.info(f"\n{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
