#!/usr/bin/env python3
"""
ML Meta-Labeling System - Main Orchestrator

End-to-end ML meta-labeling optimization using LightGBM ensemble with
walk-forward validation and hierarchical feature selection.

Usage:
    python ml_meta_labeling_optimizer.py --symbol ES

Author: Rooney Capital
Date: 2025-01-18
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.ml_meta_labeling.components.data_preparation import DataPreparation
from research.ml_meta_labeling.components.feature_selection import FeatureSelection
from research.ml_meta_labeling.components.walk_forward import WalkForwardValidator
from research.ml_meta_labeling.components.ensemble import EnsembleModel
from research.ml_meta_labeling.components.lightgbm_trainer import LightGBMTrainer
from research.ml_meta_labeling.components.config_defaults import (
    HELD_OUT_TEST,
    OUTPUT_TEMPLATES
)
from research.ml_meta_labeling.utils.reporting import (
    generate_executive_summary,
    save_walk_forward_results_csv,
    save_hyperparameter_stability_csv
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    """Main orchestrator for ML meta-labeling system."""
    logger.info("=" * 100)
    logger.info("ML META-LABELING SYSTEM - OPTIMIZATION PIPELINE")
    logger.info("=" * 100)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Task Type: {args.task_type}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output_dir) / args.symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # ============================================================================
    # COMPONENT 1: DATA PREPARATION
    # ============================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("COMPONENT 1: DATA PREPARATION")
    logger.info("=" * 100)

    data_prep = DataPreparation(
        symbol=args.symbol,
        data_dir=args.data_dir,
        remove_title_case=args.remove_title_case,
        remove_enable_params=args.remove_enable_params,
        remove_vix=args.remove_vix,
        lambda_decay=args.lambda_decay,
        min_samples_per_class=args.min_samples_per_class,
        missing_value_threshold=args.missing_value_threshold,
        task_type=args.task_type
    )

    df, feature_columns = data_prep.load_and_prepare()
    logger.info(f"Data loaded: {len(df)} samples, {len(feature_columns)} features")

    # Save data summary
    summary_stats = data_prep.get_summary_stats()
    with open(output_dir / f"{args.symbol}_data_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    # ============================================================================
    # COMPONENT 2: FEATURE SELECTION
    # ============================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("COMPONENT 2: FEATURE CLUSTERING AND SELECTION")
    logger.info("=" * 100)

    # Get development period data for feature selection
    dev_df = data_prep.get_date_split(
        start_date="2011-01-01",
        end_date="2020-12-31"
    )

    X_dev, y_dev, sw_dev = data_prep.get_features_and_target(dev_df)

    feature_selector = FeatureSelection(
        n_clusters=args.n_clusters,
        linkage_method=args.linkage_method,
        rf_n_estimators=args.rf_n_estimators,
        random_state=args.seed
    )

    selected_features, selection_report = feature_selector.select_features(
        X_dev,
        y_dev,
        sw_dev
    )

    logger.info(f"Selected {len(selected_features)} features from {len(feature_columns)}")

    # Save selected features
    with open(output_dir / OUTPUT_TEMPLATES['selected_features'].format(symbol=args.symbol), 'w') as f:
        json.dump(selection_report, f, indent=2, default=str)

    # Save feature selection summary
    selection_summary = feature_selector.get_selection_summary()
    (output_dir / OUTPUT_TEMPLATES['feature_clustering_report'].format(symbol=args.symbol)).write_text(
        selection_summary
    )

    # ============================================================================
    # COMPONENT 6: WALK-FORWARD VALIDATION
    # ============================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("COMPONENT 6: WALK-FORWARD VALIDATION")
    logger.info("=" * 100)

    walk_forward = WalkForwardValidator(
        data_prep=data_prep,
        selected_features=selected_features,
        n_trials_per_window=args.n_trials,
        cv_folds=args.cv_folds,
        embargo_days=args.embargo_days,
        optimization_metric=args.optimization_metric,
        precision_threshold=args.precision_threshold,
        random_state=args.seed,
        task_type=args.task_type,
        output_dir=output_dir
    )

    window_results, oos_predictions = walk_forward.run_walk_forward()

    # Save walk-forward results
    save_walk_forward_results_csv(
        window_results,
        output_dir / OUTPUT_TEMPLATES['walk_forward_results'].format(symbol=args.symbol)
    )

    save_hyperparameter_stability_csv(
        window_results,
        output_dir / "hyperparameter_stability.csv"
    )

    oos_predictions.to_csv(
        output_dir / OUTPUT_TEMPLATES['oos_predictions'].format(symbol=args.symbol),
        index=False
    )

    # ============================================================================
    # HELD-OUT TEST EVALUATION (2021-2024)
    # ============================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("HELD-OUT TEST PERIOD EVALUATION (2021-2024)")
    logger.info("=" * 100)

    # Get training data (full development period 2011-2020)
    train_full_df = data_prep.get_date_split(
        start_date="2011-01-01",
        end_date="2020-12-31"
    )

    # Get held-out test data (2021-2024)
    test_df = data_prep.get_date_split(
        start_date=HELD_OUT_TEST['start'],
        end_date=HELD_OUT_TEST['end']
    )

    logger.info(f"Training samples (2011-2020): {len(train_full_df)}")
    logger.info(f"Test samples (2021-2024): {len(test_df)}")

    # Extract features and targets
    X_train_full, y_train_full, sw_train_full = data_prep.get_features_and_target(train_full_df)
    X_test, y_test, sw_test = data_prep.get_features_and_target(test_df)

    # Select features
    X_train_selected = X_train_full[selected_features]
    X_test_selected = X_test[selected_features]

    # Get best hyperparameters from walk-forward (use last window)
    best_hyperparams = window_results[-1]['best_hyperparameters']

    if args.use_ensemble:
        # ============================================================================
        # COMPONENT 7: ENSEMBLE MODEL
        # ============================================================================
        logger.info("")
        logger.info("=" * 100)
        logger.info("COMPONENT 7: ENSEMBLE MODEL TRAINING")
        logger.info("=" * 100)

        # Add Date column for CV
        X_train_with_date = X_train_selected.copy()
        X_train_with_date['Date'] = train_full_df['Date'].values

        ensemble = EnsembleModel(
            lightgbm_params=best_hyperparams,
            random_state=args.seed
        )

        ensemble.train_ensemble(
            X_train_with_date,
            y_train_full,
            sw_train_full,
            cv_folds=args.cv_folds,
            embargo_days=args.embargo_days
        )

        # Predict on held-out test
        if args.task_type == 'classification':
            y_pred = ensemble.predict_proba(X_test_selected)
        else:  # regression
            # Note: Ensemble doesn't support regression yet, so this would error
            logger.warning("Ensemble mode does not support regression yet. Use --no-use-ensemble flag.")
            y_pred = ensemble.predict_proba(X_test_selected)  # Will error if reached

        # Save ensemble
        ensemble.save_ensemble(
            str(output_dir / OUTPUT_TEMPLATES['final_model'].format(symbol=args.symbol))
        )

        # Save ensemble weights
        with open(output_dir / OUTPUT_TEMPLATES['ensemble_weights'].format(symbol=args.symbol), 'w') as f:
            json.dump(ensemble.get_model_weights(), f, indent=2)

    else:
        # ============================================================================
        # SINGLE LIGHTGBM MODEL
        # ============================================================================
        logger.info("")
        logger.info("Training final LightGBM model on 2011-2020...")

        trainer = LightGBMTrainer(
            hyperparameters=best_hyperparams,
            random_state=args.seed,
            task_type=args.task_type
        )

        trainer.train(X_train_selected, y_train_full, sw_train_full)

        # Predict on held-out test
        if args.task_type == 'classification':
            y_pred = trainer.predict_proba(X_test_selected)
        else:  # regression
            y_pred = trainer.predict(X_test_selected)

        # Save model
        trainer.save_model(
            str(output_dir / OUTPUT_TEMPLATES['final_model'].format(symbol=args.symbol))
        )

    # Calculate held-out metrics

    # Save held-out predictions to CSV
    if args.task_type == 'classification':
        held_out_pred_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'y_true': y_test,
            'y_pred_proba': y_pred,
            'y_pred_binary': (y_pred >= 0.5).astype(int)
        })
    else:  # regression
        held_out_pred_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'y_true_pnl': y_test,
            'y_pred_pnl': y_pred
        })

    if 'y_return' in test_df.columns:
        held_out_pred_df['y_return'] = test_df['y_return'].values

    held_out_pred_df.to_csv(
        output_dir / f"{args.symbol}_ml_meta_labeling_held_out_predictions.csv",
        index=False
    )
    logger.info(f"Saved held-out predictions to CSV")

    # Calculate metrics based on task type
    if args.task_type == 'classification':
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        # Filter trades based on ML threshold (0.50)
        threshold = 0.5
        filter_mask = y_pred >= threshold

        # Unfiltered metrics (for comparison)
        unfiltered_metrics = {
            'n_trades': len(y_test),
            'win_rate': y_test.mean(),
        }

        if 'y_return' in test_df.columns:
            returns_unfiltered = test_df['y_return'].values
            from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics
            unfiltered_perf = calculate_performance_metrics(returns_unfiltered)
            unfiltered_metrics['sharpe_ratio'] = unfiltered_perf['sharpe_ratio']
            unfiltered_metrics['profit_factor'] = unfiltered_perf['profit_factor']

        # Filtered metrics (actual ML meta-labeling performance)
        y_test_filtered = y_test[filter_mask]

        held_out_results = {
            'task_type': 'classification',
            'threshold': threshold,
            'auc': roc_auc_score(y_test, y_pred),
            'precision': precision_score(y_test, (y_pred >= threshold).astype(int), zero_division=0),
            'recall': recall_score(y_test, (y_pred >= threshold).astype(int), zero_division=0),
            'f1': f1_score(y_test, (y_pred >= threshold).astype(int), zero_division=0),
            'n_trades_unfiltered': unfiltered_metrics['n_trades'],
            'n_trades_filtered': int(filter_mask.sum()),
            'filter_rate': float(1 - filter_mask.mean()),
            'win_rate_unfiltered': float(unfiltered_metrics['win_rate']),
            'win_rate_filtered': float(y_test_filtered.mean()) if len(y_test_filtered) > 0 else 0.0,
        }

        # Calculate financial metrics on FILTERED trades
        if 'y_return' in test_df.columns and filter_mask.sum() > 0:
            returns_filtered = test_df['y_return'].values[filter_mask]
            from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics

            perf_metrics = calculate_performance_metrics(returns_filtered)
            held_out_results.update({
                'sharpe_ratio_unfiltered': unfiltered_metrics['sharpe_ratio'],
                'sharpe_ratio_filtered': perf_metrics['sharpe_ratio'],
                'profit_factor_unfiltered': unfiltered_metrics['profit_factor'],
                'profit_factor_filtered': perf_metrics['profit_factor'],
                'sortino_ratio': perf_metrics['sortino_ratio'],
                'calmar_ratio': perf_metrics['calmar_ratio'],
                'max_drawdown': perf_metrics['max_drawdown'],
                'total_return': perf_metrics['total_return'],
            })

        logger.info("Held-Out Test Results (2021-2024):")
        logger.info(f"  AUC:                    {held_out_results['auc']:.4f}")
        logger.info(f"  Threshold:              {threshold:.2f}")
        logger.info("")
        logger.info("  UNFILTERED (Primary Strategy):")
        logger.info(f"    Trades:               {held_out_results['n_trades_unfiltered']}")
        logger.info(f"    Win Rate:             {held_out_results['win_rate_unfiltered']:.2%}")
        logger.info(f"    Sharpe:               {held_out_results.get('sharpe_ratio_unfiltered', 0):.3f}")
        logger.info(f"    Profit Factor:        {held_out_results.get('profit_factor_unfiltered', 0):.2f}")
        logger.info("")
        logger.info("  FILTERED (ML Meta-Labeling):")
        logger.info(f"    Trades:               {held_out_results['n_trades_filtered']}")
        logger.info(f"    Filter Rate:          {held_out_results['filter_rate']:.1%}")
        logger.info(f"    Win Rate:             {held_out_results['win_rate_filtered']:.2%}")
        logger.info(f"    Sharpe:               {held_out_results.get('sharpe_ratio_filtered', 0):.3f}")
        logger.info(f"    Profit Factor:        {held_out_results.get('profit_factor_filtered', 0):.2f}")

    else:  # regression
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        held_out_results = {
            'task_type': 'regression',
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mean_true_pnl': float(y_test.mean()),
            'mean_pred_pnl': float(y_pred.mean()),
            'n_samples': len(y_test)
        }

        # Calculate financial metrics on all trades
        if 'y_return' in test_df.columns:
            returns = test_df['y_return'].values
            from research.ml_meta_labeling.utils.metrics import calculate_performance_metrics

            perf_metrics = calculate_performance_metrics(returns)
            held_out_results.update({
                'sharpe_ratio': perf_metrics['sharpe_ratio'],
                'profit_factor': perf_metrics['profit_factor'],
                'sortino_ratio': perf_metrics['sortino_ratio'],
                'calmar_ratio': perf_metrics['calmar_ratio'],
                'max_drawdown': perf_metrics['max_drawdown'],
                'total_return': perf_metrics['total_return'],
            })

        logger.info("Held-Out Test Results (2021-2024):")
        logger.info(f"  R²:                     {held_out_results['r2']:.4f}")
        logger.info(f"  MAE:                    ${held_out_results['mae']:.2f}")
        logger.info(f"  RMSE:                   ${held_out_results['rmse']:.2f}")
        logger.info(f"  Mean True P&L:          ${held_out_results['mean_true_pnl']:.2f}")
        logger.info(f"  Mean Predicted P&L:     ${held_out_results['mean_pred_pnl']:.2f}")
        logger.info(f"  Samples:                {held_out_results['n_samples']}")
        logger.info(f"  Sharpe:                 {held_out_results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Profit Factor:          {held_out_results.get('profit_factor', 0):.2f}")

    # Save held-out results
    with open(output_dir / OUTPUT_TEMPLATES['held_out_results'].format(symbol=args.symbol), 'w') as f:
        json.dump(held_out_results, f, indent=2)

    # ============================================================================
    # GENERATE EXECUTIVE SUMMARY
    # ============================================================================
    logger.info("")
    logger.info("Generating executive summary...")

    generate_executive_summary(
        symbol=args.symbol,
        walk_forward_results=window_results,
        held_out_results=held_out_results,
        selected_features=selected_features,
        output_path=output_dir / OUTPUT_TEMPLATES['executive_summary'].format(symbol=args.symbol)
    )

    # ============================================================================
    # COMPLETE
    # ============================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("ML META-LABELING OPTIMIZATION COMPLETE!")
    logger.info("=" * 100)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Key files:")
    logger.info(f"  - Executive Summary: {OUTPUT_TEMPLATES['executive_summary'].format(symbol=args.symbol)}")
    logger.info(f"  - Final Model: {OUTPUT_TEMPLATES['final_model'].format(symbol=args.symbol)}")
    logger.info(f"  - Walk-Forward Results: {OUTPUT_TEMPLATES['walk_forward_results'].format(symbol=args.symbol)}")
    logger.info(f"  - Selected Features: {OUTPUT_TEMPLATES['selected_features'].format(symbol=args.symbol)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Meta-Labeling System for Trading Signal Filtering"
    )

    # Required arguments
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ES)")

    # Data arguments
    parser.add_argument("--data-dir", default="data/training", help="Training data directory")
    parser.add_argument("--output-dir", default="research/ml_meta_labeling/results", help="Output directory")

    # Feature filtering
    parser.add_argument("--remove-title-case", action="store_true", default=True, help="Remove columns with spaces")
    parser.add_argument("--remove-enable-params", action="store_true", default=True, help="Remove enable* columns")
    parser.add_argument("--remove-vix", action="store_true", default=True, help="Remove VIX features")

    # Feature selection
    parser.add_argument("--n-clusters", type=int, default=30, help="Number of feature clusters")
    parser.add_argument("--linkage-method", default="ward", choices=["ward", "complete"], help="Clustering linkage")
    parser.add_argument("--rf-n-estimators", type=int, default=500, help="RF trees for MDA importance")

    # Cross-validation
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--embargo-days", type=int, default=2, help="Embargo period (days)")

    # Task type
    parser.add_argument(
        "--task-type",
        default="classification",
        choices=["classification", "regression"],
        help="Task type: classification (predict win/loss) or regression (predict P&L)"
    )

    # Optuna
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per window")
    parser.add_argument(
        "--optimization-metric",
        default="precision",
        choices=["auc", "f1", "precision", "r2", "neg_mae", "neg_rmse"],
        help="Metric to optimize (classification: auc/f1/precision, regression: r2/neg_mae/neg_rmse)"
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=0.60,
        help="Threshold for precision metric (default: 0.60 - only take trades with >60%% confidence)"
    )

    # Ensemble
    parser.add_argument("--use-ensemble", action="store_true", default=True, help="Use ensemble model")

    # Data preparation
    parser.add_argument("--lambda-decay", type=float, default=0.10, help="Exponential decay for sample weights")
    parser.add_argument("--min-samples-per-class", type=int, default=500, help="Min samples per class")
    parser.add_argument("--missing-value-threshold", type=float, default=0.25, help="Max allowed missing value fraction")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
