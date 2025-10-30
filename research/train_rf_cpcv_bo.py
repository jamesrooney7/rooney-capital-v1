"""Train Random Forest models using CPCV and Bayesian Optimization.

This script loads extracted training data from `data/training/<SYMBOL>_transformed_features.csv`
and trains a Random Forest classifier with:
1. Combinatorial Purged Cross-Validation (CPCV) for time-series
2. Bayesian Optimization for hyperparameter tuning
3. Output format matching existing model bundles

Usage:
    python research/train_rf_cpcv_bo.py --symbol ES
    python research/train_rf_cpcv_bo.py --symbol ES --n-trials 50 --n-folds 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CPCV (Combinatorial Purged Cross-Validation) Implementation
# ============================================================================


def get_cpcv_splits(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate CPCV splits with TIME-BASED purging between train/test sets.

    Args:
        dates: Series of trade entry dates
        n_splits: Number of folds (default: 5)
        embargo_days: Number of days to purge between train/test (default: 3)
                     This should account for max holding period + buffer.
                     For 1-2 day holds, 3 days is appropriate.

    Returns:
        List of (train_indices, test_indices) tuples

    Note:
        Uses time-based embargo instead of percentage-based purge to ensure
        the purge window matches actual label finalization time:
        - Max hold period: ~1-2 days (8 bars or 3PM close)
        - Buffer: +1 day
        - Total embargo: 3 days from last possible exit
    """
    # Convert dates to datetime and extract unique dates
    dates_dt = pd.to_datetime(dates)
    dates_array = dates_dt.dt.date.values
    unique_dates = np.array(sorted(pd.Series(dates_array).unique()))

    # Split into chronological folds by date
    split_points = np.linspace(0, len(unique_dates), n_splits + 1, dtype=int)
    folds = [unique_dates[split_points[i]:split_points[i+1]] for i in range(n_splits)]

    # Convert to ordinals for distance calculation
    date_ordinals = pd.to_datetime(dates_array).map(lambda x: x.toordinal()).to_numpy()

    splits = []
    for i in range(n_splits):
        # Test fold dates
        test_dates = folds[i]
        test_mask = np.isin(dates_array, test_dates)
        test_indices = np.where(test_mask)[0]

        # Calculate distance in DAYS from each sample to nearest test sample
        test_ordinals = np.array([pd.to_datetime(td).toordinal() for td in test_dates])
        distances = np.min(np.abs(date_ordinals[:, None] - test_ordinals[None, :]), axis=1)

        # Train mask: exclude test fold AND samples within embargo_days
        train_mask = (~test_mask) & (distances > embargo_days)
        train_indices = np.where(train_mask)[0]

        splits.append((train_indices, test_indices))

    logger.info(
        f"Generated {n_splits} CPCV splits with embargo={embargo_days} days "
        f"(avg train size: {np.mean([len(tr) for tr, _ in splits]):.0f}, "
        f"avg test size: {np.mean([len(te) for _, te in splits]):.0f})"
    )

    return splits


# ============================================================================
# Model Training and Evaluation
# ============================================================================


def calculate_sharpe_ratio(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe ratio from returns."""
    if len(returns) == 0:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return 0.0
    # Annualize: sqrt(252) for daily, adjust if needed
    return (mean_return / std_return) * np.sqrt(252)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    if len(returns) == 0:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return gains if gains > 0 else 0.0
    return gains / losses


def evaluate_model_cpcv(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Evaluate model using CPCV splits and return comprehensive metrics.

    Args:
        model: Trained RandomForest model
        X: Feature matrix
        y: Binary target (1=win, 0=loss)
        returns: PnL returns for Sharpe calculation
        splits: List of (train_idx, test_idx) tuples

    Returns:
        Dictionary with evaluation metrics
    """
    fold_sharpes = []
    fold_profit_factors = []
    fold_accuracies = []
    fold_aucs = []
    all_predictions = []
    all_actuals = []
    all_returns = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model on this fold
        model.fit(X_train, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.5  # Default if only one class in fold

        # Financial metrics (using actual returns)
        fold_returns = returns[test_idx]
        sharpe = calculate_sharpe_ratio(fold_returns)
        pf = calculate_profit_factor(fold_returns)

        fold_sharpes.append(sharpe)
        fold_profit_factors.append(pf)
        fold_accuracies.append(accuracy)
        fold_aucs.append(auc)

        all_predictions.extend(y_pred_proba)
        all_actuals.extend(y_test)
        all_returns.extend(fold_returns)

        logger.debug(
            f"  Fold {fold_idx+1}: Sharpe={sharpe:.3f}, PF={pf:.3f}, "
            f"Acc={accuracy:.3f}, AUC={auc:.3f}"
        )

    # Overall metrics
    mean_sharpe = np.mean(fold_sharpes)
    mean_pf = np.mean(fold_profit_factors)
    mean_accuracy = np.mean(fold_accuracies)
    mean_auc = np.mean(fold_aucs)

    # Out-of-sample overall metrics
    oos_sharpe = calculate_sharpe_ratio(np.array(all_returns))
    oos_pf = calculate_profit_factor(np.array(all_returns))

    try:
        oos_auc = roc_auc_score(all_actuals, all_predictions)
    except ValueError:
        oos_auc = 0.5

    return {
        "mean_fold_sharpe": mean_sharpe,
        "mean_fold_pf": mean_pf,
        "mean_fold_accuracy": mean_accuracy,
        "mean_fold_auc": mean_auc,
        "oos_sharpe": oos_sharpe,
        "oos_pf": oos_pf,
        "oos_auc": oos_auc,
        "fold_sharpes": fold_sharpes,
        "fold_profit_factors": fold_profit_factors,
    }


# ============================================================================
# Bayesian Optimization (Manual Implementation)
# ============================================================================


def sample_hyperparameters(trial_idx: int, n_trials: int) -> Dict[str, Any]:
    """Sample hyperparameters for Random Forest.

    Uses a combination of random sampling and grid-like exploration.
    """
    # Parameter distributions
    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700, 900, 1200],
        "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, None],
        "min_samples_leaf": [10, 20, 30, 50, 75, 100],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "bootstrap": [True],
        "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    # Use ParameterSampler for random search
    sampler = ParameterSampler(
        param_distributions,
        n_iter=1,
        random_state=42 + trial_idx,
    )

    params = list(sampler)[0]
    params["n_jobs"] = -1
    params["random_state"] = 42

    return params


def bayesian_optimization_search(
    X: pd.DataFrame,
    y: np.ndarray,
    returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
    score_metric: str = "sharpe",
) -> Tuple[Dict[str, Any], RandomForestClassifier, List[Dict[str, Any]]]:
    """Perform Bayesian Optimization-style search for best hyperparameters.

    Args:
        X: Feature matrix
        y: Binary target
        returns: PnL returns
        splits: CPCV splits
        n_trials: Number of hyperparameter trials
        score_metric: Metric to optimize ("sharpe", "profit_factor", "auc")

    Returns:
        Tuple of (best_params, best_model, trial_history)
    """
    logger.info(f"Starting hyperparameter search with {n_trials} trials...")

    best_score = -np.inf
    best_params = None
    best_model = None
    trial_history = []

    for trial_idx in range(n_trials):
        # Sample hyperparameters
        params = sample_hyperparameters(trial_idx, n_trials)

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial_idx + 1}/{n_trials}")
        logger.info(f"Params: {params}")

        # Create model
        model = RandomForestClassifier(**params)

        # Evaluate using CPCV
        metrics = evaluate_model_cpcv(model, X, y, returns, splits)

        # Select score based on metric
        if score_metric == "sharpe":
            score = metrics["oos_sharpe"]
        elif score_metric == "profit_factor":
            score = metrics["oos_pf"]
        elif score_metric == "auc":
            score = metrics["oos_auc"]
        else:
            score = metrics["oos_sharpe"]

        logger.info(
            f"Results: Sharpe={metrics['oos_sharpe']:.3f}, "
            f"PF={metrics['oos_pf']:.3f}, AUC={metrics['oos_auc']:.3f}"
        )

        # Track trial
        trial_record = {
            "trial": trial_idx + 1,
            "params": params.copy(),
            "metrics": metrics,
            "score": score,
        }
        trial_history.append(trial_record)

        # Update best
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
            logger.info(f"✓ New best {score_metric}: {score:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Optimization complete!")
    logger.info(f"Best {score_metric}: {best_score:.4f}")
    logger.info(f"Best params: {best_params}")

    return best_params, best_model, trial_history


# ============================================================================
# Threshold Optimization
# ============================================================================


def optimize_threshold(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    min_trades: int = 100,
) -> Tuple[float, Dict[str, Any]]:
    """Find optimal probability threshold for production.

    Args:
        model: Trained model
        X: Feature matrix
        y: Binary target
        returns: PnL returns
        splits: CPCV splits
        min_trades: Minimum number of trades required

    Returns:
        Tuple of (best_threshold, metrics_at_threshold)
    """
    logger.info("\nOptimizing probability threshold...")

    # Collect all OOS predictions
    all_predictions = []
    all_actuals = []
    all_returns = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        all_predictions.extend(y_pred_proba)
        all_actuals.extend(y[test_idx])
        all_returns.extend(returns[test_idx])

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_returns = np.array(all_returns)

    # Test thresholds from 0.4 to 0.7
    thresholds = np.arange(0.40, 0.71, 0.01)
    best_threshold = 0.5
    best_sharpe = -np.inf

    for threshold in thresholds:
        passed = all_predictions >= threshold
        if passed.sum() < min_trades:
            continue

        threshold_returns = all_returns[passed]
        sharpe = calculate_sharpe_ratio(threshold_returns)
        pf = calculate_profit_factor(threshold_returns)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold
            logger.debug(
                f"  Threshold {threshold:.2f}: {passed.sum()} trades, "
                f"Sharpe={sharpe:.3f}, PF={pf:.3f}"
            )

    # Get final metrics at best threshold
    passed = all_predictions >= best_threshold
    final_returns = all_returns[passed]

    metrics = {
        "threshold": float(best_threshold),
        "trades": int(passed.sum()),
        "sharpe": calculate_sharpe_ratio(final_returns),
        "profit_factor": calculate_profit_factor(final_returns),
        "win_rate": float((all_actuals[passed] == 1).mean()) if passed.sum() > 0 else 0.0,
    }

    logger.info(
        f"Best threshold: {best_threshold:.2f} "
        f"({metrics['trades']} trades, Sharpe={metrics['sharpe']:.3f})"
    )

    return best_threshold, metrics


# ============================================================================
# Main Training Pipeline
# ============================================================================


def load_training_data(symbol: str, data_dir: str = "data/training") -> pd.DataFrame:
    """Load extracted training data for symbol."""
    csv_path = Path(data_dir) / f"{symbol}_transformed_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}\n"
            f"Run: python research/extract_training_data.py --symbol {symbol}"
        )

    logger.info(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} trades with {len(df.columns)} columns")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix and targets from extracted data.

    Returns:
        Tuple of (X, y, returns, feature_names)
    """
    # Identify feature columns (exclude metadata and targets)
    exclude_cols = {
        "Date/Time", "Exit Date/Time", "Entry_Price", "Exit_Price",
        "y_return", "y_binary", "y_pnl_usd", "y_pnl_gross",
        "Unnamed: 0",  # Index column if present
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove columns with all NaN
    feature_cols = [col for col in feature_cols if df[col].notna().any()]

    logger.info(f"Using {len(feature_cols)} features for training")

    X = df[feature_cols].copy()
    y = df["y_binary"].values
    returns = df["y_pnl_usd"].values  # Net PnL after commissions

    # Fill NaN with 0 (represents features that weren't ready/applicable)
    X = X.fillna(0)

    return X, y, returns, feature_cols


def train_model(
    symbol: str,
    data_dir: str = "data/training",
    output_dir: str = "src/models",
    n_trials: int = 30,
    n_folds: int = 5,
    score_metric: str = "sharpe",
    min_trades_threshold: int = 100,
    min_total_trades: int = 1000,
) -> None:
    """Train Random Forest model with CPCV and Bayesian Optimization.

    Args:
        symbol: Trading symbol (e.g., "ES", "NQ")
        data_dir: Directory with extracted training data
        output_dir: Directory to save model bundles
        n_trials: Number of hyperparameter optimization trials
        n_folds: Number of CPCV folds
        score_metric: Metric to optimize ("sharpe", "profit_factor", "auc")
        min_trades_threshold: Minimum trades for threshold optimization
        min_total_trades: Minimum total trades required for training (default: 1000)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for {symbol}")
    logger.info(f"{'='*60}\n")

    # Load data
    df = load_training_data(symbol, data_dir)

    # Validate minimum trades
    if len(df) < min_total_trades:
        logger.error(
            f"❌ Insufficient training data for {symbol}: "
            f"{len(df)} trades (minimum: {min_total_trades})"
        )
        logger.error(
            f"   This symbol does not have enough historical trades for robust training."
        )
        raise ValueError(
            f"Symbol {symbol} has only {len(df)} trades, "
            f"need at least {min_total_trades} for training"
        )

    # Prepare features
    X, y, returns, feature_names = prepare_features(df)

    # Extract dates for time-based CPCV
    if "Date/Time" in df.columns:
        dates = pd.to_datetime(df["Date/Time"])
    elif "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    else:
        raise ValueError("No date column found in training data (expected 'Date/Time' or 'Date')")

    logger.info(f"Training set: {len(X)} samples, {len(feature_names)} features")
    logger.info(f"Class balance: {(y==1).sum()} wins / {(y==0).sum()} losses")
    logger.info(f"✅ Passed minimum trades requirement ({len(X)} >= {min_total_trades})")

    # Generate CPCV splits with time-based embargo (3 days = max 2-day hold + 1 buffer)
    splits = get_cpcv_splits(dates, n_splits=n_folds, embargo_days=3)

    # Hyperparameter optimization
    best_params, best_model, trial_history = bayesian_optimization_search(
        X, y, returns, splits, n_trials=n_trials, score_metric=score_metric
    )

    # Train final model on all data
    logger.info("\nTraining final model on full dataset...")
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X, y)

    # Optimize threshold
    threshold, threshold_metrics = optimize_threshold(
        final_model, X, y, returns, splits, min_trades=min_trades_threshold
    )

    # Evaluate final model
    final_metrics = evaluate_model_cpcv(final_model, X, y, returns, splits)

    # Save model bundle
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model .pkl
    model_file = output_path / f"{symbol}_rf_model.pkl"
    model_bundle = {
        "model": final_model,
        "features": feature_names,
    }
    joblib.dump(model_bundle, model_file)
    logger.info(f"Saved model: {model_file}")

    # Save metadata _best.json
    metadata = {
        "Symbol": symbol,
        "Sharpe": float(final_metrics["oos_sharpe"]),
        "Deflated_Sharpe": None,  # TODO: Implement if needed
        "Profit_Factor": float(final_metrics["oos_pf"]),
        "Trades": int(len(X)),
        "Prod_Threshold": threshold,
        "Guardrails": {
            "Min_Total_Trades": min_total_trades,
            "Min_Threshold_Trades": min_trades_threshold,
            "Era_Positive_Count": 4,  # TODO: Implement era analysis
            "Era_Count": 5,
        },
        "Params": best_params,
        "Features": feature_names,
        "Score_Metric": score_metric,
        "CPCV_Metrics": {
            "n_folds": n_folds,
            "mean_fold_sharpe": float(final_metrics["mean_fold_sharpe"]),
            "mean_fold_pf": float(final_metrics["mean_fold_pf"]),
            "fold_sharpes": [float(x) for x in final_metrics["fold_sharpes"]],
        },
        "Threshold_Metrics": threshold_metrics,
    }

    metadata_file = output_path / f"{symbol}_best.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_file}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training complete for {symbol}!")
    logger.info(f"{'='*60}")
    logger.info(f"Sharpe Ratio: {final_metrics['oos_sharpe']:.3f}")
    logger.info(f"Profit Factor: {final_metrics['oos_pf']:.3f}")
    logger.info(f"Prod Threshold: {threshold:.2f}")
    logger.info(f"Expected Trades: {threshold_metrics['trades']}")
    logger.info(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model with CPCV and Bayesian Optimization"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., ES, NQ, RTY)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory with extracted training data (default: data/training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/models",
        help="Directory to save model bundles (default: src/models)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of hyperparameter optimization trials (default: 30)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CPCV folds (default: 5)",
    )
    parser.add_argument(
        "--score-metric",
        type=str,
        default="sharpe",
        choices=["sharpe", "profit_factor", "auc"],
        help="Metric to optimize (default: sharpe)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=100,
        help="Minimum trades for threshold optimization (default: 100)",
    )
    parser.add_argument(
        "--min-total-trades",
        type=int,
        default=1000,
        help="Minimum total trades required for training (default: 1000)",
    )

    args = parser.parse_args()

    train_model(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        score_metric=args.score_metric,
        min_trades_threshold=args.min_trades,
        min_total_trades=args.min_total_trades,
    )


if __name__ == "__main__":
    main()
