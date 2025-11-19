"""
Component 5: Optuna TPE Hyperparameter Optimization

Uses Tree-structured Parzen Estimator (TPE) for efficient Bayesian optimization
of LightGBM hyperparameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .lightgbm_trainer import create_lightgbm_from_trial
from .purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna TPE sampler."""

    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = -1,
        optimization_metric: str = 'auc',
        precision_threshold: float = 0.60,
        pruning_patience: int = 25,
        cv_folds: int = 5,
        embargo_days: int = 60,
        random_state: int = 42
    ):
        """
        Initialize Optuna optimizer.

        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all cores)
            optimization_metric: Metric to optimize ('auc', 'f1', 'precision')
            precision_threshold: Threshold for precision metric (default: 0.60)
            pruning_patience: Patience for pruning unpromising trials
            cv_folds: Number of CV folds
            embargo_days: Embargo period for Purged K-Fold
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.optimization_metric = optimization_metric
        self.precision_threshold = precision_threshold
        self.pruning_patience = pruning_patience
        self.cv_folds = cv_folds
        self.embargo_days = embargo_days
        self.random_state = random_state

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        self.best_value: Optional[float] = None

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        scoring_fn: Optional[Callable] = None
    ) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization.

        Args:
            X: Feature matrix (must have 'Date' column)
            y: Target labels
            sample_weight: Optional sample weights
            scoring_fn: Optional custom scoring function (y_true, y_pred) -> score

        Returns:
            Tuple of (best_hyperparameters, best_score)
        """
        logger.info(f"Starting Optuna optimization: {self.n_trials} trials")
        logger.info(f"Optimization metric: {self.optimization_metric}")

        # Create sampler and pruner
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(
            n_startup_trials=self.pruning_patience,
            n_warmup_steps=10
        )

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        # Create objective function
        def objective(trial):
            return self._objective(trial, X, y, sample_weight, scoring_fn)

        # Optimize
        if self.n_jobs == -1:
            # Use all cores - don't pass n_jobs parameter
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=False
            )
        else:
            # Use specific number of jobs
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=False
            )

        # Extract best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Optimization complete!")
        logger.info(f"Best {self.optimization_metric}: {self.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")

        return self.best_params, self.best_value

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
        scoring_fn: Optional[Callable]
    ) -> float:
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial
            X: Features
            y: Target
            sample_weight: Sample weights
            scoring_fn: Scoring function

        Returns:
            Score to maximize
        """
        # Create model from trial
        trainer = create_lightgbm_from_trial(trial, self.random_state)

        # Setup Purged K-Fold CV
        # Use k_test=1 (standard k-fold) for hyperparameter optimization
        # to reduce computational cost and avoid empty folds with small datasets
        pkf = PurgedKFold(
            n_splits=self.cv_folds,
            embargo_days=self.embargo_days,
            k_test=1
        )

        # Evaluate across folds
        fold_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X), 1):
            # Validate fold has sufficient training samples
            min_train_samples = 100
            if len(train_idx) < min_train_samples:
                logger.warning(
                    f"Fold {fold_idx} has only {len(train_idx)} training samples "
                    f"(minimum: {min_train_samples}). Skipping this fold."
                )
                continue

            # Validate fold has test samples
            if len(test_idx) == 0:
                logger.warning(f"Fold {fold_idx} has no test samples. Skipping this fold.")
                continue

            # Get fold data
            X_train_fold = X.iloc[train_idx].drop(columns=['Date'])
            X_test_fold = X.iloc[test_idx].drop(columns=['Date'])
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]

            # Get weights
            sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

            # Train
            trainer.train(X_train_fold, y_train_fold, sw_train)

            # Predict
            y_pred_proba = trainer.predict_proba(X_test_fold)

            # Score
            if scoring_fn is not None:
                score = scoring_fn(y_test_fold, y_pred_proba)
            else:
                score = self._default_scoring(y_test_fold, y_pred_proba)

            fold_scores.append(score)

            # Report intermediate value for pruning
            trial.report(score, fold_idx)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Validate we have at least some fold scores
        if len(fold_scores) == 0:
            logger.warning(
                "All CV folds were skipped due to insufficient data. "
                "This can happen with small datasets and large embargo periods."
            )
            # Return a poor score to discourage this hyperparameter combination
            return 0.5  # Random baseline for binary classification

        # Return mean score across folds
        mean_score = np.mean(fold_scores)
        logger.debug(f"Trial CV score: {mean_score:.4f} (from {len(fold_scores)} folds)")
        return mean_score

    def _default_scoring(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """
        Default scoring function.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Score (metric depends on optimization_metric setting)
        """
        from sklearn.metrics import roc_auc_score, precision_score, f1_score

        if self.optimization_metric == 'auc':
            return roc_auc_score(y_true, y_pred_proba)
        elif self.optimization_metric == 'f1':
            y_pred = (y_pred_proba >= 0.5).astype(int)
            return f1_score(y_true, y_pred, zero_division=0)
        elif self.optimization_metric == 'precision':
            # Use configurable threshold (López de Prado approach)
            y_pred = (y_pred_proba >= self.precision_threshold).astype(int)
            return precision_score(y_true, y_pred, zero_division=0)
        else:
            # Default to AUC
            return roc_auc_score(y_true, y_pred_proba)

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns:
            DataFrame with trial number, value, and parameters
        """
        if self.study is None:
            raise ValueError("No optimization performed yet")

        trials = self.study.trials
        history = []

        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                record = {
                    'trial': trial.number,
                    'value': trial.value,
                    **trial.params
                }
                history.append(record)

        return pd.DataFrame(history)

    def get_hyperparameter_importances(self) -> Dict[str, float]:
        """
        Get hyperparameter importance scores.

        Returns:
            Dictionary of parameter importances
        """
        if self.study is None:
            raise ValueError("No optimization performed yet")

        importances = optuna.importance.get_param_importances(self.study)
        return importances

    def save_study(self, filepath: str):
        """Save Optuna study to file."""
        if self.study is None:
            raise ValueError("No optimization performed yet")

        import joblib
        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to {filepath}")

    def load_study(self, filepath: str):
        """Load Optuna study from file."""
        import joblib
        self.study = joblib.load(filepath)
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        logger.info(f"Study loaded from {filepath}")
