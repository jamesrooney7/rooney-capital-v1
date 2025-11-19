"""
Component 3: Purged K-Fold Cross-Validation

Implements Purged K-Fold CV with embargo to prevent information leakage
in time-series data with overlapping labels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List
from itertools import combinations

from .config_defaults import CV_DEFAULTS

logger = logging.getLogger(__name__)


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation with Embargo.

    Prevents information leakage by:
    1. Purging training samples whose labels overlap with test period
    2. Adding embargo period after each test fold
    3. Respecting temporal ordering
    """

    def __init__(
        self,
        n_splits: int = CV_DEFAULTS['n_splits'],
        embargo_days: int = CV_DEFAULTS['embargo_days'],
        k_test: int = 2
    ):
        """
        Initialize Purged K-Fold CV.

        Args:
            n_splits: Number of folds
            embargo_days: Embargo period in days (buffer after test period)
            k_test: Number of test folds per split (for combinatorial CV)
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.k_test = k_test

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.

        Args:
            X: Feature matrix (must have 'Date' column)
            y: Target labels (not used, for sklearn compatibility)
            groups: Group labels (not used)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if 'Date' not in X.columns:
            raise ValueError("X must have 'Date' column for temporal splitting")

        dates = pd.to_datetime(X['Date'])
        n_samples = len(X)

        # Create time-based folds
        fold_size = n_samples // self.n_splits
        fold_indices = []

        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            fold_indices.append(np.arange(start_idx, end_idx))

        # Generate k-fold test combinations
        for test_fold_combo in combinations(range(self.n_splits), self.k_test):
            # Get test indices
            test_idx = np.concatenate([fold_indices[i] for i in test_fold_combo])

            # Get test date range
            test_dates = dates.iloc[test_idx]
            test_start = test_dates.min()
            test_end = test_dates.max()

            # Get potential training indices (all non-test folds)
            train_idx = np.concatenate([
                fold_indices[i] for i in range(self.n_splits)
                if i not in test_fold_combo
            ])

            # Apply purging: remove training samples within embargo days of test period
            embargo_delta = pd.Timedelta(days=self.embargo_days)

            # Calculate embargo boundaries
            embargo_start = test_start - embargo_delta
            embargo_end = test_end + embargo_delta

            # Keep only training samples outside embargo window
            train_dates = dates.iloc[train_idx]
            valid_train_mask = (train_dates < embargo_start) | (train_dates > embargo_end)
            purged_train_idx = train_idx[valid_train_mask]

            logger.debug(
                f"Fold {test_fold_combo}: {len(purged_train_idx)} train samples "
                f"({len(train_idx) - len(purged_train_idx)} purged), "
                f"{len(test_idx)} test samples"
            )

            yield purged_train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Return the number of splits.

        Returns:
            Number of CV splits (combinations of k_test folds from n_splits)
        """
        from math import comb
        return comb(self.n_splits, self.k_test)


class PurgedKFoldValidator:
    """
    Validation utilities for Purged K-Fold CV.
    """

    @staticmethod
    def validate_no_leakage(
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        embargo_days: int
    ) -> Tuple[bool, str]:
        """
        Validate that there's no information leakage between train and test sets.

        Args:
            X: Feature matrix with 'Date' column
            train_idx: Training indices
            test_idx: Test indices
            embargo_days: Embargo period in days

        Returns:
            Tuple of (is_valid, message)
        """
        dates = pd.to_datetime(X['Date'])

        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]

        test_start = test_dates.min()
        test_end = test_dates.max()

        embargo_delta = pd.Timedelta(days=embargo_days)
        embargo_start = test_start - embargo_delta
        embargo_end = test_end + embargo_delta

        # Check for training samples in embargo window
        leakage_mask = (train_dates >= embargo_start) & (train_dates <= embargo_end)
        n_leakage = leakage_mask.sum()

        if n_leakage > 0:
            return False, f"Found {n_leakage} training samples in embargo window"

        return True, "No leakage detected"

    @staticmethod
    def get_split_summary(
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> dict:
        """
        Get summary statistics for a train/test split.

        Args:
            X: Feature matrix with 'Date' column
            train_idx: Training indices
            test_idx: Test indices

        Returns:
            Dictionary with split statistics
        """
        dates = pd.to_datetime(X['Date'])

        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]

        return {
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'train_date_range': {
                'start': str(train_dates.min().date()),
                'end': str(train_dates.max().date())
            },
            'test_date_range': {
                'start': str(test_dates.min().date()),
                'end': str(test_dates.max().date())
            },
            'train_test_gap_days': (test_dates.min() - train_dates.max()).days
        }


def evaluate_model_with_purged_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series = None,
    n_splits: int = 5,
    embargo_days: int = 60,
    scoring_fn=None
) -> Tuple[List[float], np.ndarray]:
    """
    Evaluate model using Purged K-Fold CV.

    Args:
        model: Sklearn-compatible model with fit() and predict_proba()
        X: Feature matrix (must have 'Date' column)
        y: Target labels
        sample_weight: Optional sample weights
        n_splits: Number of CV folds
        embargo_days: Embargo period in days
        scoring_fn: Function to score predictions, default is accuracy

    Returns:
        Tuple of (fold_scores, oof_predictions)
    """
    pkf = PurgedKFold(n_splits=n_splits, embargo_days=embargo_days, k_test=2)

    fold_scores = []
    oof_predictions = np.zeros(len(X))

    if scoring_fn is None:
        # Default: accuracy
        scoring_fn = lambda y_true, y_pred: (y_true == y_pred).mean()

    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X), 1):
        # Get fold data
        X_train = X.iloc[train_idx].drop(columns=['Date'])
        X_test = X.iloc[test_idx].drop(columns=['Date'])
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Get sample weights for this fold
        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx]
            model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)

        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        # Score
        score = scoring_fn(y_test, y_pred)
        fold_scores.append(score)

        # Store OOF predictions
        if hasattr(model, 'predict_proba'):
            oof_predictions[test_idx] = y_pred_proba
        else:
            oof_predictions[test_idx] = y_pred

        logger.debug(f"Fold {fold_idx}/{pkf.get_n_splits()}: score={score:.4f}")

    logger.info(f"Purged CV complete: mean score={np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    return fold_scores, oof_predictions
