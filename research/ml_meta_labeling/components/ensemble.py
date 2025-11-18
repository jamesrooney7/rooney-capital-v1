"""
Component 7: Ensemble Model Construction

Combines LightGBM, CatBoost, and XGBoost with a logistic regression meta-learner.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
import joblib

from .lightgbm_trainer import LightGBMTrainer
from .purged_kfold import PurgedKFold
from .config_defaults import CATBOOST_DEFAULTS, XGBOOST_DEFAULTS, ENSEMBLE_DEFAULTS

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of LightGBM, CatBoost, and XGBoost with meta-learner."""

    def __init__(
        self,
        lightgbm_params: Dict,
        random_state: int = 42
    ):
        """
        Initialize ensemble model.

        Args:
            lightgbm_params: Optimized hyperparameters for LightGBM
            random_state: Random seed
        """
        self.lightgbm_params = lightgbm_params
        self.random_state = random_state

        # Base models
        self.lightgbm_model: Optional[LightGBMTrainer] = None
        self.catboost_model = None
        self.xgboost_model = None

        # Meta-learner
        self.meta_learner: Optional[LogisticRegression] = None

        # Weights
        self.model_weights: Optional[Dict[str, float]] = None

    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        cv_folds: int = 5,
        embargo_days: int = 60
    ):
        """
        Train ensemble with out-of-fold predictions for meta-learner.

        Args:
            X: Feature matrix (with 'Date' column for CV)
            y: Target labels
            sample_weight: Optional sample weights
            cv_folds: Number of CV folds for meta-learner training
            embargo_days: Embargo period for Purged K-Fold
        """
        logger.info("Training ensemble model...")

        # Setup Purged K-Fold for out-of-fold predictions
        # Use k_test=1 (standard k-fold) to avoid empty folds with small datasets
        pkf = PurgedKFold(n_splits=cv_folds, embargo_days=embargo_days, k_test=1)

        # Store out-of-fold predictions from each model
        oof_predictions = {
            'lightgbm': np.zeros(len(X)),
            'catboost': np.zeros(len(X)),
            'xgboost': np.zeros(len(X))
        }

        # Train base models and generate OOF predictions
        for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X), 1):
            # Validate fold has sufficient data
            min_train_samples = 100
            if len(train_idx) < min_train_samples:
                logger.warning(
                    f"Fold {fold_idx} has only {len(train_idx)} training samples. Skipping."
                )
                continue

            if len(test_idx) == 0:
                logger.warning(f"Fold {fold_idx} has no test samples. Skipping.")
                continue

            logger.info(f"Training base models on fold {fold_idx}/{pkf.get_n_splits()}...")

            X_train_fold = X.iloc[train_idx].drop(columns=['Date'])
            X_test_fold = X.iloc[test_idx].drop(columns=['Date'])
            y_train_fold = y.iloc[train_idx]

            sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

            # LightGBM
            lgbm = LightGBMTrainer(self.lightgbm_params, self.random_state)
            lgbm.train(X_train_fold, y_train_fold, sw_train)
            oof_predictions['lightgbm'][test_idx] = lgbm.predict_proba(X_test_fold)

            # CatBoost
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(**CATBOOST_DEFAULTS, random_state=self.random_state)
            cb.fit(X_train_fold, y_train_fold, sample_weight=sw_train)
            oof_predictions['catboost'][test_idx] = cb.predict_proba(X_test_fold)[:, 1]

            # XGBoost
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(**XGBOOST_DEFAULTS, random_state=self.random_state)
            xgb_model.fit(X_train_fold, y_train_fold, sample_weight=sw_train)
            oof_predictions['xgboost'][test_idx] = xgb_model.predict_proba(X_test_fold)[:, 1]

        # Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner...")

        X_meta = np.column_stack([
            oof_predictions['lightgbm'],
            oof_predictions['catboost'],
            oof_predictions['xgboost']
        ])

        self.meta_learner = LogisticRegression(
            C=ENSEMBLE_DEFAULTS['meta_learner_C'],
            class_weight=ENSEMBLE_DEFAULTS['class_weight'],
            random_state=self.random_state
        )

        self.meta_learner.fit(X_meta, y)

        # Extract model weights from meta-learner coefficients
        self._extract_model_weights()

        # Retrain base models on full dataset
        logger.info("Retraining base models on full dataset...")

        X_full = X.drop(columns=['Date'])

        # LightGBM
        self.lightgbm_model = LightGBMTrainer(self.lightgbm_params, self.random_state)
        self.lightgbm_model.train(X_full, y, sample_weight)

        # CatBoost
        from catboost import CatBoostClassifier
        self.catboost_model = CatBoostClassifier(**CATBOOST_DEFAULTS, random_state=self.random_state)
        self.catboost_model.fit(X_full, y, sample_weight=sample_weight)

        # XGBoost
        import xgboost as xgb
        self.xgboost_model = xgb.XGBClassifier(**XGBOOST_DEFAULTS, random_state=self.random_state)
        self.xgboost_model.fit(X_full, y, sample_weight=sample_weight)

        logger.info("Ensemble training complete!")

    def _extract_model_weights(self):
        """Extract and normalize model weights from meta-learner."""
        coefficients = self.meta_learner.coef_[0]

        # Normalize to sum to 1
        weights_sum = np.abs(coefficients).sum()
        normalized_weights = np.abs(coefficients) / weights_sum

        self.model_weights = {
            'lightgbm': normalized_weights[0],
            'catboost': normalized_weights[1],
            'xgboost': normalized_weights[2]
        }

        logger.info("Model weights:")
        for model, weight in self.model_weights.items():
            logger.info(f"  {model}: {weight:.3f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble probability predictions.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        # Get predictions from each base model
        lgbm_pred = self.lightgbm_model.predict_proba(X)
        cb_pred = self.catboost_model.predict_proba(X)[:, 1]
        xgb_pred = self.xgboost_model.predict_proba(X)[:, 1]

        # Stack predictions
        X_meta = np.column_stack([lgbm_pred, cb_pred, xgb_pred])

        # Meta-learner prediction
        ensemble_pred = self.meta_learner.predict_proba(X_meta)[:, 1]

        return ensemble_pred

    def get_model_weights(self) -> Dict[str, float]:
        """Get normalized model weights."""
        return self.model_weights

    def save_ensemble(self, filepath: str):
        """Save ensemble to disk."""
        ensemble_data = {
            'lightgbm_model': self.lightgbm_model,
            'catboost_model': self.catboost_model,
            'xgboost_model': self.xgboost_model,
            'meta_learner': self.meta_learner,
            'model_weights': self.model_weights
        }

        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str):
        """Load ensemble from disk."""
        ensemble_data = joblib.load(filepath)

        self.lightgbm_model = ensemble_data['lightgbm_model']
        self.catboost_model = ensemble_data['catboost_model']
        self.xgboost_model = ensemble_data['xgboost_model']
        self.meta_learner = ensemble_data['meta_learner']
        self.model_weights = ensemble_data['model_weights']

        logger.info(f"Ensemble loaded from {filepath}")
