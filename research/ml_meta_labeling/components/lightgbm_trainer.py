"""
Component 4: LightGBM Model Training

Handles LightGBM model training with class balancing and feature importance tracking.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import lightgbm as lgb

from .config_defaults import LIGHTGBM_SEARCH_SPACE

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """Train LightGBM models for meta-labeling."""

    def __init__(
        self,
        hyperparameters: Optional[Dict] = None,
        random_state: int = 42,
        task_type: str = 'classification'
    ):
        """
        Initialize LightGBM trainer.

        Args:
            hyperparameters: Dictionary of LightGBM hyperparameters
            random_state: Random seed
            task_type: 'classification' or 'regression'
        """
        self.random_state = random_state
        self.task_type = task_type
        self.hyperparameters = hyperparameters or self._get_default_hyperparameters()
        self.model = None  # Will be LGBMClassifier or LGBMRegressor
        self.feature_importance_: Optional[Dict[str, float]] = None

    def _get_default_hyperparameters(self) -> Dict:
        """Get default hyperparameters from search space defaults."""
        return {
            'num_leaves': LIGHTGBM_SEARCH_SPACE['num_leaves']['default'],
            'max_depth': LIGHTGBM_SEARCH_SPACE['max_depth']['default'],
            'n_estimators': LIGHTGBM_SEARCH_SPACE['n_estimators']['default'],
            'learning_rate': LIGHTGBM_SEARCH_SPACE['learning_rate']['default'],
            'feature_fraction': LIGHTGBM_SEARCH_SPACE['feature_fraction']['default'],
            'bagging_fraction': LIGHTGBM_SEARCH_SPACE['bagging_fraction']['default'],
            'min_data_in_leaf': LIGHTGBM_SEARCH_SPACE['min_data_in_leaf']['default'],
            'reg_alpha': LIGHTGBM_SEARCH_SPACE['reg_alpha']['default'],
            'reg_lambda': LIGHTGBM_SEARCH_SPACE['reg_lambda']['default']
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ):
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels (binary for classification, continuous for regression)
            sample_weight: Optional sample weights
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels for early stopping
            early_stopping_rounds: Rounds for early stopping (0 to disable)

        Returns:
            Trained LightGBM model
        """
        logger.info(f"Training LightGBM {self.task_type} model...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {X_train.shape[1]}")

        if self.task_type == 'classification':
            # Calculate class weights
            class_weights = self._calculate_class_weights(y_train)
            logger.info(f"Class weights: {class_weights}")

            # Build classification model
            model_params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': self.random_state,
                'class_weight': class_weights,
                **self.hyperparameters
            }

            self.model = lgb.LGBMClassifier(**model_params)
        else:  # regression
            logger.info(f"Target (P&L) stats: mean=${y_train.mean():.2f}, std=${y_train.std():.2f}")

            # Build regression model
            model_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'random_state': self.random_state,
                **self.hyperparameters
            }

            self.model = lgb.LGBMRegressor(**model_params)

        # Prepare callbacks
        callbacks = []
        if X_val is not None and y_val is not None and early_stopping_rounds > 0:
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        else:
            eval_set = None

        # Train
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Extract feature importance
        self._extract_feature_importance(X_train.columns)

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration_}")

        return self.model

    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """
        Calculate balanced class weights.

        Args:
            y: Target labels

        Returns:
            Dictionary of class weights
        """
        class_counts = y.value_counts()
        n_samples = len(y)

        weights = {}
        for class_label, count in class_counts.items():
            weights[class_label] = n_samples / (2 * count)

        return weights

    def _extract_feature_importance(self, feature_names):
        """Extract and store feature importance."""
        if self.model is None:
            return

        # Get importance (gain-based, more meaningful than split-based)
        importance_values = self.model.booster_.feature_importance(importance_type='gain')

        self.feature_importance_ = dict(zip(feature_names, importance_values))

        # Log top 10
        sorted_importance = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Top 10 most important features:")
        for i, (feat, imp) in enumerate(sorted_importance[:10], 1):
            logger.info(f"  {i}. {feat}: {imp:.1f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0 or 1) for classification, continuous values for regression
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions (classification only).

        Args:
            X: Feature matrix

        Returns:
            Probability predictions (0 to 1) for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification models")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        if self.feature_importance_ is None:
            raise ValueError("Model not trained yet")

        return self.feature_importance_

    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.booster_.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = lgb.Booster(model_file=filepath)
        logger.info(f"Model loaded from {filepath}")


def create_lightgbm_from_trial(trial, random_state: int = 42, task_type: str = 'classification') -> LightGBMTrainer:
    """
    Create LightGBM trainer from Optuna trial.

    Args:
        trial: Optuna trial object
        random_state: Random seed
        task_type: 'classification' or 'regression'

    Returns:
        LightGBMTrainer instance
    """
    # Sample hyperparameters from search space
    hyperparameters = {}

    for param_name, config in LIGHTGBM_SEARCH_SPACE.items():
        if config['type'] == 'int':
            hyperparameters[param_name] = trial.suggest_int(
                param_name,
                config['min'],
                config['max']
            )
        elif config['type'] == 'float':
            if config.get('log_scale', False):
                hyperparameters[param_name] = trial.suggest_float(
                    param_name,
                    config['min'],
                    config['max'],
                    log=True
                )
            else:
                hyperparameters[param_name] = trial.suggest_float(
                    param_name,
                    config['min'],
                    config['max']
                )

    return LightGBMTrainer(hyperparameters=hyperparameters, random_state=random_state, task_type=task_type)
