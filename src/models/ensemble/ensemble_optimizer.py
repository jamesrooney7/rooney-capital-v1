"""Ensemble Model Optimizer for Random Forest Trading Models.

This module implements ensemble model selection and weight optimization to improve
trading strategy performance. Key features:
- Diversity-filtered model selection (correlation < 0.7)
- Convex weight optimization (equal-weight, Sharpe-weighted, or GEM)
- Production-ready ensemble predictor

Expected improvement: 12-20% Sharpe increase over best single model.

Usage:
    from src.models.ensemble.ensemble_optimizer import EnsembleOptimizer

    # After hyperparameter optimization completes
    ensemble = EnsembleOptimizer(
        models=trained_models,
        X_val=X_validation,
        y_val=y_validation,
        returns_val=returns_validation,
        method="sharpe_weighted"
    )

    ensemble.fit()
    predictions = ensemble.predict(X_test)
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class EnsembleOptimizer:
    """Ensemble optimizer for Random Forest trading models.

    Selects diverse models (correlation < 0.7) and optimizes ensemble weights
    to maximize validation Sharpe ratio.

    Attributes:
        models: List of (model, params, metrics) tuples from optimization
        X_val: Validation feature matrix
        y_val: Validation binary labels
        returns_val: Validation PnL returns
        method: Weight optimization method ("equal", "sharpe_weighted", "gem")
        max_models: Maximum models in ensemble (default: 12)
        min_models: Minimum models in ensemble (default: 7)
        correlation_threshold: Maximum pairwise correlation (default: 0.7)
    """

    def __init__(
        self,
        models: List[Tuple[RandomForestClassifier, Dict[str, Any], Dict[str, float]]],
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        returns_val: np.ndarray,
        method: str = "sharpe_weighted",
        max_models: int = 12,
        min_models: int = 7,
        correlation_threshold: float = 0.7,
    ):
        """Initialize ensemble optimizer.

        Args:
            models: List of (model, params, metrics) from optimization trials
            X_val: Validation feature matrix (for diversity calculation)
            y_val: Validation binary labels (for metrics)
            returns_val: Validation PnL returns (for Sharpe calculation)
            method: "equal", "sharpe_weighted", or "gem" (default: "sharpe_weighted")
            max_models: Maximum models to include (default: 12)
            min_models: Minimum models required (default: 7)
            correlation_threshold: Max pairwise correlation (default: 0.7)
        """
        self.models = models
        self.X_val = X_val
        self.y_val = y_val
        self.returns_val = returns_val
        self.method = method
        self.max_models = max_models
        self.min_models = min_models
        self.correlation_threshold = correlation_threshold

        # Results (populated by fit())
        self.selected_models = []
        self.weights = None
        self.ensemble_sharpe = None
        self.predictions_val = None  # N x M array of model predictions

    def _get_model_predictions(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Get probability predictions from a single model."""
        return model.predict_proba(X)[:, 1]

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        return (mean_return / std_return) * np.sqrt(252)

    def _calculate_pairwise_correlation(
        self,
        pred1: np.ndarray,
        pred2: np.ndarray,
    ) -> float:
        """Calculate Pearson correlation between two prediction arrays."""
        return np.corrcoef(pred1, pred2)[0, 1]

    def select_diverse_models(self) -> List[int]:
        """Select diverse models using greedy diversity filtering.

        Algorithm:
        1. Rank all models by validation Sharpe ratio (descending)
        2. Select top model automatically
        3. Iteratively add next-best model if correlation with all existing < threshold
        4. Stop when max_models reached or no more diverse models available

        Returns:
            List of model indices (into self.models)
        """
        logger.info("\n" + "=" * 60)
        logger.info("Ensemble Model Selection - Diversity Filtering")
        logger.info("=" * 60)

        # Rank models by Sharpe ratio
        model_sharpes = []
        for i, (model, params, metrics) in enumerate(self.models):
            sharpe = metrics.get("oos_sharpe", metrics.get("mean_fold_sharpe", 0.0))
            model_sharpes.append((i, sharpe))

        model_sharpes.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Ranked {len(model_sharpes)} models by Sharpe ratio")
        logger.info(f"Top 5 Sharpe ratios: {[s for _, s in model_sharpes[:5]]}")

        # Get predictions for all models on validation set
        all_predictions = []
        for i, (model, _, _) in enumerate(self.models):
            preds = self._get_model_predictions(model, self.X_val)
            all_predictions.append(preds)

        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)

        # Greedy selection with diversity constraint
        selected_indices = []

        # Always select top model
        top_idx = model_sharpes[0][0]
        selected_indices.append(top_idx)
        logger.info(f"\n1. Selected model {top_idx} (Sharpe={model_sharpes[0][1]:.3f}) - Top performer")

        # Iteratively add diverse models
        for rank, (candidate_idx, candidate_sharpe) in enumerate(model_sharpes[1:], start=2):
            if len(selected_indices) >= self.max_models:
                logger.info(f"\nReached max_models={self.max_models}, stopping selection")
                break

            # Check correlation with all already-selected models
            candidate_preds = all_predictions[candidate_idx]
            max_correlation = 0.0

            for selected_idx in selected_indices:
                selected_preds = all_predictions[selected_idx]
                corr = self._calculate_pairwise_correlation(candidate_preds, selected_preds)
                max_correlation = max(max_correlation, abs(corr))

            # Add if sufficiently diverse
            if max_correlation < self.correlation_threshold:
                selected_indices.append(candidate_idx)
                logger.info(
                    f"{len(selected_indices)}. Selected model {candidate_idx} "
                    f"(Sharpe={candidate_sharpe:.3f}, max_corr={max_correlation:.3f})"
                )
            else:
                logger.debug(
                    f"   Rejected model {candidate_idx} "
                    f"(Sharpe={candidate_sharpe:.3f}, max_corr={max_correlation:.3f} >= {self.correlation_threshold})"
                )

        # Validate minimum models
        if len(selected_indices) < self.min_models:
            logger.warning(
                f"Only found {len(selected_indices)} diverse models (min={self.min_models}). "
                f"Consider relaxing correlation_threshold={self.correlation_threshold}"
            )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Selected {len(selected_indices)} diverse models")
        logger.info(f"{'=' * 60}\n")

        return selected_indices

    def optimize_weights(self, selected_indices: List[int]) -> np.ndarray:
        """Optimize ensemble weights for selected models.

        Methods:
        - "equal": Equal weights (1/N)
        - "sharpe_weighted": Weight proportional to individual Sharpe ratios
        - "gem": Convex optimization to maximize ensemble Sharpe (experimental)

        Args:
            selected_indices: Indices of selected models

        Returns:
            Array of weights (length = len(selected_indices))
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Weight Optimization - Method: {self.method}")
        logger.info("=" * 60)

        n_models = len(selected_indices)

        # Get predictions for selected models
        selected_predictions = []
        selected_sharpes = []

        for idx in selected_indices:
            model, params, metrics = self.models[idx]
            preds = self._get_model_predictions(model, self.X_val)
            selected_predictions.append(preds)

            sharpe = metrics.get("oos_sharpe", metrics.get("mean_fold_sharpe", 0.0))
            selected_sharpes.append(sharpe)

        selected_predictions = np.array(selected_predictions)  # Shape: (n_models, n_samples)

        if self.method == "equal":
            # Equal weights
            weights = np.ones(n_models) / n_models
            logger.info(f"Using equal weights: {weights}")

        elif self.method == "sharpe_weighted":
            # Weight proportional to Sharpe ratios
            sharpe_array = np.array(selected_sharpes)
            sharpe_array = np.maximum(sharpe_array, 0.01)  # Floor at 0.01 to avoid division issues
            weights = sharpe_array / sharpe_array.sum()
            logger.info(f"Individual Sharpe ratios: {sharpe_array}")
            logger.info(f"Sharpe-weighted weights: {weights}")

        elif self.method == "gem":
            # GEM (Generalized Ensemble Method) - convex optimization
            logger.info("Running convex optimization to maximize ensemble Sharpe...")

            def negative_sharpe(w):
                """Objective: negative Sharpe ratio (for minimization)."""
                ensemble_probs = np.dot(w, selected_predictions)
                ensemble_returns = self.returns_val * (ensemble_probs >= 0.5)
                return -self._calculate_sharpe(ensemble_returns)

            # Constraints: weights >= 0, sum(weights) = 1
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0) for _ in range(n_models)]

            # Initial guess: equal weights
            w0 = np.ones(n_models) / n_models

            result = minimize(
                negative_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                weights = result.x
                logger.info(f"Optimization converged. Optimal weights: {weights}")
            else:
                logger.warning(f"Optimization failed: {result.message}")
                logger.warning("Falling back to equal weights")
                weights = np.ones(n_models) / n_models

        else:
            raise ValueError(f"Unknown weight method: {self.method}")

        # Normalize to ensure sum = 1 (numerical stability)
        weights = weights / weights.sum()

        logger.info(f"\nFinal weights (normalized): {weights}")
        logger.info(f"Weight sum: {weights.sum():.6f}")
        logger.info(f"{'=' * 60}\n")

        return weights

    def fit(self) -> "EnsembleOptimizer":
        """Build ensemble: select models + optimize weights.

        Returns:
            self (for chaining)
        """
        logger.info("\n" + "=" * 70)
        logger.info("ENSEMBLE OPTIMIZER - Building Optimized Model Ensemble")
        logger.info("=" * 70)

        # Step 1: Select diverse models
        selected_indices = self.select_diverse_models()
        self.selected_models = [self.models[i] for i in selected_indices]

        # Step 2: Optimize weights
        self.weights = self.optimize_weights(selected_indices)

        # Step 3: Calculate ensemble validation performance
        logger.info("\n" + "=" * 60)
        logger.info("Ensemble Validation Performance")
        logger.info("=" * 60)

        # Get predictions for all selected models
        ensemble_predictions = []
        for model, _, _ in self.selected_models:
            preds = self._get_model_predictions(model, self.X_val)
            ensemble_predictions.append(preds)

        ensemble_predictions = np.array(ensemble_predictions)  # Shape: (n_models, n_samples)
        self.predictions_val = ensemble_predictions

        # Weighted ensemble prediction
        ensemble_probs = np.dot(self.weights, ensemble_predictions)
        ensemble_binary = (ensemble_probs >= 0.5).astype(int)

        # Calculate metrics
        ensemble_returns = self.returns_val[ensemble_binary == 1]
        self.ensemble_sharpe = self._calculate_sharpe(ensemble_returns)

        win_rate = (self.y_val[ensemble_binary == 1] == 1).mean()
        n_trades = ensemble_binary.sum()

        logger.info(f"Ensemble models: {len(self.selected_models)}")
        logger.info(f"Ensemble Sharpe: {self.ensemble_sharpe:.3f}")
        logger.info(f"Ensemble trades: {n_trades}")
        logger.info(f"Ensemble win rate: {win_rate*100:.1f}%")

        # Compare with best single model
        best_single_sharpe = max(
            m[2].get("oos_sharpe", m[2].get("mean_fold_sharpe", 0.0))
            for m in self.models
        )
        improvement = ((self.ensemble_sharpe / best_single_sharpe) - 1.0) * 100

        logger.info(f"\nBest single model Sharpe: {best_single_sharpe:.3f}")
        logger.info(f"Ensemble improvement: {improvement:+.1f}%")
        logger.info(f"{'=' * 60}\n")

        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Generate ensemble predictions for new data.

        Args:
            X: Feature matrix
            threshold: Probability threshold (default: 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        if self.weights is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all selected models
        predictions = []
        for model, _, _ in self.selected_models:
            preds = self._get_model_predictions(model, X)
            predictions.append(preds)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        # Weighted ensemble
        ensemble_probs = np.dot(self.weights, predictions)
        ensemble_binary = (ensemble_probs >= threshold).astype(int)

        return ensemble_binary

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions (continuous 0-1)
        """
        if self.weights is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all selected models
        predictions = []
        for model, _, _ in self.selected_models:
            preds = self._get_model_predictions(model, X)
            predictions.append(preds)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        # Weighted ensemble
        ensemble_probs = np.dot(self.weights, predictions)

        return ensemble_probs

    def get_metadata(self) -> Dict[str, Any]:
        """Get ensemble metadata for saving.

        Returns:
            Dictionary with ensemble configuration and performance
        """
        if self.weights is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        return {
            "method": self.method,
            "n_models": len(self.selected_models),
            "weights": self.weights.tolist(),
            "ensemble_sharpe": float(self.ensemble_sharpe),
            "correlation_threshold": self.correlation_threshold,
            "model_params": [params for _, params, _ in self.selected_models],
            "model_metrics": [
                {
                    "sharpe": m[2].get("oos_sharpe", m[2].get("mean_fold_sharpe", 0.0)),
                    "pf": m[2].get("oos_pf", m[2].get("mean_fold_pf", 0.0)),
                }
                for m in self.selected_models
            ],
        }


def build_ensemble_from_trials(
    trial_results: List[Dict[str, Any]],
    trained_models: List[RandomForestClassifier],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    returns_val: np.ndarray,
    method: str = "sharpe_weighted",
    max_models: int = 12,
    min_models: int = 7,
) -> EnsembleOptimizer:
    """Convenience function to build ensemble from hyperparameter optimization results.

    Args:
        trial_results: List of trial dictionaries with "params" and "metrics"
        trained_models: List of trained RandomForestClassifier models
        X_val: Validation features
        y_val: Validation labels
        returns_val: Validation returns
        method: Weight optimization method
        max_models: Maximum ensemble size
        min_models: Minimum ensemble size

    Returns:
        Fitted EnsembleOptimizer
    """
    # Combine into (model, params, metrics) tuples
    models = [
        (trained_models[i], trial_results[i]["params"], trial_results[i]["metrics"])
        for i in range(len(trained_models))
    ]

    # Build and fit ensemble
    ensemble = EnsembleOptimizer(
        models=models,
        X_val=X_val,
        y_val=y_val,
        returns_val=returns_val,
        method=method,
        max_models=max_models,
        min_models=min_models,
    )

    ensemble.fit()

    return ensemble
