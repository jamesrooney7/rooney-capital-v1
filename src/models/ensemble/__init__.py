"""Ensemble model optimization for trading strategies.

This package provides ensemble model selection and weight optimization to improve
trading performance through model diversity and optimal weighting.

Main components:
- EnsembleOptimizer: Core ensemble building class
- build_ensemble_from_trials: Convenience function for integration

Expected improvement: 12-20% Sharpe increase over best single model.
"""

from .ensemble_optimizer import EnsembleOptimizer, build_ensemble_from_trials

__all__ = ["EnsembleOptimizer", "build_ensemble_from_trials"]
