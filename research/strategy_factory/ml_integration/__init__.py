"""
ML Integration Layer for Strategy Factory.

Provides tools to extract training data from vectorized strategies
for use with the ML pipeline (ml_meta_labeling_optimizer.py).
"""

from .feature_extractor import (
    FeatureExtractor,
    calculate_additional_features,
    calculate_all_features,
    calculate_cross_asset_features,
    CROSS_ASSET_SYMBOLS
)
from .extract_training_data import extract_training_data

__all__ = [
    'FeatureExtractor',
    'extract_training_data',
    'calculate_additional_features',
    'calculate_all_features',
    'calculate_cross_asset_features',
    'CROSS_ASSET_SYMBOLS'
]
