"""
ML Integration Layer for Strategy Factory.

Provides tools to extract training data from vectorized strategies
for use with the ML pipeline (train_rf_cpcv_bo.py).
"""

from .feature_extractor import FeatureExtractor
from .extract_training_data import extract_training_data

__all__ = ['FeatureExtractor', 'extract_training_data']
