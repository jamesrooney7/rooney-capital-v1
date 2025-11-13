#!/usr/bin/env python3
"""Feature utilities for ML preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_normalised_feature_matrix(Xy: pd.DataFrame, scaler=None):
    """
    Build normalized feature matrix from raw features.

    Extracts and normalizes only feature columns, excluding target variables and metadata.

    IMPORTANT: To prevent data leakage, the scaler should be:
    - Fitted on TRAINING data only
    - Reused (passed in) for threshold and test sets

    Args:
        Xy: DataFrame with features and target variables
        scaler: Optional pre-fitted StandardScaler. If None, fits a new scaler on this data.
                For proper train/test splits, fit on train data and pass to test data.

    Returns:
        Tuple of (X_normalized, scaler) where:
        - X_normalized: DataFrame with ONLY normalized feature columns (excludes targets and metadata)
        - scaler: The StandardScaler used (either fitted or passed in)
    """
    # Columns to EXCLUDE (these are NOT features)
    exclude_cols = [
        # Metadata
        'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',

        # Price information (look-ahead bias)
        'Entry_Price', 'Exit_Price',

        # Target variables (what we're predicting)
        'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross',

        # Other metadata
        'Symbol', 'Trade_ID'
    ]

    # Find feature columns (numeric columns not in exclude list)
    feature_cols = [col for col in Xy.columns
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(Xy[col])]

    if not feature_cols:
        return pd.DataFrame(), scaler

    # Extract ONLY features (not the whole DataFrame!)
    X_features = Xy[feature_cols].copy()

    # Normalize features
    if scaler is None:
        # Fit new scaler (training data only!)
        scaler = StandardScaler()
        X_normalized_array = scaler.fit_transform(X_features.fillna(0))
    else:
        # Use pre-fitted scaler (for threshold/test data to prevent leakage)
        X_normalized_array = scaler.transform(X_features.fillna(0))

    X_normalized = pd.DataFrame(
        X_normalized_array,
        index=X_features.index,
        columns=X_features.columns
    )

    # Replace any inf or extreme values
    X_normalized = X_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_normalized, scaler
