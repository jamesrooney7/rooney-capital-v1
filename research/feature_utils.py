#!/usr/bin/env python3
"""Feature utilities for ML preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_normalised_feature_matrix(
    Xy: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Build normalized feature matrix from raw features.

    Extracts and normalizes only feature columns, excluding target variables and metadata.

    IMPORTANT: To avoid lookahead bias in time-series splits:
    - FIT the scaler on training data only (fit_scaler=True, scaler=None)
    - TRANSFORM validation/test data with the fitted scaler (fit_scaler=False, scaler=fitted_scaler)

    Args:
        Xy: DataFrame with features and target variables
        scaler: Pre-fitted StandardScaler (optional). If None, creates new scaler.
        fit_scaler: If True, fits the scaler on this data. If False, only transforms.

    Returns:
        Tuple of (normalized_features_df, fitted_scaler)
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
        return pd.DataFrame(), scaler or StandardScaler()

    # Extract ONLY features (not the whole DataFrame!)
    X_features = Xy[feature_cols].copy()

    # Normalize features
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        # FIT and TRANSFORM (use for training data)
        X_normalized = pd.DataFrame(
            scaler.fit_transform(X_features.fillna(0)),
            index=X_features.index,
            columns=X_features.columns
        )
    else:
        # TRANSFORM only (use for validation/test data with pre-fitted scaler)
        X_normalized = pd.DataFrame(
            scaler.transform(X_features.fillna(0)),
            index=X_features.index,
            columns=X_features.columns
        )

    # Replace any inf or extreme values
    X_normalized = X_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_normalized, scaler
