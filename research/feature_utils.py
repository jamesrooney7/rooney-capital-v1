#!/usr/bin/env python3
"""Feature utilities for ML preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_normalised_feature_matrix(Xy: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalized feature matrix from raw features.

    Extracts and normalizes only feature columns, excluding target variables and metadata.

    Args:
        Xy: DataFrame with features and target variables

    Returns:
        DataFrame with ONLY normalized feature columns (excludes targets and metadata)
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
        return pd.DataFrame()

    # Extract ONLY features (not the whole DataFrame!)
    X_features = Xy[feature_cols].copy()

    # Normalize features
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X_features.fillna(0)),
        index=X_features.index,
        columns=X_features.columns
    )

    # Replace any inf or extreme values
    X_normalized = X_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_normalized
