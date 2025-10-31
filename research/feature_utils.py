#!/usr/bin/env python3
"""Feature utilities for ML preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_normalised_feature_matrix(Xy: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalized feature matrix from raw features.

    Normalizes all numeric columns except the target variables and metadata columns.

    Args:
        Xy: DataFrame with features and target variables

    Returns:
        DataFrame with normalized features
    """
    # Columns to exclude from normalization
    exclude_cols = [
        'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',
        'Entry_Price', 'Exit_Price',
        'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross',
        'Symbol', 'Trade_ID'
    ]

    # Find feature columns (numeric columns not in exclude list)
    feature_cols = [col for col in Xy.columns
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(Xy[col])]

    if not feature_cols:
        return Xy

    # Create copy
    result = Xy.copy()

    # Normalize features
    scaler = StandardScaler()
    result[feature_cols] = scaler.fit_transform(Xy[feature_cols].fillna(0))

    # Replace any inf or extreme values
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan)
    result[feature_cols] = result[feature_cols].fillna(0)

    return result
