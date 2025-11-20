"""
Component 1: Data Preparation Pipeline

Handles loading, filtering, and preprocessing of training data for ML meta-labeling.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

from .config_defaults import (
    should_remove_column,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    DATA_DEFAULTS
)

logger = logging.getLogger(__name__)


class DataPreparation:
    """Prepare and filter training data for ML meta-labeling."""

    def __init__(
        self,
        symbol: str,
        data_dir: str = "data/training",
        remove_title_case: bool = True,
        remove_enable_params: bool = True,
        remove_vix: bool = True,
        lambda_decay: float = DATA_DEFAULTS['lambda_decay'],
        min_samples_per_class: int = DATA_DEFAULTS['min_samples_per_class'],
        missing_value_threshold: float = DATA_DEFAULTS['missing_value_threshold'],
        task_type: str = 'classification'
    ):
        """
        Initialize data preparation pipeline.

        Args:
            symbol: Trading symbol (e.g., 'ES')
            data_dir: Directory containing transformed_features.csv files
            remove_title_case: Remove columns with spaces
            remove_enable_params: Remove enable* parameter columns
            remove_vix: Remove VIX features
            lambda_decay: Exponential decay rate for recency weighting
            min_samples_per_class: Minimum samples required per class
            missing_value_threshold: Maximum allowed missing value fraction
            task_type: 'classification' or 'regression' (predict P&L instead of win/loss)
        """
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.remove_title_case = remove_title_case
        self.remove_enable_params = remove_enable_params
        self.remove_vix = remove_vix
        self.lambda_decay = lambda_decay
        self.min_samples_per_class = min_samples_per_class
        self.missing_value_threshold = missing_value_threshold
        self.task_type = task_type

        # Will be populated by load_and_prepare()
        self.raw_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.removed_columns: List[str] = []

    def load_and_prepare(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and prepare data.

        Returns:
            Tuple of (prepared_dataframe, feature_column_names)
        """
        logger.info(f"Loading data for {self.symbol}...")

        # Load raw data
        csv_path = self.data_dir / f"{self.symbol}_transformed_features.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {csv_path}\n"
                f"Please ensure the file exists at this location."
            )

        self.raw_df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.raw_df)} trades with {len(self.raw_df.columns)} columns")

        # Parse dates
        self._parse_dates()

        # Filter columns
        self._filter_columns()

        # Validate data quality
        self._validate_data()

        # Handle missing values
        self._handle_missing_values()

        # Calculate sample weights
        self._calculate_sample_weights()

        logger.info(f"Data preparation complete: {len(self.feature_columns)} features selected")
        return self.filtered_df, self.feature_columns

    def _parse_dates(self):
        """Parse date columns."""
        date_col = None
        if 'Date/Time' in self.raw_df.columns:
            date_col = 'Date/Time'
        elif 'Date' in self.raw_df.columns:
            date_col = 'Date'
        else:
            raise ValueError("No date column found (expected 'Date/Time' or 'Date')")

        self.raw_df['Date'] = pd.to_datetime(self.raw_df[date_col])
        logger.info(f"Date range: {self.raw_df['Date'].min()} to {self.raw_df['Date'].max()}")

    def _filter_columns(self):
        """Filter columns based on removal criteria."""
        logger.info("Filtering columns...")

        kept_columns = []
        removed_columns = []

        for col in self.raw_df.columns:
            if should_remove_column(
                col,
                self.remove_title_case,
                self.remove_enable_params,
                self.remove_vix
            ):
                removed_columns.append(col)
            else:
                kept_columns.append(col)

        self.removed_columns = removed_columns
        self.filtered_df = self.raw_df[kept_columns].copy()

        # Identify feature columns (exclude metadata and targets)
        self.feature_columns = [
            col for col in kept_columns
            if col not in METADATA_COLUMNS and col not in TARGET_COLUMNS
            and pd.api.types.is_numeric_dtype(self.filtered_df[col])
        ]

        logger.info(f"Removed {len(removed_columns)} columns:")
        logger.info(f"  - Title case (with spaces): {sum(1 for c in removed_columns if ' ' in c)}")
        logger.info(f"  - Enable parameters: {sum(1 for c in removed_columns if c.lower().startswith('enable'))}")
        logger.info(f"  - VIX features: {sum(1 for c in removed_columns if 'vix' in c.lower())}")
        logger.info(f"Kept {len(kept_columns)} columns ({len(self.feature_columns)} features)")

    def _validate_data(self):
        """Validate data quality."""
        logger.info("Validating data quality...")

        if self.task_type == 'classification':
            # Check for target column
            if 'y_binary' not in self.filtered_df.columns:
                raise ValueError("Target column 'y_binary' not found in data")

            # Check class balance
            class_counts = self.filtered_df['y_binary'].value_counts()
            logger.info(f"Class distribution: {dict(class_counts)}")

            for class_label, count in class_counts.items():
                if count < self.min_samples_per_class:
                    raise ValueError(
                        f"Insufficient samples for class {class_label}: "
                        f"{count} < {self.min_samples_per_class}"
                    )
        else:  # regression
            # Check for P&L target column
            if 'y_pnl_usd' not in self.filtered_df.columns:
                # Try to calculate from y_return
                if 'y_return' in self.filtered_df.columns:
                    logger.info("Calculating y_pnl_usd from y_return...")
                    avg_price = 4500  # Approximate ES price
                    self.filtered_df['y_pnl_usd'] = self.filtered_df['y_return'] * avg_price * 50
                else:
                    raise ValueError("Neither 'y_pnl_usd' nor 'y_return' found in data")

            # Log P&L statistics
            pnl_stats = self.filtered_df['y_pnl_usd'].describe()
            logger.info(f"P&L statistics:\n{pnl_stats}")

        # Check for all-NaN features
        all_nan_features = [
            col for col in self.feature_columns
            if self.filtered_df[col].isna().all()
        ]
        if all_nan_features:
            logger.warning(f"Removing {len(all_nan_features)} all-NaN features: {all_nan_features}")
            self.feature_columns = [col for col in self.feature_columns if col not in all_nan_features]

    def _handle_missing_values(self):
        """Handle missing values in features."""
        logger.info("Handling missing values...")

        missing_stats = []
        for col in self.feature_columns:
            missing_frac = self.filtered_df[col].isna().mean()
            if missing_frac > 0:
                missing_stats.append((col, missing_frac))

        if missing_stats:
            # Sort by missing fraction
            missing_stats.sort(key=lambda x: x[1], reverse=True)

            # Remove features with too many missing values
            high_missing = [
                col for col, frac in missing_stats
                if frac > self.missing_value_threshold
            ]
            if high_missing:
                logger.warning(
                    f"Removing {len(high_missing)} features with >{self.missing_value_threshold*100:.1f}% "
                    f"missing values"
                )
                logger.debug(f"Removed features: {high_missing}")
                self.feature_columns = [col for col in self.feature_columns if col not in high_missing]

            # Fill remaining missing values with 0 (after normalization will be mean)
            for col in self.feature_columns:
                if self.filtered_df[col].isna().any():
                    self.filtered_df.loc[:, col] = self.filtered_df[col].fillna(0)

            logger.info(f"Missing value handling complete. {len(self.feature_columns)} features remain.")

    def _calculate_sample_weights(self):
        """Calculate exponential recency weights for samples."""
        logger.info("Calculating sample weights...")

        # Sort by date
        self.filtered_df = self.filtered_df.sort_values('Date').reset_index(drop=True)

        # Calculate exponential weights
        # Most recent sample gets weight 1.0, older samples decay exponentially
        n_samples = len(self.filtered_df)
        indices = np.arange(n_samples)
        weights = np.exp(-self.lambda_decay * (n_samples - 1 - indices) / n_samples)

        # Normalize to sum to n_samples (maintains effective sample size)
        weights = weights / weights.mean()

        self.filtered_df['sample_weight'] = weights

        logger.info(f"Sample weights calculated (lambda={self.lambda_decay:.3f})")
        logger.info(f"  Weight range: {weights.min():.3f} to {weights.max():.3f}")
        logger.info(f"  Mean weight: {weights.mean():.3f}")

    def get_date_split(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for a specific date range.

        Args:
            start_date: Start date (inclusive), format 'YYYY-MM-DD'
            end_date: End date (inclusive), format 'YYYY-MM-DD'

        Returns:
            Filtered dataframe for date range
        """
        if self.filtered_df is None:
            raise ValueError("Must call load_and_prepare() first")

        df = self.filtered_df.copy()

        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        logger.info(f"Date split: {len(df)} samples from {start_date} to {end_date}")
        return df

    def get_features_and_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features, target, and weights from dataframe.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (X_features, y_target, sample_weights)
        """
        X = df[self.feature_columns].copy()

        # Select target based on task type
        if self.task_type == 'classification':
            y = df['y_binary'].copy()
        else:  # regression
            y = df['y_pnl_usd'].copy()

        weights = df['sample_weight'].copy() if 'sample_weight' in df.columns else pd.Series(
            np.ones(len(df)),
            index=df.index
        )

        return X, y, weights

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the data."""
        if self.filtered_df is None:
            raise ValueError("Must call load_and_prepare() first")

        stats = {
            'symbol': self.symbol,
            'task_type': self.task_type,
            'total_samples': int(len(self.filtered_df)),
            'n_features': int(len(self.feature_columns)),
            'date_range': {
                'start': str(self.filtered_df['Date'].min()),
                'end': str(self.filtered_df['Date'].max())
            },
            'columns_removed': int(len(self.removed_columns)),
            'columns_kept': int(len(self.filtered_df.columns))
        }

        if self.task_type == 'classification':
            # Convert numpy types to Python types for JSON serialization
            class_dist = self.filtered_df['y_binary'].value_counts()
            class_distribution = {int(k): int(v) for k, v in class_dist.items()}
            stats['class_distribution'] = class_distribution
        else:  # regression
            pnl_stats = self.filtered_df['y_pnl_usd'].describe()
            stats['pnl_statistics'] = {
                'mean': float(pnl_stats['mean']),
                'std': float(pnl_stats['std']),
                'min': float(pnl_stats['min']),
                'max': float(pnl_stats['max'])
            }

        return stats
