"""
Configuration defaults and constants for ML Meta-Labeling System.
"""

# Walk-Forward Windows (Expanding, Non-Overlapping)
# Development Period: 2011-2020, Held-Out Test: 2021-2024
WALK_FORWARD_WINDOWS = [
    {
        "name": "Window 1",
        "train_start": "2011-01-01",
        "train_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2016-12-31",
        "train_years": 5
    },
    {
        "name": "Window 2",
        "train_start": "2011-01-01",
        "train_end": "2016-12-31",
        "test_start": "2017-01-01",
        "test_end": "2017-12-31",
        "train_years": 6
    },
    {
        "name": "Window 3",
        "train_start": "2011-01-01",
        "train_end": "2017-12-31",
        "test_start": "2018-01-01",
        "test_end": "2018-12-31",
        "train_years": 7
    },
    {
        "name": "Window 4",
        "train_start": "2011-01-01",
        "train_end": "2018-12-31",
        "test_start": "2019-01-01",
        "test_end": "2019-12-31",
        "train_years": 8
    },
    {
        "name": "Window 5",
        "train_start": "2011-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
        "train_years": 9
    }
]

# Held-Out Test Period (never touched during optimization)
HELD_OUT_TEST = {
    "start": "2021-01-01",
    "end": "2024-12-31"
}

# Development Period (all optimization happens here)
DEVELOPMENT_PERIOD = {
    "start": "2011-01-01",
    "end": "2020-12-31"
}

# Feature Filtering Patterns
def should_remove_column(col_name: str, remove_title_case: bool = True,
                        remove_enable_params: bool = True,
                        remove_vix: bool = True) -> bool:
    """
    Check if a column should be removed based on filtering criteria.

    Args:
        col_name: Column name to check
        remove_title_case: Remove columns with spaces (title case duplicates)
        remove_enable_params: Remove enable* parameter columns
        remove_vix: Remove all VIX-related features

    Returns:
        True if column should be removed, False otherwise
    """
    # Title case (has spaces)
    if remove_title_case and ' ' in col_name:
        return True

    # Enable parameters
    if remove_enable_params and col_name.lower().startswith('enable'):
        return True

    # VIX features (case-insensitive)
    if remove_vix and 'vix' in col_name.lower():
        return True

    return False


# Metadata columns (not features, but needed for processing)
METADATA_COLUMNS = {
    'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',
    'Entry_Price', 'Exit_Price', 'Symbol', 'Trade_ID'
}

# Target columns (what we're predicting)
TARGET_COLUMNS = {
    'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross'
}

# LightGBM Hyperparameter Search Space (Model N Informed)
LIGHTGBM_SEARCH_SPACE = {
    'num_leaves': {
        'min': 31,
        'max': 127,
        'default': 127,
        'type': 'int'
    },
    'max_depth': {
        'min': 5,
        'max': 9,
        'default': 7,
        'type': 'int'
    },
    'n_estimators': {
        'min': 500,
        'max': 1500,
        'default': 1200,
        'type': 'int'
    },
    'learning_rate': {
        'min': 0.01,
        'max': 0.1,
        'default': 0.03,
        'log_scale': True,
        'type': 'float'
    },
    'feature_fraction': {
        'min': 0.3,
        'max': 0.7,
        'default': 0.5,
        'type': 'float'
    },
    'bagging_fraction': {
        'min': 0.5,
        'max': 1.0,
        'default': 0.8,
        'type': 'float'
    },
    'min_data_in_leaf': {
        'min': 20,
        'max': 100,
        'default': 50,
        'type': 'int'
    },
    'reg_alpha': {
        'min': 1e-8,
        'max': 1.0,
        'default': 0.01,
        'log_scale': True,
        'type': 'float'
    },
    'reg_lambda': {
        'min': 1e-8,
        'max': 1.0,
        'default': 0.1,
        'log_scale': True,
        'type': 'float'
    }
}

# CatBoost Default Parameters
CATBOOST_DEFAULTS = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.03,
    'loss_function': 'Logloss',
    'verbose': False
}

# XGBoost Default Parameters
XGBOOST_DEFAULTS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'eta': 0.05,
    'lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'verbosity': 0
}

# Feature Selection Defaults
FEATURE_SELECTION_DEFAULTS = {
    'n_clusters': 30,
    'linkage_method': 'ward',
    'rf_n_estimators': 500,
    'min_own_instrument_features': 12,
    'min_cross_asset_features': 8,
    'min_risk_proxy_features': 6,
    'min_microstructure_features': 3
}

# Cross-Validation Defaults
CV_DEFAULTS = {
    'n_splits': 5,
    'embargo_days': 2,  # Conservative: 1 day label evaluation + 1 day buffer
    'holding_period': 8  # Maximum days
}

# Optuna Defaults
OPTUNA_DEFAULTS = {
    'n_trials': 100,
    'timeout': None,
    'n_jobs': -1,  # Use all cores
    'optimization_metric': 'auc',
    'pruning_patience': 25,
    'sampler': 'TPE'
}

# Data Preparation Defaults
DATA_DEFAULTS = {
    'lambda_decay': 0.10,  # Exponential recency weighting
    'min_samples_per_class': 500,
    'missing_value_threshold': 0.25,  # 25% - features with >25% missing values removed
    'portfolio_size': 1_000_000_000  # $1B for context
}

# Ensemble Meta-Learner Defaults
ENSEMBLE_DEFAULTS = {
    'use_ensemble': True,
    'meta_learner_C': 1.0,  # Logistic regression regularization
    'class_weight': 'balanced'
}

# Output File Naming Templates
OUTPUT_TEMPLATES = {
    'selected_features': '{symbol}_ml_meta_labeling_selected_features.json',
    'feature_clustering_report': '{symbol}_ml_meta_labeling_feature_clustering_report.txt',
    'walk_forward_results': '{symbol}_ml_meta_labeling_walk_forward_results.csv',
    'walk_forward_report': '{symbol}_ml_meta_labeling_walk_forward_report.txt',
    'optimal_hyperparameters': '{symbol}_ml_meta_labeling_optimal_hyperparameters.json',
    'optimization_history': '{symbol}_ml_meta_labeling_optimization_history.csv',
    'ensemble_weights': '{symbol}_ml_meta_labeling_ensemble_weights.json',
    'final_model': '{symbol}_ml_meta_labeling_final_model.pkl',
    'oos_predictions': '{symbol}_ml_meta_labeling_oos_predictions.csv',
    'executive_summary': '{symbol}_ml_meta_labeling_executive_summary.txt',
    'held_out_results': '{symbol}_ml_meta_labeling_held_out_results.json'
}
