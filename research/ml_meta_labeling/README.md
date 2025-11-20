# ML Meta-Labeling System

A comprehensive machine learning meta-labeling implementation for filtering trading signals using ensemble models with walk-forward validation.

## Overview

This system implements a sophisticated ML meta-labeling approach that:
- Uses **hierarchical clustering** to select diverse, representative features
- Employs **Purged K-Fold CV** to prevent information leakage
- Optimizes **LightGBM hyperparameters** using Optuna TPE
- Validates performance through **walk-forward analysis** (2011-2020)
- Optionally builds **ensemble models** (LightGBM + CatBoost + XGBoost)
- Evaluates on **held-out test period** (2021-2024)

## Quick Start

### 1. Prerequisites

Ensure you have the required dependencies:
```bash
pip install lightgbm catboost xgboost optuna scikit-learn scipy pandas numpy joblib
```

### 2. Run Optimization

Basic usage for ES (S&P 500 E-mini):
```bash
python research/ml_meta_labeling/ml_meta_labeling_optimizer.py --symbol ES
```

With custom parameters:
```bash
python research/ml_meta_labeling/ml_meta_labeling_optimizer.py \
    --symbol ES \
    --n-clusters 30 \
    --n-trials 100 \
    --cv-folds 5 \
    --embargo-days 60 \
    --use-ensemble
```

### 3. Review Results

All results are saved to `research/ml_meta_labeling/results/{SYMBOL}/`:
- `ES_ml_meta_labeling_executive_summary.txt` - High-level summary
- `ES_ml_meta_labeling_walk_forward_results.csv` - Window-by-window performance
- `ES_ml_meta_labeling_final_model.pkl` - Trained production model
- `ES_ml_meta_labeling_selected_features.json` - Selected features and clustering info

## System Architecture

### Component 1: Data Preparation
- Loads transformed features from `data/training/{symbol}_transformed_features.csv`
- Filters out title case duplicates, enable parameters, and VIX features
- Applies exponential recency weighting
- Handles missing values

### Component 2: Feature Clustering and Selection
- Computes absolute correlation matrix
- Performs hierarchical clustering (Ward linkage)
- Trains preliminary Random Forest for MDA importance
- Selects top feature from each of 30 clusters
- Validates cross-asset diversity

### Component 3: Purged K-Fold Cross-Validation
- 5-fold temporal splits
- 60-day embargo period to prevent label leakage
- Respects temporal ordering
- Standard k-fold (k_test=1) for hyperparameter optimization

### Component 4: LightGBM Model Training
- Balanced class weights
- Configurable hyperparameters (Model N informed defaults)
- Early stopping support
- Feature importance tracking

### Component 5: Optuna TPE Optimization
- Tree-structured Parzen Estimator (Bayesian optimization)
- 100 trials per walk-forward window
- Median pruning for efficiency
- Hyperparameter importance analysis

### Component 6: Walk-Forward Validation
- 5 expanding windows (2016-2020 test years)
- Per-window reoptimization
- Walk-Forward Efficiency (WFE) calculation
- Hyperparameter stability tracking

### Component 7: Ensemble Model (Optional)
- LightGBM + CatBoost + XGBoost base models
- Logistic regression meta-learner
- Out-of-fold predictions for meta-training
- Model weight analysis

## Walk-Forward Windows

Development Period: **2011-2020** (all optimization happens here)

| Window | Training Period | Test Period | Train Years |
|--------|----------------|-------------|-------------|
| 1 | 2011-2015 | 2016 | 5 years |
| 2 | 2011-2016 | 2017 | 6 years |
| 3 | 2011-2017 | 2018 | 7 years |
| 4 | 2011-2018 | 2019 | 8 years |
| 5 | 2011-2019 | 2020 | 9 years |

Held-Out Test: **2021-2024** (never touched during optimization)

## Feature Filtering

The system automatically removes:
1. **Title Case Duplicates**: Features with spaces (e.g., "ES Hourly Return")
2. **Enable Parameters**: Boolean parameter columns (e.g., "enableESReturnHour")
3. **VIX Features**: All VIX-related features (not available)

Result: **489 → 277 columns** → **30 selected features** after clustering

## Command-Line Arguments

### Required
- `--symbol`: Trading symbol (e.g., ES, NQ, RTY)

### Data
- `--data-dir`: Training data directory (default: `data/training`)
- `--output-dir`: Output directory (default: `research/ml_meta_labeling/results`)

### Feature Selection
- `--n-clusters`: Number of feature clusters (default: 30)
- `--linkage-method`: Clustering linkage method (default: `ward`)
- `--rf-n-estimators`: RF trees for MDA importance (default: 500)

### Cross-Validation
- `--cv-folds`: Number of CV folds (default: 5)
- `--embargo-days`: Embargo period in days (default: 60)

### Optimization
- `--n-trials`: Optuna trials per window (default: 100)
- `--use-ensemble`: Use ensemble model (default: True)

### Other
- `--seed`: Random seed (default: 42)
- `--lambda-decay`: Exponential decay for sample weights (default: 0.10)

## Performance Metrics

### Walk-Forward Metrics
- **AUC-ROC**: Area under ROC curve
- **Sharpe Ratio**: Risk-adjusted return
- **WFE**: Walk-Forward Efficiency (OOS / IS performance)
- **Win Rate**: Percentage of profitable trades

### Key Expectations
- **WFE > 0.6**: Excellent generalization
- **WFE 0.4-0.6**: Acceptable
- **WFE < 0.4**: Warning, potential overfitting

## Output Files

All files saved to `research/ml_meta_labeling/results/{SYMBOL}/`:

| File | Description |
|------|-------------|
| `*_executive_summary.txt` | High-level results summary |
| `*_walk_forward_results.csv` | Window-by-window metrics |
| `*_final_model.pkl` | Trained production model |
| `*_selected_features.json` | Selected features with clustering info |
| `*_oos_predictions.csv` | Out-of-sample predictions |
| `*_held_out_results.json` | Held-out test metrics (2021-2024) |
| `*_ensemble_weights.json` | Ensemble model weights (if enabled) |
| `hyperparameter_stability.csv` | Hyperparameter evolution across windows |

## Implementation Notes

### Data Requirements
- Transformed features CSV must exist at `data/training/{symbol}_transformed_features.csv`
- Must have `Date` or `Date/Time` column for temporal splitting
- Must have `y_binary` target column
- Minimum 500 samples per class

### Computational Requirements
- **Single symbol (100 trials/window)**: ~4-8 hours on modern CPU
- **Ensemble mode**: +50% runtime
- **Memory**: ~4-8 GB RAM for typical datasets

### Reproducibility
- Set `--seed 42` for reproducible results
- Same seed → identical hyperparameter search → identical model

## Troubleshooting

### Issue: "Training data not found"
**Solution**: Ensure transformed features CSV exists:
```bash
ls data/training/ES_transformed_features.csv
```

### Issue: Optimization too slow
**Solution**: Reduce trials or disable ensemble:
```bash
--n-trials 50 --use-ensemble false
```

### Issue: Low WFE (<0.4)
**Possible causes**:
- Market regime change between train/test periods
- Overfitting to training period
- Insufficient training data

**Solutions**:
- Review walk-forward consistency (is Sharpe positive in most windows?)
- Consider reducing model complexity (fewer features, simpler hyperparameters)
- Analyze hyperparameter stability across windows

## Next Steps

1. **Run on ES**: Start with S&P 500 E-mini
2. **Review Results**: Check WFE and held-out performance
3. **Run on Other Instruments**: NQ, RTY, YM, etc. (18 instruments total)
4. **Deploy**: Use trained models for production signal filtering

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Bergstra, J., et al. (2013). "Making a Science of Model Search"

---

**Status**: ✅ Ready for production use
**Created**: 2025-01-18
**Author**: Rooney Capital
