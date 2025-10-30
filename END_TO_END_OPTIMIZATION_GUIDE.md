# End-to-End Optimization Guide
## Complete Documentation for IBS Mean Reversion Strategy Optimization

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Signal Generation & Strategy Logic](#signal-generation--strategy-logic)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Optimization Methods](#optimization-methods)
6. [Training Approaches](#training-approaches)
7. [Performance Metrics](#performance-metrics)
8. [Usage Guide](#usage-guide)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Executive Summary

This system implements a sophisticated machine learning pipeline for optimizing an **IBS (Internal Bar Strength) mean reversion trading strategy**. The optimization combines multiple advanced techniques to find robust trading parameters while avoiding overfitting.

### Key Components

- **Strategy**: IBS-based mean reversion with volume/price filters
- **ML Model**: Random Forest classifier predicting trade outcomes
- **Optimization**: Random Search (120 trials) + Bayesian Optimization (300 trials)
- **Validation**: Combinatorial Purged Cross-Validation (CPCV) with time-based embargo
- **Metrics**: Sharpe Ratio, Deflated Sharpe Ratio, win rate, total return

### What Makes This Approach Robust

1. **Multiple Testing Correction**: Deflated Sharpe Ratio accounts for 420 trials
2. **Temporal Integrity**: Time-based embargo prevents label leakage
3. **Bias Elimination**: Three-way time split separates hyperparameter and threshold optimization
4. **Feature Engineering**: Expanding window percentiles prevent look-ahead bias
5. **Comprehensive Validation**: CPCV provides robust out-of-sample estimates

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA EXTRACTION                          │
│  extract_training_data.py                                   │
│  - Runs backtest with collect_filter_values=True            │
│  - Captures 50+ features at entry time                      │
│  - Records outcomes when positions close                    │
│  Output: training_data/{symbol}_training_data.csv           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE SCREENING                          │
│  - Univariate feature selection                             │
│  - SelectKBest with f_classif or mutual_info_classif        │
│  - Reduces 50+ features to top K (default: 20)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              RANDOM SEARCH (120 trials)                     │
│  - Uniform sampling of hyperparameter space                 │
│  - Each config evaluated via CPCV (5 folds, k=2)            │
│  - Tracks: mean Sharpe, DSR, win rate, total return         │
│  - Selects top performers for Bayesian optimization         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          BAYESIAN OPTIMIZATION (300 trials)                 │
│  - TPE sampler (Tree-structured Parzen Estimator)           │
│  - Refines search around promising regions                  │
│  - Same CPCV evaluation as Random Search                    │
│  - Optimizes Sharpe ratio                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              THRESHOLD OPTIMIZATION                         │
│                                                             │
│  APPROACH 1: Standard (Same Data)                           │
│  - Optimize threshold on same CPCV folds                    │
│  - Risk: 5-15% optimistic bias                              │
│                                                             │
│  APPROACH 2: Three-Way Split (Recommended)                  │
│  - Phase 1: Hyperparameters on 2010-2018                    │
│  - Phase 2: Threshold on 2019-2020                          │
│  - Phase 3: Final eval on 2021-2024                         │
│  - Eliminates threshold optimization bias                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRODUCTION MODEL                            │
│  - Best hyperparameters from optimization                   │
│  - Best threshold from threshold optimization               │
│  - Retrained on all available training data                 │
│  - Saved as: {symbol}_rf_model.pkl                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Signal Generation & Strategy Logic

### Internal Bar Strength (IBS)

The core signal is **IBS**, a normalized measure of where the close is within the day's range:

```python
IBS = (Close - Low) / (High - Low)
```

**Interpretation**:
- `IBS = 0.0`: Closed at the low (maximum bearishness)
- `IBS = 1.0`: Closed at the high (maximum bullishness)
- `IBS = 0.5`: Closed mid-range (neutral)

**Mean Reversion Logic**: When IBS is very low (e.g., < 0.3), the security is "oversold" and likely to bounce.

### Entry Conditions

The strategy enters long positions when ALL conditions are met:

1. **IBS Range**: `ibs_min <= IBS <= ibs_max` (default: 0.0 to 0.3)
2. **Volume Filter**: Volume >= Nth percentile over lookback window (e.g., 50th percentile over 252 days)
3. **Price Filter**: Close >= $N (e.g., $5) to avoid penny stocks
4. **SPY IBS Filter** (optional): SPY's IBS also in range
5. **Random Forest Prediction**: `model.predict_proba(features)[:, 1] >= threshold`

### Exit Conditions

Positions are exited when ANY condition is met:

1. **IBS Exit**: IBS rises above `ibs_exit` (e.g., 0.7) - profit target
2. **Max Hold Period**: Position held for `max_hold` days (default: 2 days)
3. **Trailing Stop**: Price drops `trail_pct` below highest close since entry (e.g., 5%)

**Important**: Exit parameters are **FIXED** (not optimized) to prevent overfitting.

### Why This Works

- **Momentum Exhaustion**: Extreme selling (low IBS) often reverses
- **Volume Confirmation**: High volume increases signal reliability
- **Risk Management**: Multiple exit conditions limit downside
- **ML Enhancement**: Random Forest filters out weak setups

---

## Feature Engineering Pipeline

### Feature Categories

The system extracts **50+ features** at trade entry, grouped into:

#### 1. Price Action Features
- `Close`, `High`, `Low`, `Open`
- `Close_pct_change`: 1-day return
- `High_to_Low_ratio`: Daily range normalized

#### 2. Volume Features
- `Volume`
- `Volume_pct_change`: 1-day volume change
- `Volume_percentile_N`: Volume percentile over N-day window (20/50/100/252 days)

#### 3. IBS Features
- `IBS`: Current IBS value
- `IBS_percentile_N`: IBS percentile over lookback (20/50/100 days)
- `IBS_mean_N`: Moving average of IBS
- `IBS_std_N`: Rolling standard deviation of IBS

#### 4. SPY (Market) Features
- `SPY_IBS`: S&P 500 IBS
- `SPY_Close_pct_change`: Market return
- `SPY_Volume_pct_change`: Market volume change

#### 5. Volatility Features
- `ATR_14`: 14-day Average True Range
- `ATR_percentile_N`: ATR percentile (volatility regime)

#### 6. Momentum Features
- `RSI_14`: 14-day Relative Strength Index
- `return_N`: N-day cumulative return (5/10/20 days)

#### 7. Moving Average Features
- `SMA_N`: Simple moving average (20/50/200 days)
- `Close_to_SMA_N_ratio`: Price relative to moving average

### Look-Ahead Bias Prevention

**Critical**: All features use **expanding window percentiles** to prevent look-ahead bias:

```python
# PercentileCache in ibs_strategy.py (lines 460-509)
class PercentileCache:
    """Maintains sorted list of historical values for fast percentile calculation."""

    def update(self, value):
        """Add new value to history."""
        bisect.insort(self._sorted_values, value)

    def percentile(self, pct):
        """Get percentile using ONLY historical data."""
        idx = int(len(self._sorted_values) * pct / 100)
        return self._sorted_values[idx]
```

**How It Works**:
1. On each bar, features are calculated using ONLY data up to that point
2. Percentiles expand as more data accumulates (e.g., day 50 uses 50 days, day 100 uses 100 days)
3. No future data ever leaks into feature calculations

### Feature Screening

To reduce dimensionality and remove noisy features:

```python
# SelectKBest with f_classif (ANOVA F-statistic)
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
```

**Methods**:
- `f_classif`: ANOVA F-statistic (linear relationships)
- `mutual_info_classif`: Mutual information (non-linear relationships)

**Default**: Select top 20 features from 50+ candidates.

---

## Optimization Methods

### 1. Random Search (120 Trials)

**Purpose**: Explore hyperparameter space uniformly to find promising regions.

**Hyperparameters Optimized**:

```python
{
    'n_estimators': [50, 100, 200, 300],          # Number of trees
    'max_depth': [3, 5, 7, 10, 15, None],         # Tree depth
    'min_samples_split': [2, 5, 10, 20],          # Min samples to split node
    'min_samples_leaf': [1, 2, 5, 10],            # Min samples in leaf
    'max_features': ['sqrt', 'log2', None],       # Features per split
    'class_weight': ['balanced', None],           # Handle class imbalance
    'bootstrap': [True, False],                   # Bootstrap samples
    'criterion': ['gini', 'entropy', 'log_loss'], # Split criterion
}
```

**Process**:
1. Sample each hyperparameter uniformly from its range
2. Train Random Forest with sampled config
3. Evaluate via CPCV (5 folds, k=2 test folds)
4. Record: Sharpe, DSR, win rate, total return, number of trades
5. Repeat 120 times

**Output**: CSV file with all trial results, sorted by Sharpe ratio.

### 2. Bayesian Optimization (300 Trials)

**Purpose**: Intelligently refine search around promising regions found by Random Search.

**Algorithm**: TPE (Tree-structured Parzen Estimator) via Optuna

**How TPE Works**:
1. Build probabilistic model of `P(hyperparameters | Sharpe ratio)`
2. Separate trials into "good" (high Sharpe) and "bad" (low Sharpe)
3. Model each group with Parzen estimators (kernel density)
4. Suggest next trial by maximizing `P(good) / P(bad)`
5. Update model with new trial result

**Advantages**:
- Focuses on promising regions faster than Random Search
- Handles discrete and continuous parameters
- Adapts to non-smooth objective functions

**Process**:
1. Initialize with Random Search results (warm start)
2. TPE sampler suggests next hyperparameter config
3. Evaluate via CPCV (same as Random Search)
4. Update Optuna study with result
5. Repeat 300 times

**Output**: Best hyperparameters saved to `{symbol}_best_rf_params.json`.

### 3. Random Forest Classifier

**Purpose**: Predict whether a trade setup will be profitable.

**Why Random Forest**:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- No need for feature scaling
- Handles mixed feature types

**Training Target**:
```python
y_binary = (y_return > 0).astype(int)  # 1 = win, 0 = loss
```

**Prediction**:
```python
proba = model.predict_proba(features)[:, 1]  # Probability of win
take_trade = proba >= threshold  # e.g., threshold = 0.52
```

**Threshold Optimization**: Threshold (e.g., 0.40-0.70) optimized separately to maximize Sharpe ratio.

### 4. Combinatorial Purged Cross-Validation (CPCV)

**Purpose**: Robust time-series cross-validation that prevents label leakage.

**Standard K-Fold Problems**:
- Trains on future data to predict past (breaks causality)
- No purging between train/test (overlapping positions leak labels)

**CPCV Solution**:

```
Timeline: [===============================================]
          2010                                         2024

5 Folds:  [F1][F2][F3][F4][F5]

k=2 Combinations (10 total):
  Test: [F1][F2], Train: [F3][F4][F5] + embargo
  Test: [F1][F3], Train: [F2][F4][F5] + embargo
  Test: [F1][F4], Train: [F2][F3][F5] + embargo
  ...
```

**Key Features**:

1. **Combinatorial**: Test on k=2 folds at a time (10 combinations from 5 folds)
2. **Purging**: Embargo period between train and test (default: 3 days)
3. **Temporal Ordering**: Only use train data before test data
4. **Label Protection**: Exclude trades within embargo of test period

**Embargo Calculation**:

```python
def embargoed_cpcv_splits(dates, n_splits=5, k_test=2, embargo_days=3):
    """Generate CPCV splits with TIME-BASED embargo."""

    # For each combination of k=2 test folds:
    for test_fold_combo in combinations(range(n_splits), k_test):
        # Get dates in test folds
        test_dates = dates[test_mask]

        # Calculate distance in DAYS from each sample to nearest test sample
        distances = min_days_to_test(dates, test_dates)

        # Train mask: exclude test fold AND samples within embargo_days
        train_mask = (~test_mask) & (distances > embargo_days)

        yield train_indices, test_indices
```

**Why 3 Days**:
- Max hold period: 2 days
- Buffer: 1 day
- Total: 3 days ensures no position overlap

**Performance Aggregation**:
```python
# Across all 10 CPCV splits:
sharpe_mean = np.mean([split_sharpe for split in splits])
sharpe_std = np.std([split_sharpe for split in splits])
dsr = deflated_sharpe_ratio(sharpe_mean, n=n_days, kurt_excess=kurt, m=n_effective)
```

---

## Training Approaches

### Approach 1: Standard CPCV (rf_cpcv_random_then_bo.py)

**Timeline**: Uses all available data (2010-2024) for both hyperparameter and threshold optimization.

**Process**:

1. **Feature Screening**: Select top 20 features using SelectKBest
2. **Random Search**: 120 trials, each evaluated via CPCV
3. **Bayesian Optimization**: 300 trials, TPE-guided, evaluated via CPCV
4. **Threshold Optimization**: Optimize threshold (0.40-0.70) on CPCV results
5. **Best Config**: Select hyperparameters + threshold with highest Sharpe
6. **Final Training**: Retrain on ALL data with best config

**Advantages**:
- Maximizes data usage
- Simpler workflow
- Good for smaller datasets

**Disadvantages**:
- Threshold optimized on same folds as hyperparameters (5-15% optimistic bias)
- Risk of overfitting to threshold

**When to Use**:
- Limited historical data (< 10 years)
- Quick prototyping
- When accepting slight optimistic bias

**Usage**:
```bash
python research/rf_cpcv_random_then_bo.py --symbol SPY --rs-trials 120 --bo-trials 300
```

---

### Approach 2: Three-Way Time Split (train_rf_three_way_split.py) ⭐ RECOMMENDED

**Timeline**: Splits data into three temporal periods with distinct purposes.

```
[========== 2010-2018 ==========][== 2019-2020 ==][=== 2021-2024 ===]
    Hyperparameter Tuning          Threshold Opt     Final Evaluation
         (Phase 1)                    (Phase 2)          (Phase 3)
```

#### Phase 1: Hyperparameter Tuning (2010-2018)

**Purpose**: Find best Random Forest hyperparameters.

**Process**:
1. Feature screening on 2010-2018 data
2. Random Search: 120 trials via CPCV (5 folds, k=2)
3. Bayesian Optimization: 300 trials via CPCV
4. Select hyperparameters with highest mean Sharpe

**Output**:
- Best hyperparameters (n_estimators, max_depth, etc.)
- Selected features
- Trial history CSV

#### Phase 2: Threshold Optimization (2019-2020)

**Purpose**: Find best probability threshold on SEPARATE data.

**Process**:
1. Train Random Forest on ALL 2010-2018 data with best hyperparameters
2. Predict on 2019-2020 data (completely separate from Phase 1)
3. For each threshold (0.40 to 0.70 in steps of 0.01):
   - Apply threshold to predictions
   - Simulate trades
   - Calculate Sharpe ratio
4. Select threshold with highest Sharpe (minimum 100 trades)

**Output**:
- Best threshold (e.g., 0.52)
- Threshold performance metrics

**Why This Works**: Threshold chosen on data NEVER seen during hyperparameter optimization.

#### Phase 3: Final Evaluation (2021-2024)

**Purpose**: Get TRUE out-of-sample performance estimate.

**Process**:
1. Retrain Random Forest on ALL 2010-2020 data (Phase 1 + Phase 2) with best hyperparameters
2. Predict on 2021-2024 data with best threshold
3. Calculate final metrics: Sharpe, DSR, win rate, total return

**Output**:
- Production model (`{symbol}_rf_model.pkl`)
- True OOS performance metrics
- Detailed trade log

**Why This Works**: Test data NEVER used for ANY optimization decisions.

#### Advantages of Three-Way Split

✅ **Eliminates Threshold Optimization Bias**: Separate data for hyperparameters vs threshold
✅ **True OOS Estimate**: Phase 3 test data completely untouched
✅ **Temporal Integrity**: Strict chronological ordering
✅ **Production-Ready**: Model trained on max available data (Phase 1 + 2)
✅ **Realistic Expectations**: Test performance typically 10-20% lower than training (expected degradation)

#### Disadvantages

❌ **Requires More Data**: Needs ~14 years minimum (8 train, 2 threshold, 4 test)
❌ **Reduced Training Data**: Phase 1 only uses 2010-2018 (~8 years)
❌ **Longer Runtime**: Three distinct phases vs one

#### When to Use

✅ **Required for**:
- Production deployment
- Publishing research
- Regulatory compliance
- Accurate performance expectations

❌ **Skip if**:
- < 10 years data available
- Quick prototyping only
- Already validated strategy

#### Usage

```bash
# Full three-way split
python research/train_rf_three_way_split.py \
    --symbol SPY \
    --rs-trials 120 \
    --bo-trials 300 \
    --folds 5 \
    --k-test 2 \
    --embargo-days 3

# Custom date splits
python research/train_rf_three_way_split.py \
    --symbol SPY \
    --train-end 2018-12-31 \
    --threshold-end 2020-12-31
```

#### Expected Output

**Phase 1 (Hyperparameters)**:
```
Phase 1: Hyperparameter Tuning on 2010-01-01 to 2018-12-31
Feature screening: Selected 20 features from 52
Random Search (120 trials)...
  Trial 120/120: Sharpe=0.45, DSR=0.38, Wins=52.3%, Trades=450
Bayesian Optimization (300 trials)...
  Trial 300/300: Sharpe=0.52, DSR=0.44, Wins=54.1%, Trades=478
Best hyperparameters: {'n_estimators': 200, 'max_depth': 7, ...}
```

**Phase 2 (Threshold)**:
```
Phase 2: Threshold Optimization on 2019-01-01 to 2020-12-31
Training final model on 2010-2018...
Optimizing threshold on 2019-2020...
  Threshold 0.52: Sharpe=0.48, Trades=120, Wins=55.0%
Best threshold: 0.52
```

**Phase 3 (Test)**:
```
Phase 3: Final Evaluation on 2021-01-01 to 2024-12-31
Retraining production model on 2010-2020...
Evaluating on 2021-2024...

Final Test Metrics:
  Total Trades: 285
  Win Rate: 51.9%
  Total Return: 18.3%
  Sharpe Ratio: 0.41
  Deflated Sharpe: 0.35
  Max Drawdown: -8.2%
```

**Performance Degradation**: Test Sharpe (0.41) being 10-20% lower than training (0.52) is **EXPECTED and HEALTHY**. Suspicious if test >= training.

---

## Performance Metrics

### 1. Sharpe Ratio

**Definition**: Risk-adjusted return measuring excess return per unit of volatility.

```python
sharpe_ratio = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

**Interpretation**:
- `< 0.5`: Poor
- `0.5 - 1.0`: Acceptable
- `1.0 - 2.0`: Good
- `> 2.0`: Excellent (rare, verify no overfitting)

**Why We Use It**: Standard metric for comparing strategies with different risk profiles.

### 2. Deflated Sharpe Ratio (DSR)

**Definition**: Sharpe ratio corrected for multiple testing and non-normality.

**Formula** (Bailey & López de Prado, 2014):

```python
def deflated_sharpe_ratio(sr, n, kurt_excess=0, m=1, rho=0.0, sr_ref=0.0):
    """
    sr: Observed Sharpe ratio
    n: Number of returns (e.g., 252 trading days)
    kurt_excess: Excess kurtosis of returns
    m: Number of trials tested
    rho: Average correlation between trials
    sr_ref: Reference Sharpe (e.g., risk-free rate)
    """
    # Adjust for multiple testing
    n_effective = m / (1 + (m - 1) * rho)

    # Variance of Sharpe ratio under null
    var_sr = (1 + 0.5 * sr**2 - kurt_excess * sr**2 / 4) / (n - 1)
    var_sr_adj = var_sr * (1 + (n_effective - 1) * rho)

    # Z-score
    z = (sr - sr_ref) / sqrt(var_sr_adj)

    # Deflated Sharpe
    dsr = sr - z * sqrt(var_sr)
    return dsr
```

**Our Implementation**:
```python
# Account for 420 trials (120 Random + 300 Bayesian)
rho_avg = 0.7  # Typical correlation between hyperparameter configs
n_effective = max(1, int(420 / (1 + (420 - 1) * 0.7)))  # ≈ 1.4
dsr = deflated_sharpe_ratio(sr, n=252, kurt_excess=daily.kurt(), m=n_effective)
```

**Interpretation**:
- `DSR > 0.5`: Statistically significant edge after multiple testing correction
- `DSR < 0.3`: Likely overfitting, strategy may not be robust

**Why Critical**: Without DSR correction, testing 420 configurations can make random noise look like signal.

### 3. Win Rate

**Definition**: Percentage of trades that are profitable.

```python
win_rate = (number of winning trades) / (total trades) * 100
```

**Interpretation**:
- `< 45%`: Poor (unless high win size compensates)
- `45-55%`: Acceptable
- `55-60%`: Good
- `> 60%`: Excellent (verify not overfitting)

**Note**: Win rate alone is misleading. A 60% win rate with small wins and large losses still loses money.

### 4. Total Return

**Definition**: Cumulative percentage return over the period.

```python
total_return = (final_equity / initial_equity - 1) * 100
```

**Why We Track It**: Sharpe measures risk-adjusted returns, but total return shows absolute profit.

### 5. Max Drawdown

**Definition**: Largest peak-to-trough decline in equity.

```python
max_drawdown = max((peak - trough) / peak for all peak-trough pairs)
```

**Interpretation**:
- `< 10%`: Low risk
- `10-20%`: Moderate risk
- `20-30%`: High risk
- `> 30%`: Very high risk (may be unacceptable)

### 6. Number of Trades

**Importance**: Strategies with < 100 trades have unreliable statistics.

**Requirements**:
- Minimum 100 trades for any optimization phase
- Minimum 200 trades preferred for final evaluation
- More trades = more reliable Sharpe ratio estimate

---

## Usage Guide

### Step 1: Extract Training Data

```bash
# Extract features and outcomes for a symbol
python research/extract_training_data.py --symbol SPY

# Output: training_data/SPY_training_data.csv
```

**What This Does**:
- Runs backtest with `collect_filter_values=True`
- Captures 50+ features at each entry
- Records outcome when position closes
- Saves to CSV with features (X) and target (y)

### Step 2A: Standard CPCV Training

```bash
# Run Random Search + Bayesian Optimization
python research/rf_cpcv_random_then_bo.py \
    --symbol SPY \
    --rs-trials 120 \
    --bo-trials 300 \
    --folds 5 \
    --k-test 2 \
    --embargo-days 3 \
    --k-features 20 \
    --seed 42

# Outputs:
# - SPY_rf_random_search_results.csv
# - SPY_rf_bo_results.csv
# - SPY_best_rf_params.json
# - SPY_rf_model.pkl
```

**Parameters**:
- `--rs-trials`: Number of Random Search trials (default: 120)
- `--bo-trials`: Number of Bayesian Optimization trials (default: 300)
- `--folds`: Number of CPCV folds (default: 5)
- `--k-test`: Number of test folds per combination (default: 2)
- `--embargo-days`: Embargo period in days (default: 3)
- `--k-features`: Number of features to select (default: 20)
- `--screen-method`: Feature selection ('f_classif' or 'mutual_info', default: 'f_classif')
- `--seed`: Random seed for reproducibility

### Step 2B: Three-Way Split Training (Recommended)

```bash
# Run three-phase optimization
python research/train_rf_three_way_split.py \
    --symbol SPY \
    --train-end 2018-12-31 \
    --threshold-end 2020-12-31 \
    --rs-trials 120 \
    --bo-trials 300 \
    --folds 5 \
    --k-test 2 \
    --embargo-days 3 \
    --k-features 20 \
    --seed 42

# Outputs (all saved to models/ directory):
# Phase 1:
# - SPY_three_way_phase1_random_search.csv
# - SPY_three_way_phase1_bo_study.pkl
# - SPY_three_way_phase1_best_params.json
# - SPY_three_way_phase1_feature_list.json
#
# Phase 2:
# - SPY_three_way_phase2_threshold_search.csv
# - SPY_three_way_phase2_best_threshold.json
#
# Phase 3:
# - SPY_three_way_phase3_test_metrics.json
# - SPY_three_way_phase3_production_model.pkl
# - SPY_three_way_full_report.txt
```

**Additional Parameters**:
- `--train-end`: End date for Phase 1 training (default: 2018-12-31)
- `--threshold-end`: End date for Phase 2 threshold optimization (default: 2020-12-31)
- `--min-threshold-trades`: Minimum trades required for threshold optimization (default: 100)

### Step 3: Deploy Production Model

```bash
# Use the trained model in production
python your_production_script.py --model models/SPY_rf_model.pkl
```

**Model Contains**:
- Trained Random Forest classifier
- Best hyperparameters
- Feature list (order matters!)
- Probability threshold
- Metadata (symbol, training dates, performance metrics)

---

## Best Practices

### 1. Data Requirements

✅ **Minimum**: 10 years of daily data
✅ **Recommended**: 14+ years for three-way split (8 train, 2 threshold, 4 test)
✅ **Ideal**: 20+ years for robust statistics

### 2. Feature Engineering

✅ **Always use expanding windows** for percentiles (no look-ahead bias)
✅ **Screen features** to remove noise (default: top 20 of 50+)
✅ **Verify no future data** leaks into feature calculations
❌ **Never use forward-looking indicators** (future peaks, future volatility, etc.)

### 3. Hyperparameter Optimization

✅ **Start with Random Search** (120 trials) to explore space
✅ **Refine with Bayesian Optimization** (300 trials)
✅ **Use CPCV** (5 folds, k=2) for all evaluations
✅ **Set time-based embargo** (3 days for 1-2 day holds)
❌ **Don't optimize exit parameters** (keeps IBS exit, trailing stop, max hold fixed)

### 4. Threshold Optimization

✅ **Use three-way split** for production deployment
✅ **Require minimum 100 trades** per threshold evaluation
✅ **Test range 0.40-0.70** (balance precision vs recall)
❌ **Don't optimize threshold on same data as hyperparameters**

### 5. Performance Validation

✅ **Always report DSR** (accounts for 420 trials)
✅ **Expect test performance 10-20% lower** than training
✅ **Verify minimum 200 trades** on test set
✅ **Check max drawdown < 20%** for acceptable risk
❌ **Suspicious if test Sharpe >= training Sharpe** (likely overfitting)

### 6. Model Deployment

✅ **Retrain on all available data** before production (Phase 1 + Phase 2 combined)
✅ **Save model, features, threshold** together
✅ **Version control** all models with timestamps
✅ **Monitor live performance** vs backtested expectations
❌ **Don't deploy if DSR < 0.3** (not statistically significant)

### 7. Common Pitfalls

❌ **Look-Ahead Bias**: Using future data in features (e.g., full-dataset percentiles)
❌ **Label Leakage**: Train/test overlap in positions (no embargo)
❌ **Overfitting**: Too many trials without DSR correction
❌ **Sample Size**: < 100 trades gives unreliable metrics
❌ **Data Snooping**: Re-optimizing after seeing test results

---

## Troubleshooting

### Issue: Low Number of Trades (< 100)

**Symptoms**:
```
WARNING: Only 45 trades in test set (minimum 100 recommended)
Sharpe ratio: 1.85 (not reliable due to small sample)
```

**Causes**:
- IBS range too narrow (e.g., 0.1-0.2 instead of 0.0-0.3)
- Volume filter too restrictive (e.g., 90th percentile instead of 50th)
- Probability threshold too high (e.g., 0.70 instead of 0.52)
- Limited historical data

**Solutions**:
1. Widen IBS range in `IBS_ENTRY_EXIT_DEFAULTS`:
   ```python
   'ibs_min': 0.0,  # Lower bound
   'ibs_max': 0.3,  # Upper bound
   ```
2. Lower volume filter percentile:
   ```python
   'volume_percentile': 50,  # Instead of 75 or 90
   ```
3. Lower probability threshold during optimization (allow more trades)
4. Use longer historical period

---

### Issue: Test Performance Much Higher Than Training

**Symptoms**:
```
Phase 1 (Training): Sharpe=0.45, Win Rate=52%
Phase 3 (Test): Sharpe=0.68, Win Rate=58%
```

**Causes** (all indicate problems):
- Look-ahead bias in features
- Label leakage (insufficient embargo)
- Lucky test period (market regime change)
- Data snooping (re-optimized after seeing test results)

**Solutions**:
1. **Audit for look-ahead bias**:
   ```bash
   # Check feature calculation uses only historical data
   grep -n "percentile" src/strategy/ibs_strategy.py
   ```
2. **Increase embargo period**:
   ```python
   embargo_days = 5  # Instead of 3
   ```
3. **Use different test period** (multiple test periods if possible)
4. **Never re-optimize** after viewing test results

**Expected**: Test Sharpe should be 10-20% lower than training. If higher, investigate.

---

### Issue: DSR Much Lower Than Sharpe

**Symptoms**:
```
Sharpe Ratio: 0.65
Deflated Sharpe: 0.12
```

**Causes**:
- Many trials tested (420 configs)
- High correlation between trials (rho=0.7)
- High kurtosis in returns (fat tails)
- Overfitting to noise

**Interpretation**:
- DSR < 0.3 suggests strategy not statistically significant
- DSR 0.3-0.5 suggests marginal edge
- DSR > 0.5 suggests robust edge

**Solutions**:
1. **Accept reality**: DSR correctly accounts for multiple testing
2. **Reduce trials** (but less thorough optimization):
   ```bash
   --rs-trials 50 --bo-trials 100  # Instead of 120 + 300
   ```
3. **Lower correlation assumption** (if justified):
   ```python
   rho_avg = 0.5  # Instead of 0.7 (requires validation)
   ```
4. **Improve strategy fundamentals** (better signal, better features)

**Don't**: Ignore DSR and only report Sharpe (misleading).

---

### Issue: Optimization Taking Too Long

**Symptoms**:
```
Estimated time: 18 hours for 420 trials
```

**Causes**:
- Too many CPCV folds/combinations
- Large dataset (many years of daily data)
- Many features (50+ without screening)

**Solutions**:
1. **Reduce CPCV folds**:
   ```bash
   --folds 3 --k-test 1  # 3 combinations instead of 10
   ```
2. **Reduce trials**:
   ```bash
   --rs-trials 50 --bo-trials 150  # Instead of 120 + 300
   ```
3. **Screen features earlier**:
   ```bash
   --k-features 15  # Instead of 20
   ```
4. **Use smaller date range** for Phase 1:
   ```bash
   --train-end 2016-12-31  # Fewer years
   ```
5. **Parallel processing** (if available):
   ```python
   # Modify code to use joblib for parallel CPCV evaluation
   from joblib import Parallel, delayed
   ```

**Trade-off**: Faster optimization but less thorough search.

---

### Issue: High Win Rate But Low Sharpe

**Symptoms**:
```
Win Rate: 62%
Sharpe Ratio: 0.15
Average Win: +0.3%
Average Loss: -1.2%
```

**Causes**:
- Small wins, large losses (inverse of what we want)
- Trailing stop too tight (cuts winners short)
- IBS exit too aggressive (exits winners early)
- Loses money despite high win rate

**Solutions**:
1. **Loosen trailing stop**:
   ```python
   'trail_pct': 0.10  # 10% instead of 5%
   ```
2. **Raise IBS exit**:
   ```python
   'ibs_exit': 0.80  # Instead of 0.70
   ```
3. **Extend max hold**:
   ```python
   'max_hold': 3  # Days instead of 2
   ```
4. **Check for data issues** (wrong sign on returns?)

**Note**: High win rate with low Sharpe indicates win/loss asymmetry is wrong.

---

### Issue: Overfitting Suspected

**Symptoms**:
- Training Sharpe 0.85, Test Sharpe 0.30 (> 50% degradation)
- DSR very low despite high Sharpe
- Perfect performance on training, poor on test
- Win rate drops from 60% to 48%

**Causes**:
- Too many features (overfitting noise)
- Threshold optimized on training data
- Not enough regularization in Random Forest
- Small sample size (< 200 trades)

**Solutions**:
1. **Use three-way split** (eliminates threshold bias)
2. **Reduce features**:
   ```bash
   --k-features 10  # Instead of 20
   ```
3. **Increase min_samples_leaf** in hyperparameter space:
   ```python
   'min_samples_leaf': [5, 10, 20]  # Instead of [1, 2, 5, 10]
   ```
4. **Use simpler model**:
   ```python
   'max_depth': [3, 5, 7]  # Instead of [3, 5, 7, 10, 15, None]
   ```
5. **Get more data** (more years of history)

---

## References

### Academic Papers

1. **Combinatorial Purged Cross-Validation**:
   - López de Prado, M. (2018). "Advances in Financial Machine Learning"
   - Chapter 7: Cross-Validation in Finance

2. **Deflated Sharpe Ratio**:
   - Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
   - Journal of Portfolio Management, 40(5), 94-107

3. **Bayesian Optimization**:
   - Bergstra, J., Yamins, D., Cox, D. D. (2013). "Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures"
   - ICML 2013

4. **IBS Strategy**:
   - Original research by Larry Connors
   - "How Markets Really Work" (2012)

### Tools & Libraries

- **Optuna**: Bayesian optimization framework (https://optuna.org)
- **scikit-learn**: Random Forest, feature selection (https://scikit-learn.org)
- **backtrader**: Backtesting framework (https://www.backtrader.com)

---

## Appendix: File Structure

```
rooney-capital-v1/
│
├── research/
│   ├── extract_training_data.py           # Step 1: Extract features and outcomes
│   ├── rf_cpcv_random_then_bo.py          # Step 2A: Standard CPCV training
│   ├── train_rf_three_way_split.py        # Step 2B: Three-way split training
│   └── train_rf_cpcv_bo.py                # Alternative: Bayesian only (no random search)
│
├── src/
│   └── strategy/
│       └── ibs_strategy.py                # Core strategy: IBS signals, exits, features
│
├── training_data/
│   └── {SYMBOL}_training_data.csv         # Extracted features and outcomes
│
├── models/
│   ├── {SYMBOL}_rf_model.pkl              # Trained production model
│   ├── {SYMBOL}_best_rf_params.json       # Best hyperparameters
│   ├── {SYMBOL}_rf_random_search_results.csv    # Random search trial history
│   ├── {SYMBOL}_rf_bo_results.csv         # Bayesian optimization trial history
│   │
│   # Three-way split outputs:
│   ├── {SYMBOL}_three_way_phase1_best_params.json
│   ├── {SYMBOL}_three_way_phase2_best_threshold.json
│   ├── {SYMBOL}_three_way_phase3_test_metrics.json
│   └── {SYMBOL}_three_way_phase3_production_model.pkl
│
└── Documentation/
    ├── THREE_WAY_SPLIT_GUIDE.md           # Quick start guide for three-way split
    ├── OPTIMIZATION_FIXES_ANALYSIS.md     # Technical analysis of optimization fixes
    ├── OPTIMIZATION_FIXES_IMPLEMENTED.md  # Implementation summary
    ├── OPTIMIZATION_VERIFICATION_REPORT.md # Look-ahead bias audit
    └── END_TO_END_OPTIMIZATION_GUIDE.md   # This document
```

---

## Quick Start Checklist

### For Standard CPCV Approach:

- [ ] Extract training data: `python research/extract_training_data.py --symbol SPY`
- [ ] Run optimization: `python research/rf_cpcv_random_then_bo.py --symbol SPY`
- [ ] Check results: Review `SPY_best_rf_params.json` and trial CSVs
- [ ] Verify performance: Sharpe > 0.5, DSR > 0.3, Trades > 200
- [ ] Deploy model: Use `SPY_rf_model.pkl` in production

### For Three-Way Split Approach (Recommended):

- [ ] Extract training data: `python research/extract_training_data.py --symbol SPY`
- [ ] Run three-way split: `python research/train_rf_three_way_split.py --symbol SPY`
- [ ] Review Phase 1: Check training Sharpe, DSR, feature importance
- [ ] Review Phase 2: Verify best threshold on separate data
- [ ] Review Phase 3: Confirm test Sharpe 10-20% below training (healthy)
- [ ] Check DSR: Must be > 0.3 (preferably > 0.5)
- [ ] Deploy: Use `SPY_three_way_phase3_production_model.pkl`

---

## Summary

This end-to-end optimization system combines:

✅ **Robust Signal**: IBS mean reversion with volume/price filters
✅ **Advanced ML**: Random Forest with 420-trial hyperparameter optimization
✅ **Rigorous Validation**: CPCV with time-based embargo prevents leakage
✅ **Statistical Rigor**: Deflated Sharpe corrects for multiple testing
✅ **Bias Elimination**: Three-way split separates hyperparameter and threshold optimization
✅ **Production Ready**: Comprehensive documentation and tested implementation

**Recommended Workflow**:
1. Use **three-way split** for production deployment
2. Require **DSR > 0.5** before going live
3. Expect **10-20% degradation** from training to test
4. **Monitor live performance** and retrain periodically

For questions or issues, refer to the troubleshooting section or examine the source code directly.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Maintained By**: Rooney Capital
