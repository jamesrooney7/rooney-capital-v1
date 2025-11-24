# Strategy Factory: Gold Standard Strategy Identification System

## Executive Summary

The Strategy Factory is a systematic research pipeline designed to identify, validate, and rank trading strategies that feed into Rooney Capital's ML optimization system. This guide establishes a best-in-class methodology for discovering robust, diverse strategies that provide the raw material (10,000+ trades) needed for effective machine learning meta-labeling.

**Core Philosophy:**
- Strategy research generates "ore" (weak but real edges with high volume)
- ML pipeline refines it into "gold" (high-Sharpe filtered strategies)
- Focus on robustness, diversity, and statistical rigor over raw performance
- Think portfolio-level, not single-strategy optimization

**Implementation Approach (AGREED - Jan 2025):**
- **Timeline**: 1 week to production-ready system
- **Scope**: 10 high-priority strategies from catalogue (Tier 1)
- **Data**: 15-minute bars, 2010-2024, local file storage
- **Symbols**: ES (Phase 1) → All symbols (Phase 2): ES, NQ, YM, RTY, GC, SI, HG, CL, NG, 6A, 6B, 6C, 6J, 6M, 6N, 6S, TLT
- **Exit Strategy**: Fixed exits initially (1.0 ATR stop/take), optimize later
- **Compute**: 16 CPU cores, local execution, parallel processing
- **Integration**: Leverage existing backtest_runner.py, extract_training_data.py, train_rf_cpcv_bo.py

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [Strategy Archetype System](#strategy-archetype-system)
4. [Three-Phase Pipeline](#three-phase-pipeline)
5. [Success Criteria & Filters](#success-criteria--filters)
6. [Statistical Rigor Framework](#statistical-rigor-framework)
7. [Implementation Specifications](#implementation-specifications)
8. [Integration with ML Pipeline](#integration-with-ml-pipeline)
9. [Tier 1 Strategy Implementations](#tier-1-strategy-implementations)
10. [Execution Guide](#execution-guide)
11. [Experiment Tracking & Reproducibility](#experiment-tracking--reproducibility)

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                      STRATEGY FACTORY PIPELINE                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: 50+ Strategy Hypotheses (Organized by Archetype)           │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  PHASE 1: Raw Strategy Screening (ES Only, 2010-2024)        │ │
│  │  • Parallel backtesting (5,000+ combinations)                │ │
│  │  • Trade volume filter (≥10,000 trades)                      │ │
│  │  • Statistical significance testing                          │ │
│  │  • Regime analysis (bull/bear/sideways)                      │ │
│  │  • Parameter stability checks                                │ │
│  │  OUTPUT: 15-25 promising strategies                          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  PHASE 2: Multi-Symbol Validation (ES/NQ/YM/RTY)            │ │
│  │  • Cross-sectional consistency testing                       │ │
│  │  • Correlation matrix analysis                               │ │
│  │  • Portfolio construction simulation                         │ │
│  │  • Incremental value assessment                              │ │
│  │  OUTPUT: 5-10 validated strategies                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  PHASE 3: ML Pipeline Integration                            │ │
│  │  • Feature extraction (existing pipeline)                    │ │
│  │  • CPCV + Bayesian optimization (existing pipeline)          │ │
│  │  • Performance comparison (raw vs ML-filtered)               │ │
│  │  • Meta-learning feedback capture                            │ │
│  │  OUTPUT: 2-4 production-ready strategies                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│         ↓                                                           │
│  FEEDBACK LOOP: ML results inform next research cycle             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Principles

### 2.1 High Volume, Low Precision → ML Refinement

**Traditional Approach (Wrong for ML Pipeline):**
- Tight filters for high win rate (50-60%)
- Selective entries for maximum Sharpe
- Result: 1,000 trades, Sharpe 1.2, insufficient data for ML

**Factory Approach (Correct for ML Pipeline):**
- Wide net with permissive filters
- High trade frequency (10,000+ trades)
- Raw Sharpe 0.2-0.5 (weak but real edge)
- Result: ML refines 10,000 → 3,500 trades, Sharpe 0.3 → 1.5+

**Analogy:** Gold refinery needs lots of ore (10k trades) to extract gold (3.5k quality trades)

### 2.2 Robustness Over Raw Performance

**Priority Ranking:**
1. **Consistency across regimes** (works in bull, bear, sideways)
2. **Parameter stability** (doesn't collapse from small tweaks)
3. **Statistical significance** (edge isn't random luck)
4. **Feature richness** (generates 20-50 diverse features for ML)
5. **Raw Sharpe ratio** (just needs to be positive)

**Why:** Your ML pipeline enhances robust weak edges better than fragile strong edges.

### 2.3 Portfolio-Level Thinking

**Don't optimize:**
- Individual strategy Sharpe in isolation

**Do optimize:**
- Portfolio Sharpe with strategy included
- Correlation matrix (low correlation = better diversification)
- Incremental value (does this add alpha beyond what we have?)

**Example:**
- Strategy A: Sharpe 1.5, correlation 0.9 with existing → REJECT
- Strategy B: Sharpe 0.8, correlation 0.2 with existing → ACCEPT

### 2.4 Archetype-Based Organization

**Don't test:**
- 50 random strategy variations

**Do test:**
- 8 strategy archetypes × 6-8 variations each
- Ensures diversification across market inefficiencies
- Tests different hypotheses systematically

### 2.5 Statistical Rigor for Multiple Testing

**Problem:** With 5,000 backtests, ~250 will have Sharpe > 1.0 by pure chance

**Solutions:**
- Multiple testing corrections (False Discovery Rate)
- Monte Carlo permutation tests (shuffle trades, test if edge is real)
- Walk-forward validation (train/test split)
- Regime analysis (consistent across market conditions)

---

## 3. Strategy Archetype System

Organize research into 8 fundamental strategy categories that test different market hypotheses:

### 3.1 Mean Reversion (High Priority)

**Hypothesis:** Price extremes tend to revert to the mean

**Strategy Examples:**
- **IBS (Internal Bar Strength):** Your existing strategy - price position within bar
- **RSI Divergence:** Price makes lower low, RSI makes higher low
- **Bollinger Reversal:** Price touches lower band, reverses
- **Opening Range Fade:** Fade large opening moves
- **Gap Fade:** Trade against overnight gaps
- **VIX Spike Fade:** Buy equity dips when VIX spikes

**Key Parameters to Test:**
- Lookback periods (5, 10, 20, 50 bars)
- Entry thresholds (20th, 30th percentile)
- Exit conditions (mean reversion, time-based, profit target)

**Expected Volume:** High (daily signals possible)
**ML Enhancement Potential:** 2-3x Sharpe improvement

---

### 3.2 Momentum

**Hypothesis:** Trends persist in the short-term

**Strategy Examples:**
- **Trend Following:** Moving average crossovers with volume confirmation
- **Breakout:** New 20-day highs with volume surge
- **Higher Highs/Lower Lows:** Sequential price momentum
- **Volume Surge:** Large volume precedes continued move
- **Gap Continuation:** Follow-through after significant gaps

**Key Parameters to Test:**
- Trend periods (10, 20, 50, 100 bars)
- Breakout thresholds (1%, 2%, ATR-based)
- Volume confirmations (1.5x, 2x average)

**Expected Volume:** Medium (fewer signals than mean reversion)
**ML Enhancement Potential:** 1.5-2x Sharpe improvement

---

### 3.3 Volatility-Based

**Hypothesis:** Volatility regimes predict future price behavior

**Strategy Examples:**
- **Volatility Expansion:** Enter when volatility increases from low base
- **Volatility Contraction:** Enter after volatility compression
- **ATR Percentile:** Trade based on ATR rank vs historical
- **Intraday Range:** Large/small intraday ranges predict next-day behavior
- **Realized Vol vs Implied Vol:** Trade the vol spread

**Key Parameters to Test:**
- Volatility lookback (20, 50, 100 days)
- Percentile thresholds (10th, 90th)
- Expansion/contraction definitions

**Expected Volume:** Medium-High
**ML Enhancement Potential:** 2-2.5x Sharpe improvement

---

### 3.4 Cross-Asset (High Priority)

**Hypothesis:** Related assets provide predictive signals

**Strategy Examples:**
- **TLT Divergence:** Equity/bond correlation breaks (you use this)
- **VIX Regime Filter:** VIX level predicts equity behavior (you use this)
- **Pairs Z-Score:** Trade ES vs NQ spread (you use this)
- **Bond-Equity Correlation:** Regime shifts in correlation
- **Dollar Strength:** DXY movements predict commodity futures
- **Gold-SPX Ratio:** Risk-on/risk-off transitions

**Key Parameters to Test:**
- Lookback for correlation (30, 60, 120 days)
- Z-score thresholds (-2, -1.5, +1.5, +2)
- Divergence magnitude (5%, 10%)

**Expected Volume:** High (daily signals)
**ML Enhancement Potential:** 3-4x Sharpe improvement (highest!)

**Why High Priority:** Your existing system shows cross-asset features work exceptionally well with ML.

---

### 3.5 Seasonality/Calendar

**Hypothesis:** Time-based patterns exist in markets

**Strategy Examples:**
- **Day of Week:** Monday reversal, Friday trend continuation
- **Month Effect:** Turn-of-month effect
- **FOMC Drift:** Pre-announcement drift patterns
- **Options Expiry:** OpEx Friday patterns
- **Quarter End:** Window dressing effects
- **Holiday Effect:** Pre/post-holiday behavior

**Key Parameters to Test:**
- Entry timing (day before, day of, day after)
- Hold periods (intraday, 1-3 days)
- Filter conditions (only in certain regimes)

**Expected Volume:** Medium (event-based)
**ML Enhancement Potential:** 2x Sharpe improvement

---

### 3.6 Market Microstructure

**Hypothesis:** Orderflow and market mechanics predict price

**Strategy Examples:**
- **Volume Profile:** Trade near high-volume nodes
- **Opening Auction Imbalance:** Order imbalance predicts direction
- **Time and Sales Flow:** Aggressive buying/selling patterns
- **Bid-Ask Spread Patterns:** Spread widening/tightening signals
- **Market-on-Close Imbalance:** MOC orders predict close

**Key Parameters to Test:**
- Volume thresholds (1.5x, 2x average)
- Imbalance ratios (60/40, 70/30, 80/20)
- Time windows (first 30min, last 30min)

**Expected Volume:** High (intraday signals)
**ML Enhancement Potential:** 2.5x Sharpe improvement

**Data Requirements:** May require tick data (Databento provides this)

---

### 3.7 Multi-Timeframe

**Hypothesis:** Multiple timeframes provide confirmation/filters

**Strategy Examples:**
- **Daily Signal + Hourly Exit:** Daily trend, hourly mean reversion exit
- **Weekly Filter + Daily Signal:** Trade daily only with weekly trend
- **Multiple Timeframe Confluence:** Entry when 3 timeframes align
- **Timeframe Divergence:** Trade when hourly diverges from daily

**Key Parameters to Test:**
- Timeframe combinations (1H/4H/1D, 1D/1W)
- Alignment thresholds (2 of 3, 3 of 3)
- Signal priority (which timeframe dominates)

**Expected Volume:** Medium (filtered by higher timeframe)
**ML Enhancement Potential:** 2x Sharpe improvement

---

### 3.8 Regime-Adaptive

**Hypothesis:** Optimal strategy changes with market regime

**Strategy Examples:**
- **Volatility Regime Switch:** Mean reversion in low vol, momentum in high vol
- **Trend vs Range Detection:** Different logic for trending vs ranging
- **Market Correlation Regime:** High/low correlation environments
- **HMM Regime Identification:** Hidden Markov Models for regime detection

**Key Parameters to Test:**
- Regime definitions (VIX level, ADX, correlation)
- Threshold values for regime switches
- Strategy combinations per regime

**Expected Volume:** High (always trading, just different strategies)
**ML Enhancement Potential:** 2-3x Sharpe improvement

**Note:** More complex implementation, consider Phase 2 after simpler strategies validated.

---

## 4. Three-Phase Pipeline

### Phase 1: Raw Strategy Screening (ES Only)

**Objective:** Rapidly filter 50+ strategies to 15-25 promising candidates

**Date Range:** 2010-01-01 to 2024-12-31 (14 years)

**Process:**

1. **Parameter Grid Generation**
   - Each strategy: 4-6 parameters
   - Each parameter: 3-4 values (coarse grid)
   - Total combinations per strategy: 81-256
   - Total backtests: 50 strategies × 100 combos = 5,000

2. **Parallel Backtesting**
   - Concurrent execution (16-32 workers)
   - Each backtest: Strategy + params + ES + 2010-2024
   - Results logged to SQLite database
   - Real-time progress tracking (tqdm)

3. **Filter Application (Sequential)**

   **Critical Filters (Must Pass ALL):**
   - ✅ Trade count ≥ 10,000 (2010-2024)
   - ✅ Raw Sharpe ≥ 0.2 (weak but real edge)
   - ✅ Profit Factor ≥ 1.15 (barely profitable is fine)
   - ✅ Max Drawdown ≤ 30%
   - ✅ Win Rate ≥ 35% (not lottery tickets)

4. **Robustness Testing (For Survivors)**

   **A. Walk-Forward Validation**
   ```
   Train: 2010-2021 (11 years)
   Test:  2022-2024 (3 years, out-of-sample)
   
   Pass Criteria:
   - Train Sharpe ≥ 0.3
   - Test Sharpe ≥ 0.15 (allowing some degradation)
   - Test Sharpe / Train Sharpe ≥ 0.5 (not complete collapse)
   ```

   **B. Regime Analysis**
   ```
   Bull Regime:   2010-2019, 2023-2024 (low VIX, uptrend)
   Bear Regime:   2020 COVID, 2022 (high VIX, drawdown)
   Sideways:      2015-2016 (range-bound)
   
   Pass Criteria:
   - Sharpe ≥ 0.2 in at least 2 of 3 regimes
   - No regime with Sharpe < -0.3 (avoid catastrophic failures)
   ```

   **C. Parameter Stability Test**
   ```
   For each parameter:
   - Test ±10% variations
   - Calculate Sharpe at each variation
   
   Pass Criteria:
   - Sharpe range / Mean Sharpe < 40%
   - No parameter where ±10% change causes Sharpe < 0
   
   Example:
   RSI Period = 14: Sharpe = 0.35
   RSI Period = 12.6 (-10%): Sharpe = 0.28
   RSI Period = 15.4 (+10%): Sharpe = 0.32
   Range = 0.32 - 0.28 = 0.04
   Mean = (0.35 + 0.28 + 0.32) / 3 = 0.317
   Ratio = 0.04 / 0.317 = 12.6% < 40% ✅ PASS
   ```

5. **Statistical Significance Testing**

   **A. Multiple Testing Correction (False Discovery Rate)**
   ```python
   from statsmodels.stats.multitest import multipletests
   
   # With 5,000 backtests, expect ~250 Sharpe > 1.0 by chance
   # FDR correction: Adjust p-values for multiple comparisons
   
   p_values = [monte_carlo_test(strategy) for strategy in candidates]
   reject, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
   
   # Keep only strategies where reject=True (statistically significant)
   ```

   **B. Monte Carlo Permutation Test**
   ```python
   def monte_carlo_test(strategy_returns, n_simulations=1000):
       """
       Test if Sharpe ratio is due to skill or luck
       
       Method:
       1. Shuffle trade outcomes randomly (destroy temporal structure)
       2. Calculate Sharpe of shuffled returns
       3. Repeat 1000 times
       4. p-value = % of shuffled Sharpes > actual Sharpe
       """
       actual_sharpe = calculate_sharpe(strategy_returns)
       
       shuffled_sharpes = []
       for _ in range(n_simulations):
           shuffled = np.random.permutation(strategy_returns)
           shuffled_sharpes.append(calculate_sharpe(shuffled))
       
       p_value = np.mean(shuffled_sharpes >= actual_sharpe)
       
       # p < 0.05 means actual Sharpe is statistically significant
       return p_value
   ```

6. **Feature Richness Check**
   ```python
   # Each strategy must generate sufficient features for ML
   features = strategy.collect_features()
   
   Pass Criteria:
   - Feature count ≥ 20
   - Feature diversity (not all price-based):
     * Price features: 30-40%
     * Volume features: 15-25%
     * Cross-asset features: 15-25%
     * Technical indicators: 20-30%
   - No single feature with >40% importance (prevents overfitting)
   ```

**Phase 1 Output:**
- 15-25 strategies that passed all filters
- Ranked by composite score (see Section 5.3)
- Detailed performance reports per strategy
- Parameter heatmaps showing stable regions
- Regime breakdown charts

**Review Gate:** Manual review of top candidates before Phase 2

---

### Phase 2: Multi-Symbol Validation

**Objective:** Ensure strategies generalize and add portfolio value

**Symbols:** ES, NQ, YM, RTY (equity index futures cluster)

**Process:**

1. **Cross-Sectional Testing**
   ```
   For each Phase 1 winner:
   - Run backtest on ES (already done in Phase 1)
   - Run backtest on NQ (mini Nasdaq)
   - Run backtest on YM (mini Dow)
   - Run backtest on RTY (mini Russell)
   
   Compare:
   - Sharpe ratios across symbols
   - Correlation of returns
   - Drawdown consistency
   ```

2. **Pass/Fail Criteria**
   ```
   Must Pass:
   - ≥3 of 4 symbols have Sharpe ≥ 0.3
   - Average Sharpe across 4 symbols ≥ 0.5
   - No symbol with Sharpe < -0.2 (avoid disasters)
   - Max Drawdown < 30% on all symbols
   
   Example:
   Strategy: RSI_Divergence_v3
   ES:  Sharpe 0.45, DD 22%  ✅
   NQ:  Sharpe 0.52, DD 25%  ✅
   YM:  Sharpe 0.31, DD 28%  ✅
   RTY: Sharpe 0.18, DD 32%  ⚠️ (low Sharpe + high DD)
   
   3 of 4 pass → PASS OVERALL
   ```

3. **Correlation Matrix Analysis**
   ```python
   # Compare new strategy with existing live strategies
   correlation_matrix = calculate_correlations([
       existing_strategy_1_returns,
       existing_strategy_2_returns,
       new_strategy_returns
   ])
   
   Pass Criteria:
   - Correlation with each existing strategy < 0.7
   - Average correlation < 0.5 (strong diversification)
   
   Example:
   New Strategy vs Strategy A: 0.45 ✅
   New Strategy vs Strategy B: 0.62 ✅
   New Strategy vs Strategy C: 0.38 ✅
   Average: (0.45 + 0.62 + 0.38) / 3 = 0.48 ✅
   ```

4. **Portfolio Construction Simulation**
   ```python
   def portfolio_simulation(existing_strategies, new_strategy):
       """
       Test incremental value of adding new strategy
       """
       # Portfolio without new strategy
       port_baseline = combine_strategies(existing_strategies, equal_weight=True)
       baseline_sharpe = calculate_sharpe(port_baseline)
       baseline_dd = max_drawdown(port_baseline)
       
       # Portfolio with new strategy added
       all_strategies = existing_strategies + [new_strategy]
       port_enhanced = combine_strategies(all_strategies, equal_weight=True)
       enhanced_sharpe = calculate_sharpe(port_enhanced)
       enhanced_dd = max_drawdown(port_enhanced)
       
       # Metrics
       sharpe_improvement = enhanced_sharpe - baseline_sharpe
       diversification_ratio = portfolio_volatility / weighted_avg_volatility
       
       Pass Criteria:
       - Sharpe improvement > 0.05 (meaningful addition)
       - Diversification ratio > 1.1 (portfolio effect)
       - Max Drawdown not worse by >2% (risk management)
       
       return {
           'baseline_sharpe': baseline_sharpe,
           'enhanced_sharpe': enhanced_sharpe,
           'sharpe_delta': sharpe_improvement,
           'div_ratio': diversification_ratio,
           'decision': 'PASS' if sharpe_improvement > 0.05 else 'REJECT'
       }
   ```

5. **Incremental Alpha Test (Regression)**
   ```python
   from sklearn.linear_model import LinearRegression
   
   # Does new strategy have alpha beyond existing portfolio?
   X = portfolio_returns.values.reshape(-1, 1)
   y = new_strategy_returns.values
   
   model = LinearRegression().fit(X, y)
   alpha = model.intercept_  # Excess return
   beta = model.coef_[0]      # Market beta
   
   # T-test for alpha significance
   residuals = y - model.predict(X)
   alpha_stderr = np.std(residuals) / np.sqrt(len(y))
   t_stat = alpha / alpha_stderr
   p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y)-2))
   
   Pass Criteria:
   - p_value < 0.05 (alpha is statistically significant)
   - alpha > 0 (positive excess return)
   ```

**Phase 2 Output:**
- 5-10 strategies that generalize across symbols
- Correlation matrix visualization
- Portfolio simulation results
- Incremental alpha statistics
- Recommendation: Which strategies to move to ML pipeline

**Review Gate:** Final manual review before ML integration

---

### Phase 3: ML Pipeline Integration

**Objective:** Run full ML optimization on validated strategies

**Process:**

1. **Feature Extraction**
   ```bash
   # Use existing extract_training_data.py
   python research/extract_training_data.py \
       --strategy RSI_Divergence_v3 \
       --symbol ES \
       --start 2010-01-01 \
       --end 2024-12-31
   
   # Output: extracted_data/ES_RSI_Divergence_v3_features.csv
   # Contains: 10,000+ rows (trades) × 50+ columns (features + labels)
   ```

2. **ML Training (CPCV + Bayesian Optimization)**
   ```bash
   # Use existing train_rf_cpcv_bo.py
   python research/train_rf_cpcv_bo.py \
       --symbol ES \
       --strategy RSI_Divergence_v3 \
       --n_trials 100 \
       --cv_splits 5
   
   # Performs:
   # - Combinatorial Purged Cross-Validation (CPCV)
   # - Bayesian hyperparameter optimization (Optuna TPE)
   # - Embargo periods to prevent leakage
   # - Fixed 0.50 threshold (no overfitting)
   
   # Output: models/ES_RSI_Divergence_v3_rf_model.pkl
   ```

3. **Backtesting with ML Filter**
   ```bash
   # Use existing backtest runner with ML models
   python research/backtest_runner.py \
       --strategy RSI_Divergence_v3 \
       --symbol ES \
       --ml_model models/ES_RSI_Divergence_v3_rf_model.pkl \
       --start 2022-01-01 \
       --end 2024-12-31
   
   # Tests on out-of-sample period (2022-2024)
   ```

4. **Performance Comparison**
   ```python
   # Compare raw vs ML-filtered
   comparison = {
       'raw_sharpe': 0.35,
       'ml_sharpe': 1.42,
       'improvement': 1.42 / 0.35,  # 4.06x ← EXCELLENT
       
       'raw_trades': 10234,
       'ml_trades': 3581,  # 35% kept
       'trade_efficiency': 3581 / 10234,
       
       'raw_profit_factor': 1.18,
       'ml_profit_factor': 1.67,
       
       'raw_max_dd': 24%,
       'ml_max_dd': 15%,
       
       'ml_precision': 0.62,  # 62% of ML-selected trades profitable
       'ml_recall': 0.85      # 85% of good trades kept
   }
   ```

5. **Meta-Learning Capture (Feedback Loop)**
   ```python
   # Log to meta-database for future research cycles
   meta_results = {
       'strategy_name': 'RSI_Divergence_v3',
       'archetype': 'mean_reversion',
       'raw_sharpe': 0.35,
       'ml_sharpe': 1.42,
       'ml_boost': 4.06,
       'feature_count': 47,
       'feature_diversity_score': 0.73,
       'regime_consistency': 0.82,
       'param_stability': 0.91,
       'cross_symbol_avg_sharpe': 0.41
   }
   
   # Save for meta-analysis
   save_meta_results('meta_learning.db', meta_results)
   ```

**Phase 3 Output:**
- 2-4 production-ready strategies
- ML-filtered performance metrics
- Feature importance reports
- Walk-forward validation results
- Recommendation for live trading deployment

---

## 5. Success Criteria & Filters

### 5.1 Phase 1 Filters (Raw Strategy)

**Critical Filters (Gate 1):**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Trade Count | ≥ 10,000 (2010-2024) | ML requires high volume for training |
| Raw Sharpe Ratio | ≥ 0.2 | Weak but real edge (ML will enhance) |
| Profit Factor | ≥ 1.15 | Barely profitable is acceptable |
| Max Drawdown | ≤ 30% | Risk-adjusted, not catastrophic |
| Win Rate | ≥ 35% | Avoid lottery-ticket strategies |

**Robustness Filters (Gate 2):**

| Test | Pass Criteria |
|------|---------------|
| Walk-Forward | Test Sharpe / Train Sharpe ≥ 0.5 |
| Regime Analysis | Sharpe ≥ 0.2 in ≥2 of 3 regimes |
| Parameter Stability | Sharpe variation < 40% from ±10% param tweaks |
| Multiple Testing | FDR-corrected p-value < 0.05 |
| Monte Carlo | Permutation test p-value < 0.05 |

**Feature Quality (Gate 3):**

| Metric | Threshold |
|--------|-----------|
| Feature Count | ≥ 20 features |
| Feature Diversity | No single type > 40% of features |
| Feature Importance Spread | Top feature < 40% importance |

### 5.2 Phase 2 Filters (Multi-Symbol)

**Cross-Sectional:**

| Metric | Threshold |
|--------|-----------|
| Symbols Passing | ≥3 of 4 (ES/NQ/YM/RTY) with Sharpe ≥ 0.3 |
| Average Sharpe | ≥ 0.5 across all 4 symbols |
| No Disasters | No symbol with Sharpe < -0.2 |
| Drawdown Consistency | Max DD < 30% on all symbols |

**Portfolio-Level:**

| Metric | Threshold |
|--------|-----------|
| Max Correlation | < 0.7 with any existing strategy |
| Avg Correlation | < 0.5 with existing portfolio |
| Sharpe Improvement | Portfolio Sharpe +0.05 when added |
| Incremental Alpha | p-value < 0.05 in regression test |

### 5.3 Composite Ranking Score

For strategies that pass all filters, rank by composite score:

```python
def composite_score(strategy):
    """
    Weighted composite score for ranking strategies
    """
    weights = {
        'sharpe': 0.20,           # Raw Sharpe (0.2-1.0 scale)
        'consistency': 0.25,       # Walk-forward + regime consistency
        'stability': 0.15,         # Parameter stability
        'diversification': 0.20,   # Low correlation with existing
        'feature_richness': 0.10,  # Feature count + diversity
        'statistical_sig': 0.10    # Monte Carlo + FDR p-values
    }
    
    # Normalize each metric to 0-1 scale
    normalized = {
        'sharpe': normalize(strategy.sharpe, min=0.2, max=1.0),
        'consistency': (strategy.regime_score + strategy.walkforward_score) / 2,
        'stability': strategy.param_stability_score,
        'diversification': 1 - strategy.avg_correlation,
        'feature_richness': (strategy.feature_count / 50) * strategy.feature_diversity,
        'statistical_sig': 1 - strategy.fdr_pvalue
    }
    
    # Weighted sum
    score = sum(weights[k] * normalized[k] for k in weights)
    
    return score  # 0-1 scale, higher is better
```

**Interpretation:**
- Score 0.7-1.0: Excellent candidate (high priority for Phase 2)
- Score 0.5-0.7: Good candidate (consider for Phase 2)
- Score 0.3-0.5: Marginal (only if novel archetype)
- Score < 0.3: Reject (doesn't meet minimum bar)

---

## 6. Statistical Rigor Framework

### 6.1 The Multiple Testing Problem

**Challenge:**
```
With 5,000 backtests:
- Expected false positives: 5,000 × 0.05 = 250
- Many "winners" will be pure luck
- Need corrections to find real edges
```

**Solution: False Discovery Rate (FDR) Control**

```python
from statsmodels.stats.multitest import multipletests

def apply_fdr_correction(strategies, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction
    
    Controls expected proportion of false discoveries
    More powerful than Bonferroni for large-scale testing
    """
    # Get p-values from Monte Carlo tests
    p_values = [monte_carlo_test(s.returns) for s in strategies]
    
    # Apply FDR correction
    reject, corrected_p, _, _ = multipletests(
        p_values, 
        alpha=alpha, 
        method='fdr_bh'  # Benjamini-Hochberg
    )
    
    # Keep strategies with reject=True (significant after correction)
    significant_strategies = [
        s for s, r in zip(strategies, reject) if r
    ]
    
    return significant_strategies

# Example:
# Before FDR: 247 strategies with p < 0.05
# After FDR:   28 strategies with corrected p < 0.05  ← Real edges
```

### 6.2 Monte Carlo Permutation Test

**Concept:** Shuffle trade outcomes to destroy temporal structure, test if real Sharpe is unusual

```python
def monte_carlo_sharpe_test(returns, n_simulations=1000, alpha=0.05):
    """
    Test null hypothesis: Sharpe ratio is due to random chance
    
    Method:
    1. Calculate actual Sharpe ratio
    2. Shuffle returns randomly (destroy any skill-based patterns)
    3. Calculate Sharpe of shuffled returns
    4. Repeat 1000 times
    5. p-value = % of shuffled Sharpes ≥ actual Sharpe
    
    If p < 0.05: Reject null hypothesis (Sharpe is not random)
    """
    actual_sharpe = calculate_sharpe(returns)
    
    shuffled_sharpes = []
    for _ in range(n_simulations):
        shuffled_returns = np.random.permutation(returns)
        shuffled_sharpe = calculate_sharpe(shuffled_returns)
        shuffled_sharpes.append(shuffled_sharpe)
    
    # p-value: How often random shuffle beats actual?
    p_value = np.mean(np.array(shuffled_sharpes) >= actual_sharpe)
    
    return {
        'actual_sharpe': actual_sharpe,
        'mean_shuffled_sharpe': np.mean(shuffled_sharpes),
        'p_value': p_value,
        'significant': p_value < alpha
    }

# Example Result:
# Actual Sharpe: 0.42
# Mean Shuffled Sharpe: 0.03
# p-value: 0.007 ← Significant! (only 0.7% of random shuffles this good)
```

### 6.3 Walk-Forward Validation

**Concept:** Train/test split prevents overfitting to specific time periods

```python
def walk_forward_validation(strategy, data):
    """
    Standard approach in quantitative finance
    
    Simulates real-world deployment:
    - Optimize on training data (past)
    - Test on out-of-sample data (future)
    - Check for performance degradation
    """
    # Split data
    train_data = data['2010-01-01':'2021-12-31']  # 11 years
    test_data = data['2022-01-01':'2024-12-31']    # 3 years OOS
    
    # Backtest on each period
    train_results = backtest(strategy, train_data)
    test_results = backtest(strategy, test_data)
    
    # Metrics
    train_sharpe = train_results.sharpe_ratio
    test_sharpe = test_results.sharpe_ratio
    degradation = (train_sharpe - test_sharpe) / train_sharpe
    
    # Pass criteria
    criteria = {
        'train_sharpe': train_sharpe >= 0.3,
        'test_sharpe': test_sharpe >= 0.15,
        'degradation': degradation <= 0.5,  # Max 50% degradation
        'test_positive': test_sharpe > 0     # Must be profitable OOS
    }
    
    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'degradation_pct': degradation * 100,
        'pass': all(criteria.values()),
        'criteria': criteria
    }

# Example:
# Train Sharpe: 0.45
# Test Sharpe: 0.28
# Degradation: 38% ← Acceptable (< 50%)
# Status: PASS
```

### 6.4 Regime Analysis

**Concept:** Test consistency across different market conditions

```python
def regime_analysis(strategy, data):
    """
    Partition data into distinct market regimes
    Test strategy performance in each regime
    """
    # Define regimes
    regimes = {
        'bull': [
            ('2010-01-01', '2019-12-31'),  # Long bull run
            ('2023-01-01', '2024-12-31')   # Post-2022 recovery
        ],
        'bear': [
            ('2020-02-01', '2020-04-30'),  # COVID crash
            ('2022-01-01', '2022-10-31')   # Rate hike selloff
        ],
        'sideways': [
            ('2015-01-01', '2016-12-31')   # Range-bound
        ]
    }
    
    results = {}
    for regime_name, periods in regimes.items():
        regime_returns = []
        for start, end in periods:
            period_data = data[start:end]
            period_results = backtest(strategy, period_data)
            regime_returns.extend(period_results.returns)
        
        results[regime_name] = {
            'sharpe': calculate_sharpe(regime_returns),
            'profit_factor': calculate_profit_factor(regime_returns),
            'max_dd': calculate_max_drawdown(regime_returns)
        }
    
    # Pass criteria
    sharpes = [r['sharpe'] for r in results.values()]
    pass_criteria = {
        'positive_regimes': sum(s >= 0.2 for s in sharpes) >= 2,  # ≥2 of 3
        'no_disasters': min(sharpes) >= -0.3,  # No catastrophic regime
        'consistency': np.std(sharpes) < 0.5   # Not too volatile across regimes
    }
    
    return {
        'regime_sharpes': results,
        'pass': all(pass_criteria.values()),
        'criteria': pass_criteria
    }

# Example:
# Bull Sharpe: 0.38
# Bear Sharpe: 0.24
# Sideways Sharpe: 0.31
# Status: PASS (all ≥ 0.2, consistent)
```

### 6.5 Parameter Stability Analysis

**Concept:** Robust strategies shouldn't collapse from small parameter changes

```python
def parameter_stability_test(strategy, base_params, data):
    """
    Test each parameter with ±10% variations
    Measure Sharpe range relative to mean
    """
    results = {}
    
    for param_name, base_value in base_params.items():
        # Test variations
        variations = {
            'base': base_value,
            'minus_10': base_value * 0.9,
            'plus_10': base_value * 1.1
        }
        
        sharpes = {}
        for var_name, var_value in variations.items():
            # Update param
            test_params = base_params.copy()
            test_params[param_name] = var_value
            
            # Backtest
            strategy_var = strategy(**test_params)
            backtest_result = backtest(strategy_var, data)
            sharpes[var_name] = backtest_result.sharpe_ratio
        
        # Calculate stability metrics
        sharpe_values = list(sharpes.values())
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        sharpe_mean = np.mean(sharpe_values)
        stability_ratio = sharpe_range / sharpe_mean if sharpe_mean > 0 else np.inf
        
        results[param_name] = {
            'sharpes': sharpes,
            'range': sharpe_range,
            'mean': sharpe_mean,
            'stability_ratio': stability_ratio,
            'stable': stability_ratio < 0.4  # Pass if < 40% variation
        }
    
    # Overall stability
    overall_stable = all(r['stable'] for r in results.values())
    
    return {
        'params': results,
        'pass': overall_stable
    }

# Example for RSI Period:
# Base (14): Sharpe 0.35
# -10% (12.6): Sharpe 0.32
# +10% (15.4): Sharpe 0.38
# Range: 0.06
# Mean: 0.35
# Stability Ratio: 17% ← Excellent (< 40%)
```

---

## 7. Implementation Specifications

### 7.1 Core Scripts Architecture

```
research/
├── strategy_factory/
│   ├── __init__.py
│   ├── config.yaml                    # Configuration
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseResearchStrategy class
│   │   ├── mean_reversion/
│   │   │   ├── ibs.py
│   │   │   ├── rsi_divergence.py
│   │   │   ├── bollinger_reversal.py
│   │   │   └── ...
│   │   ├── momentum/
│   │   │   ├── trend_following.py
│   │   │   ├── breakout.py
│   │   │   └── ...
│   │   ├── volatility/
│   │   ├── cross_asset/
│   │   ├── seasonality/
│   │   ├── microstructure/
│   │   ├── multi_timeframe/
│   │   └── regime_adaptive/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── optimizer.py               # Main optimization engine
│   │   ├── backtester.py              # Backtest execution
│   │   ├── filters.py                 # Filter implementations
│   │   ├── statistics.py              # Statistical tests
│   │   └── portfolio.py               # Portfolio simulations
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql                 # SQLite schema
│   │   └── manager.py                 # Database operations
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── generator.py               # Report generation
│   │   ├── visualizations.py          # Charts and plots
│   │   └── templates/
│   │       ├── phase1_report.md.j2
│   │       ├── phase2_report.md.j2
│   │       └── phase3_report.md.j2
│   ├── integration/
│   │   ├── __init__.py
│   │   └── ml_pipeline.py             # Phase 3 ML integration
│   ├── main.py                         # CLI entry point
│   └── README.md                       # Usage documentation
├── results/
│   ├── strategy_factory.db            # Master results database
│   └── runs/
│       ├── 20250120_143022/           # Individual run outputs
│       │   ├── config.yaml
│       │   ├── strategies.py
│       │   ├── phase1_results.db
│       │   ├── phase1_report.md
│       │   ├── phase2_results.db
│       │   ├── phase2_report.md
│       │   └── charts/
│       └── ...
└── notebooks/
    └── strategy_analysis.ipynb
```

### 7.2 BaseResearchStrategy Class

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class BaseResearchStrategy(ABC):
    """
    Base class for all research strategies
    
    Each strategy must implement:
    - entry_logic(): When to enter trades
    - exit_logic(): When to exit trades
    - collect_features(): Features for ML training
    - param_grid: Dictionary of parameters to optimize
    """
    
    def __init__(self, name: str, archetype: str):
        self.name = name
        self.archetype = archetype
        self.params = {}
        self.data = None
        self.features = None
    
    @abstractmethod
    def entry_logic(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """
        Returns boolean Series: True where entry signal occurs
        
        Args:
            data: OHLCV dataframe with indicators pre-calculated
            params: Strategy parameters from param_grid
            
        Returns:
            Boolean Series indicating entry signals
        """
        pass
    
    @abstractmethod
    def exit_logic(self, data: pd.DataFrame, params: Dict, 
                   entry_price: float, entry_bar: int) -> bool:
        """
        Returns True when exit condition met
        
        Args:
            data: OHLCV dataframe
            params: Strategy parameters
            entry_price: Price at entry
            entry_bar: Index of entry bar
            
        Returns:
            Boolean indicating if exit should occur
        """
        pass
    
    @abstractmethod
    def collect_features(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate features for ML training
        
        Must generate ≥20 diverse features including:
        - Price-based (momentum, reversals)
        - Volume-based (surge, climax)
        - Cross-asset (correlations, divergences)
        - Technical indicators (RSI, MACD, Bollinger)
        
        Args:
            data: OHLCV dataframe
            params: Strategy parameters
            
        Returns:
            DataFrame with feature columns
        """
        pass
    
    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, List]:
        """
        Parameter grid for optimization
        
        Returns:
            Dictionary mapping param names to lists of values
            Example: {
                'rsi_period': [10, 14, 20, 30],
                'threshold': [20, 25, 30]
            }
        """
        pass
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality for ML readiness
        """
        feature_count = len(features.columns)
        
        # Feature type diversity
        feature_types = {
            'price': sum('price' in col.lower() or 'close' in col.lower() 
                        for col in features.columns),
            'volume': sum('volume' in col.lower() or 'vol' in col.lower() 
                         for col in features.columns),
            'cross_asset': sum('correlation' in col.lower() or 'divergence' in col.lower() 
                              or 'vix' in col.lower() or 'tlt' in col.lower()
                              for col in features.columns),
            'technical': sum('rsi' in col.lower() or 'macd' in col.lower() 
                            or 'bb' in col.lower() or 'sma' in col.lower()
                            for col in features.columns)
        }
        
        diversity_score = 1 - max(feature_types.values()) / feature_count
        
        return {
            'count': feature_count,
            'types': feature_types,
            'diversity_score': diversity_score,
            'passes': feature_count >= 20 and diversity_score >= 0.6
        }
```

### 7.3 Example Strategy Implementation

```python
from .base import BaseResearchStrategy
import pandas as pd
import numpy as np

class RSI_DivergenceStrategy(BaseResearchStrategy):
    """
    Mean Reversion: RSI Divergence
    
    Hypothesis: When price makes lower low but RSI makes higher low,
    indicates oversold conditions and potential reversal.
    
    Entry: 
    - Price makes lower low vs N bars ago
    - RSI makes higher low vs N bars ago
    - RSI < oversold threshold
    
    Exit:
    - RSI > exit threshold
    - OR max hold bars elapsed
    - OR stop loss / profit target hit
    """
    
    def __init__(self):
        super().__init__(
            name='RSI_Divergence',
            archetype='mean_reversion'
        )
    
    def entry_logic(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Entry when RSI/price divergence detected"""
        rsi_period = params['rsi_period']
        lookback = params['divergence_lookback']
        threshold = params['oversold_threshold']
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], rsi_period)
        
        # Price lower low
        price_low = data['low']
        price_lower_low = price_low < price_low.shift(lookback)
        
        # RSI higher low (divergence)
        rsi_higher_low = rsi > rsi.shift(lookback)
        
        # RSI oversold
        rsi_oversold = rsi < threshold
        
        # Entry signal: All conditions met
        entry = price_lower_low & rsi_higher_low & rsi_oversold
        
        return entry
    
    def exit_logic(self, data: pd.DataFrame, params: Dict,
                   entry_price: float, entry_bar: int) -> bool:
        """Exit when RSI crosses above threshold or max hold"""
        current_bar = len(data) - 1
        bars_held = current_bar - entry_bar
        
        # RSI exit
        rsi = self._calculate_rsi(data['close'], params['rsi_period'])
        rsi_exit = rsi.iloc[-1] > params['exit_threshold']
        
        # Time-based exit
        time_exit = bars_held >= params['max_hold_bars']
        
        return rsi_exit or time_exit
    
    def collect_features(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Generate 47 features for ML"""
        features = pd.DataFrame(index=data.index)
        
        # RSI features (8)
        for period in [9, 14, 21, 28]:
            rsi = self._calculate_rsi(data['close'], period)
            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_zscore'] = (rsi - rsi.rolling(100).mean()) / rsi.rolling(100).std()
        
        # Price momentum features (6)
        for period in [5, 10, 20]:
            features[f'price_roc_{period}'] = data['close'].pct_change(period)
            features[f'price_zscore_{period}'] = (
                (data['close'] - data['close'].rolling(period).mean()) / 
                data['close'].rolling(period).std()
            )
        
        # Volume features (4)
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_zscore'] = (
            (data['volume'] - data['volume'].rolling(50).mean()) / 
            data['volume'].rolling(50).std()
        )
        features['volume_surge'] = (data['volume'] > data['volume'].rolling(20).mean() * 1.5).astype(int)
        features['volume_climax'] = (data['volume'] > data['volume'].rolling(100).quantile(0.95)).astype(int)
        
        # Volatility features (5)
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
        features['atr_20'] = self._calculate_atr(data, 20)
        features['atr_percentile'] = features['atr_20'].rolling(252).rank(pct=True)
        
        # Cross-asset features (8)
        # Assuming VIX, TLT data available
        if 'vix_close' in data.columns:
            features['vix_level'] = data['vix_close']
            features['vix_percentile'] = data['vix_close'].rolling(252).rank(pct=True)
            features['vix_roc_5'] = data['vix_close'].pct_change(5)
            features['vix_spike'] = (data['vix_close'] > data['vix_close'].rolling(20).mean() * 1.3).astype(int)
        
        if 'tlt_close' in data.columns:
            features['tlt_roc_5'] = data['tlt_close'].pct_change(5)
            features['tlt_es_correlation_30'] = (
                data['close'].pct_change().rolling(30).corr(data['tlt_close'].pct_change())
            )
            features['tlt_es_divergence'] = (
                (data['close'].pct_change(5) > 0) & 
                (data['tlt_close'].pct_change(5) < 0)
            ).astype(int)
        
        # Technical indicators (9)
        features['bb_position'] = self._bollinger_position(data['close'], 20, 2)
        features['macd'] = self._calculate_macd(data['close'])
        features['macd_signal'] = self._calculate_macd_signal(data['close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        for period in [20, 50, 100]:
            features[f'sma_{period}_distance'] = (data['close'] - data['close'].rolling(period).mean()) / data['close']
        
        # Time/regime features (7)
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        features['is_month_end'] = (data.index.is_month_end).astype(int)
        features['is_quarter_end'] = (data.index.is_quarter_end).astype(int)
        features['vol_regime'] = (
            data['close'].pct_change().rolling(50).std() > 
            data['close'].pct_change().rolling(252).std()
        ).astype(int)
        
        # Divergence features (specific to this strategy) (5)
        for lookback in [5, 10, 15]:
            price_direction = (data['close'] - data['close'].shift(lookback)).apply(np.sign)
            rsi = self._calculate_rsi(data['close'], 14)
            rsi_direction = (rsi - rsi.shift(lookback)).apply(np.sign)
            features[f'divergence_{lookback}'] = (price_direction != rsi_direction).astype(int)
        
        features['divergence_magnitude_10'] = abs(
            (data['close'].pct_change(10)) - 
            (self._calculate_rsi(data['close'], 14).pct_change(10))
        )
        features['oversold_duration'] = (
            self._calculate_rsi(data['close'], 14) < 30
        ).rolling(10).sum()
        
        return features.dropna()
    
    @property
    def param_grid(self) -> Dict[str, List]:
        """Parameter combinations to test"""
        return {
            'rsi_period': [10, 14, 20, 30],
            'divergence_lookback': [5, 10, 15],
            'oversold_threshold': [20, 25, 30],
            'exit_threshold': [50, 60, 70],
            'max_hold_bars': [5, 10, 20]
        }
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _bollinger_position(self, prices: pd.Series, period: int, std: float) -> pd.Series:
        sma = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return (prices - lower) / (upper - lower)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2
    
    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        macd = self._calculate_macd(prices)
        return macd.ewm(span=9).mean()
```

### 7.4 Configuration File (config.yaml)

```yaml
# Strategy Factory Configuration

# Date ranges
date_ranges:
  full:
    start: "2010-01-01"
    end: "2024-12-31"
  train:
    start: "2010-01-01"
    end: "2021-12-31"
  test:
    start: "2022-01-01"
    end: "2024-12-31"

# Regime definitions
regimes:
  bull:
    - ["2010-01-01", "2019-12-31"]
    - ["2023-01-01", "2024-12-31"]
  bear:
    - ["2020-02-01", "2020-04-30"]
    - ["2022-01-01", "2022-10-31"]
  sideways:
    - ["2015-01-01", "2016-12-31"]

# Symbols
symbols:
  phase1: ["ES"]
  phase2: ["ES", "NQ", "YM", "RTY"]
  phase3: ["ES", "NQ", "YM", "RTY"]  # ML on all if Phase 2 passes

# Phase 1 filters
phase1_filters:
  critical:
    min_trades: 10000
    min_sharpe: 0.2
    min_profit_factor: 1.15
    max_drawdown: 0.30
    min_win_rate: 0.35
  
  robustness:
    walkforward_min_test_sharpe: 0.15
    walkforward_min_ratio: 0.5  # test_sharpe / train_sharpe
    regime_min_passing: 2  # of 3 regimes
    regime_min_sharpe: 0.2
    regime_max_negative: -0.3
    param_stability_threshold: 0.4  # max variation ratio
  
  statistical:
    monte_carlo_simulations: 1000
    monte_carlo_alpha: 0.05
    fdr_alpha: 0.05
  
  features:
    min_count: 20
    min_diversity: 0.6  # No single type > 40%

# Phase 2 filters
phase2_filters:
  cross_sectional:
    min_symbols_passing: 3  # of 4
    min_symbol_sharpe: 0.3
    min_avg_sharpe: 0.5
    max_negative_sharpe: -0.2
    max_drawdown: 0.30
  
  portfolio:
    max_correlation_single: 0.7
    max_correlation_avg: 0.5
    min_sharpe_improvement: 0.05
    min_diversification_ratio: 1.1
  
  incremental_alpha:
    regression_alpha: 0.05
    min_alpha_value: 0.0

# Phase 3 integration
phase3_ml:
  extract_script: "research/extract_training_data.py"
  train_script: "research/train_rf_cpcv_bo.py"
  backtest_script: "research/backtest_runner.py"
  
  train_params:
    n_trials: 100
    cv_splits: 5
    threshold: 0.50
  
  success_criteria:
    min_ml_sharpe: 1.0
    min_improvement_ratio: 2.0  # ML Sharpe / Raw Sharpe
    min_trade_efficiency: 0.25  # ML trades / Raw trades
    max_drawdown: 0.20

# Execution settings
execution:
  max_workers: 16
  chunk_size: 50
  timeout_per_backtest: 300  # seconds
  memory_limit_gb: 8

# Output settings
output:
  database: "results/strategy_factory.db"
  run_directory: "results/runs"
  charts_directory: "charts"
  log_level: "INFO"
  
  reports:
    generate_phase1: true
    generate_phase2: true
    generate_phase3: true
    include_charts: true
    include_trade_logs: true

# Experiment tracking
tracking:
  git_auto_commit: true
  git_auto_tag: true
  meta_learning_db: "results/meta_learning.db"
  
# Strategy selection
strategies:
  # Which archetypes to include in run
  archetypes:
    - "mean_reversion"
    - "momentum"
    - "volatility"
    - "cross_asset"
    - "seasonality"
    # - "microstructure"  # Uncomment when ready
    # - "multi_timeframe"
    # - "regime_adaptive"
  
  # Specific strategies to exclude (if needed)
  exclude:
    # - "RSI_Divergence"
```

### 7.5 Database Schema

```sql
-- strategy_factory.db schema

-- Main strategies table
CREATE TABLE strategies (
    strategy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    archetype TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name)
);

-- Run metadata
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    config_snapshot TEXT,  -- JSON of config used
    git_commit_hash TEXT,
    phase_completed INTEGER,  -- 1, 2, or 3
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtest results (Phase 1)
CREATE TABLE backtest_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    params TEXT NOT NULL,  -- JSON of parameters
    
    -- Performance metrics
    sharpe_ratio REAL,
    profit_factor REAL,
    max_drawdown REAL,
    win_rate REAL,
    trade_count INTEGER,
    total_return REAL,
    
    -- Date ranges
    start_date TEXT,
    end_date TEXT,
    
    -- Filters passed
    passed_critical_filters BOOLEAN,
    passed_robustness_filters BOOLEAN,
    passed_statistical_filters BOOLEAN,
    passed_feature_filters BOOLEAN,
    
    -- Statistical test results
    monte_carlo_pvalue REAL,
    fdr_corrected_pvalue REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- Walk-forward results
CREATE TABLE walkforward_results (
    wf_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    
    train_sharpe REAL,
    train_start TEXT,
    train_end TEXT,
    
    test_sharpe REAL,
    test_start TEXT,
    test_end TEXT,
    
    degradation_ratio REAL,
    passed BOOLEAN,
    
    FOREIGN KEY (result_id) REFERENCES backtest_results(result_id)
);

-- Regime analysis results
CREATE TABLE regime_results (
    regime_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    
    regime_name TEXT,  -- bull, bear, sideways
    sharpe_ratio REAL,
    profit_factor REAL,
    max_drawdown REAL,
    trade_count INTEGER,
    
    FOREIGN KEY (result_id) REFERENCES backtest_results(result_id)
);

-- Parameter stability results
CREATE TABLE param_stability (
    stability_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    
    param_name TEXT,
    base_value REAL,
    base_sharpe REAL,
    minus_10_sharpe REAL,
    plus_10_sharpe REAL,
    stability_ratio REAL,
    passed BOOLEAN,
    
    FOREIGN KEY (result_id) REFERENCES backtest_results(result_id)
);

-- Phase 2 multi-symbol results
CREATE TABLE multisymbol_results (
    multisymbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy_id INTEGER NOT NULL,
    params TEXT NOT NULL,
    
    -- Per-symbol results
    es_sharpe REAL,
    nq_sharpe REAL,
    ym_sharpe REAL,
    rty_sharpe REAL,
    
    avg_sharpe REAL,
    symbols_passing INTEGER,
    
    -- Correlation analysis
    avg_correlation REAL,
    max_correlation REAL,
    correlation_matrix TEXT,  -- JSON
    
    -- Portfolio metrics
    portfolio_sharpe_improvement REAL,
    diversification_ratio REAL,
    incremental_alpha REAL,
    incremental_alpha_pvalue REAL,
    
    passed_phase2 BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- Phase 3 ML integration results
CREATE TABLE ml_results (
    ml_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    params TEXT NOT NULL,
    
    -- Raw vs ML comparison
    raw_sharpe REAL,
    ml_sharpe REAL,
    improvement_ratio REAL,
    
    raw_trades INTEGER,
    ml_trades INTEGER,
    trade_efficiency REAL,
    
    raw_profit_factor REAL,
    ml_profit_factor REAL,
    
    raw_max_dd REAL,
    ml_max_dd REAL,
    
    -- ML model metrics
    ml_precision REAL,
    ml_recall REAL,
    ml_f1_score REAL,
    
    -- Feature importance
    top_10_features TEXT,  -- JSON
    
    passed_phase3 BOOLEAN,
    recommended_for_production BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- Meta-learning feedback
CREATE TABLE meta_learning (
    meta_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    archetype TEXT NOT NULL,
    
    -- Raw metrics
    raw_sharpe REAL,
    feature_count INTEGER,
    feature_diversity REAL,
    regime_consistency REAL,
    param_stability REAL,
    cross_symbol_avg REAL,
    
    -- ML enhancement
    ml_sharpe REAL,
    ml_boost_ratio REAL,
    
    -- Timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- Create indexes for performance
CREATE INDEX idx_backtest_strategy ON backtest_results(strategy_id);
CREATE INDEX idx_backtest_run ON backtest_results(run_id);
CREATE INDEX idx_backtest_symbol ON backtest_results(symbol);
CREATE INDEX idx_multisymbol_strategy ON multisymbol_results(strategy_id);
CREATE INDEX idx_ml_strategy ON ml_results(strategy_id);
CREATE INDEX idx_meta_archetype ON meta_learning(archetype);
```

---

## 8. Integration with ML Pipeline

### 8.1 Handoff Process

**Phase 2 → Phase 3 Transition:**

```python
def prepare_ml_integration(phase2_winners: List[Dict]) -> List[Dict]:
    """
    Prepare strategies for ML pipeline
    
    For each Phase 2 winner:
    1. Extract optimal parameters
    2. Generate feature extraction config
    3. Set up ML training config
    4. Create backtest validation config
    """
    ml_configs = []
    
    for strategy in phase2_winners:
        config = {
            'strategy_name': strategy['name'],
            'archetype': strategy['archetype'],
            'params': strategy['best_params'],
            'symbols': ['ES', 'NQ', 'YM', 'RTY'],
            
            'extraction': {
                'script': 'research/extract_training_data.py',
                'date_range': ('2010-01-01', '2024-12-31'),
                'features': strategy['feature_list'],  # 20-50 features
                'expected_trades': strategy['trade_count']  # Should be 10k+
            },
            
            'training': {
                'script': 'research/train_rf_cpcv_bo.py',
                'n_trials': 100,
                'cv_splits': 5,
                'embargo_days': 1,
                'threshold': 0.50,
                'recency_weight': True
            },
            
            'validation': {
                'script': 'research/backtest_runner.py',
                'test_period': ('2022-01-01', '2024-12-31'),
                'expected_improvement': 2.0,  # Min 2x Sharpe improvement
                'min_ml_sharpe': 1.0
            }
        }
        
        ml_configs.append(config)
    
    return ml_configs
```

### 8.2 Automated Execution

```python
import subprocess
import json
from pathlib import Path

def execute_ml_pipeline(config: Dict) -> Dict:
    """
    Execute full ML pipeline for a strategy
    """
    results = {}
    strategy_name = config['strategy_name']
    
    # Step 1: Feature Extraction (Parallel across symbols)
    print(f"\n[1/3] Extracting features for {strategy_name}...")
    extraction_results = []
    
    for symbol in config['symbols']:
        cmd = [
            'python', config['extraction']['script'],
            '--strategy', strategy_name,
            '--symbol', symbol,
            '--start', config['extraction']['date_range'][0],
            '--end', config['extraction']['date_range'][1],
            '--params', json.dumps(config['params'])
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            extraction_results.append({
                'symbol': symbol,
                'output_file': f"extracted_data/{symbol}_{strategy_name}_features.csv",
                'status': 'success'
            })
        else:
            extraction_results.append({
                'symbol': symbol,
                'status': 'failed',
                'error': result.stderr
            })
    
    results['extraction'] = extraction_results
    
    # Step 2: ML Training (Sequential per symbol)
    print(f"\n[2/3] Training ML models for {strategy_name}...")
    training_results = []
    
    for symbol in config['symbols']:
        cmd = [
            'python', config['training']['script'],
            '--symbol', symbol,
            '--strategy', strategy_name,
            '--n_trials', str(config['training']['n_trials']),
            '--cv_splits', str(config['training']['cv_splits']),
            '--threshold', str(config['training']['threshold'])
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            training_results.append({
                'symbol': symbol,
                'model_file': f"models/{symbol}_{strategy_name}_rf_model.pkl",
                'status': 'success'
            })
        else:
            training_results.append({
                'symbol': symbol,
                'status': 'failed',
                'error': result.stderr
            })
    
    results['training'] = training_results
    
    # Step 3: Validation Backtesting
    print(f"\n[3/3] Validating ML-filtered strategy {strategy_name}...")
    validation_results = []
    
    for symbol in config['symbols']:
        # Backtest with ML filter
        cmd_ml = [
            'python', config['validation']['script'],
            '--strategy', strategy_name,
            '--symbol', symbol,
            '--ml_model', f"models/{symbol}_{strategy_name}_rf_model.pkl",
            '--start', config['validation']['test_period'][0],
            '--end', config['validation']['test_period'][1]
        ]
        
        result_ml = subprocess.run(cmd_ml, capture_output=True, text=True)
        
        # Backtest without ML (raw)
        cmd_raw = [
            'python', config['validation']['script'],
            '--strategy', strategy_name,
            '--symbol', symbol,
            '--start', config['validation']['test_period'][0],
            '--end', config['validation']['test_period'][1]
        ]
        
        result_raw = subprocess.run(cmd_raw, capture_output=True, text=True)
        
        if result_ml.returncode == 0 and result_raw.returncode == 0:
            # Parse results (assuming JSON output)
            ml_metrics = json.loads(result_ml.stdout)
            raw_metrics = json.loads(result_raw.stdout)
            
            improvement_ratio = ml_metrics['sharpe'] / raw_metrics['sharpe']
            
            validation_results.append({
                'symbol': symbol,
                'raw_sharpe': raw_metrics['sharpe'],
                'ml_sharpe': ml_metrics['sharpe'],
                'improvement': improvement_ratio,
                'meets_criteria': (
                    ml_metrics['sharpe'] >= config['validation']['min_ml_sharpe'] and
                    improvement_ratio >= config['validation']['expected_improvement']
                ),
                'status': 'success'
            })
        else:
            validation_results.append({
                'symbol': symbol,
                'status': 'failed',
                'error': result_ml.stderr or result_raw.stderr
            })
    
    results['validation'] = validation_results
    
    # Overall success
    results['overall_success'] = all(
        v['status'] == 'success' and v.get('meets_criteria', False)
        for v in validation_results
    )
    
    return results

def run_ml_pipeline_for_all_winners(ml_configs: List[Dict]):
    """
    Execute ML pipeline for all Phase 2 winners
    """
    all_results = []
    
    for config in ml_configs:
        print(f"\n{'='*80}")
        print(f"Processing: {config['strategy_name']}")
        print(f"{'='*80}")
        
        try:
            results = execute_ml_pipeline(config)
            all_results.append({
                'strategy': config['strategy_name'],
                'results': results
            })
            
            # Log to database
            log_ml_results_to_db(config, results)
            
        except Exception as e:
            print(f"ERROR: Failed to process {config['strategy_name']}: {e}")
            all_results.append({
                'strategy': config['strategy_name'],
                'error': str(e)
            })
    
    # Generate final report
    generate_phase3_report(all_results)
    
    return all_results
```

### 8.3 Performance Comparison Framework

```python
def compare_raw_vs_ml(strategy_name: str, symbol: str,
                     raw_results: Dict, ml_results: Dict) -> Dict:
    """
    Detailed comparison of raw strategy vs ML-filtered
    """
    comparison = {
        'strategy': strategy_name,
        'symbol': symbol,
        
        # Sharpe improvement
        'raw_sharpe': raw_results['sharpe_ratio'],
        'ml_sharpe': ml_results['sharpe_ratio'],
        'sharpe_improvement': ml_results['sharpe_ratio'] / raw_results['sharpe_ratio'],
        
        # Trade efficiency
        'raw_trades': raw_results['trade_count'],
        'ml_trades': ml_results['trade_count'],
        'trades_kept_pct': ml_results['trade_count'] / raw_results['trade_count'] * 100,
        
        # Profitability improvement
        'raw_profit_factor': raw_results['profit_factor'],
        'ml_profit_factor': ml_results['profit_factor'],
        'pf_improvement': ml_results['profit_factor'] / raw_results['profit_factor'],
        
        # Risk improvement
        'raw_max_dd': raw_results['max_drawdown'],
        'ml_max_dd': ml_results['max_drawdown'],
        'dd_reduction': (raw_results['max_drawdown'] - ml_results['max_drawdown']) / raw_results['max_drawdown'] * 100,
        
        # Win rate improvement
        'raw_win_rate': raw_results['win_rate'],
        'ml_win_rate': ml_results['win_rate'],
        'win_rate_improvement': ml_results['win_rate'] - raw_results['win_rate'],
        
        # ML model metrics
        'ml_precision': ml_results['precision'],
        'ml_recall': ml_results['recall'],
        'ml_f1_score': ml_results['f1_score'],
        
        # Decision metrics
        'passes_criteria': (
            ml_results['sharpe_ratio'] >= 1.0 and
            ml_results['sharpe_ratio'] / raw_results['sharpe_ratio'] >= 2.0 and
            ml_results['trade_count'] / raw_results['trade_count'] >= 0.25
        ),
        
        'recommended_for_production': None  # Set based on additional criteria
    }
    
    # Recommendation logic
    if comparison['passes_criteria']:
        if comparison['sharpe_improvement'] >= 3.0:
            comparison['recommended_for_production'] = 'STRONG_YES'
        elif comparison['sharpe_improvement'] >= 2.5:
            comparison['recommended_for_production'] = 'YES'
        else:
            comparison['recommended_for_production'] = 'MAYBE'
    else:
        comparison['recommended_for_production'] = 'NO'
    
    return comparison
```

---

## 9. Experiment Tracking & Reproducibility

### 9.1 Git Integration

```python
import git
from datetime import datetime

def create_research_commit(run_timestamp: str, summary: Dict):
    """
    Auto-commit research run with detailed message
    """
    repo = git.Repo('/home/user/rooney-capital-v1')
    
    # Stage files
    repo.index.add([
        'research/strategy_factory/config.yaml',
        'research/strategy_factory/strategies/',
        f'research/results/runs/{run_timestamp}/'
    ])
    
    # Create detailed commit message
    message = f"""Strategy Factory Run: {run_timestamp}

Phase Completed: {summary['phase_completed']}
Strategies Tested: {summary['strategies_tested']}
Total Backtests: {summary['total_backtests']}

Phase 1 Results:
- Passed Critical Filters: {summary['phase1_passed_critical']}
- Passed All Filters: {summary['phase1_passed_all']}

"""
    
    if summary['phase_completed'] >= 2:
        message += f"""Phase 2 Results:
- Multi-Symbol Validation: {summary['phase2_passed']}
- Recommended for ML: {summary['phase2_recommended']}

"""
    
    if summary['phase_completed'] >= 3:
        message += f"""Phase 3 Results:
- ML Pipeline Complete: {summary['phase3_complete']}
- Production Ready: {summary['phase3_production_ready']}

"""
    
    # Commit
    repo.index.commit(message)
    
    # Create tag
    tag_name = f"research-{run_timestamp}"
    repo.create_tag(tag_name, message=f"Research run {run_timestamp}")
    
    print(f"✅ Git commit created: {repo.head.commit.hexsha[:8]}")
    print(f"✅ Git tag created: {tag_name}")
    
    return repo.head.commit.hexsha
```

### 9.2 Experiment Metadata

```python
def capture_experiment_metadata() -> Dict:
    """
    Capture complete experiment environment for reproducibility
    """
    import platform
    import sys
    import git
    
    repo = git.Repo('/home/user/rooney-capital-v1')
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        
        # Code version
        'git_commit': repo.head.commit.hexsha,
        'git_branch': repo.active_branch.name,
        'git_dirty': repo.is_dirty(),
        
        # Environment
        'python_version': sys.version,
        'platform': platform.platform(),
        'hostname': platform.node(),
        
        # Dependencies
        'key_packages': {
            'backtrader': get_package_version('backtrader'),
            'pandas': get_package_version('pandas'),
            'numpy': get_package_version('numpy'),
            'scikit-learn': get_package_version('scikit-learn'),
            'optuna': get_package_version('optuna'),
        },
        
        # Configuration
        'config_hash': hash_config_file('config.yaml'),
        
        # Data sources
        'databento_version': get_databento_schema_version(),
        'data_last_updated': get_data_last_update_timestamp()
    }
    
    return metadata

def save_experiment_snapshot(run_dir: Path, metadata: Dict, config: Dict):
    """
    Save complete experiment snapshot
    """
    snapshot = {
        'metadata': metadata,
        'config': config,
        'strategies': capture_strategy_code_snapshot()
    }
    
    with open(run_dir / 'experiment_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"✅ Experiment snapshot saved: {run_dir / 'experiment_snapshot.json'}")
```

### 9.3 Research Journal Auto-Generation

```python
from jinja2 import Template
from pathlib import Path

def generate_research_journal(run_timestamp: str, results: Dict):
    """
    Auto-generate markdown research journal
    """
    template = Template(Path('research/strategy_factory/reporting/templates/phase1_report.md.j2').read_text())
    
    # Prepare data for template
    data = {
        'run_timestamp': run_timestamp,
        'git_commit': results['metadata']['git_commit'][:8],
        
        # Summary stats
        'total_strategies': results['stats']['total_strategies'],
        'total_backtests': results['stats']['total_backtests'],
        'execution_time': results['stats']['execution_time_minutes'],
        
        # Phase 1 results
        'phase1': {
            'passed_critical': results['phase1']['passed_critical'],
            'passed_all': results['phase1']['passed_all'],
            'top_performers': results['phase1']['top_10_strategies']
        },
        
        # Phase 2 results (if available)
        'phase2': results.get('phase2', None),
        
        # Phase 3 results (if available)
        'phase3': results.get('phase3', None),
        
        # Charts
        'charts': {
            'equity_curves': f"charts/equity_curves.png",
            'sharpe_distribution': f"charts/sharpe_distribution.png",
            'correlation_matrix': f"charts/correlation_matrix.png"
        }
    }
    
    # Render journal
    journal = template.render(**data)
    
    # Save
    journal_path = Path(f'research/results/runs/{run_timestamp}/research_journal.md')
    journal_path.write_text(journal)
    
    print(f"✅ Research journal generated: {journal_path}")
    
    return journal_path
```

**Example Research Journal Template:**

```markdown
# Strategy Factory Research Journal
**Run ID:** {{ run_timestamp }}  
**Git Commit:** {{ git_commit }}  
**Date:** {{ run_timestamp[:10] }}

---

## Executive Summary

- **Strategies Tested:** {{ total_strategies }}
- **Total Backtests:** {{ total_backtests }}
- **Execution Time:** {{ execution_time }} minutes
- **Phase Completed:** {{ 'Phase 3 (ML Integration)' if phase3 else 'Phase 2 (Multi-Symbol)' if phase2 else 'Phase 1 (Screening)' }}

---

## Phase 1: Strategy Screening (ES, 2010-2024)

### Filter Results

| Filter Stage | Strategies Passed |
|--------------|-------------------|
| Critical Filters | {{ phase1.passed_critical }} |
| Robustness Tests | {{ phase1.passed_robustness }} |
| Statistical Tests | {{ phase1.passed_statistical }} |
| **Final (All Filters)** | **{{ phase1.passed_all }}** |

### Top 10 Performers

{% for strategy in phase1.top_performers %}
#### {{ loop.index }}. {{ strategy.name }}

- **Archetype:** {{ strategy.archetype }}
- **Composite Score:** {{ strategy.composite_score | round(3) }}

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {{ strategy.sharpe | round(2) }} |
| Profit Factor | {{ strategy.profit_factor | round(2) }} |
| Max Drawdown | {{ (strategy.max_dd * 100) | round(1) }}% |
| Trade Count | {{ strategy.trades | number_format }} |
| Win Rate | {{ (strategy.win_rate * 100) | round(1) }}% |

**Robustness:**
- Walk-Forward: Train {{ strategy.wf_train_sharpe | round(2) }} / Test {{ strategy.wf_test_sharpe | round(2) }}
- Regime Consistency: Bull {{ strategy.regime_bull | round(2) }} / Bear {{ strategy.regime_bear | round(2) }} / Sideways {{ strategy.regime_sideways | round(2) }}
- Parameter Stability: {{ (strategy.param_stability * 100) | round(0) }}% stable

**Statistical Significance:**
- Monte Carlo p-value: {{ strategy.monte_carlo_p | round(4) }}
- FDR-corrected p-value: {{ strategy.fdr_p | round(4) }}

**Feature Quality:**
- Feature Count: {{ strategy.feature_count }}
- Feature Diversity: {{ (strategy.feature_diversity * 100) | round(0) }}%

![Equity Curve](charts/{{ strategy.name }}_equity.png)
![Parameter Heatmap](charts/{{ strategy.name }}_params.png)

---

{% endfor %}

{% if phase2 %}
## Phase 2: Multi-Symbol Validation

### Cross-Sectional Results

{{ phase2.summary_table }}

### Correlation Analysis

**Average Correlation with Existing Portfolio:** {{ phase2.avg_correlation | round(2) }}

![Correlation Matrix](charts/correlation_matrix.png)

### Strategies Advancing to ML Pipeline

{% for strategy in phase2.recommended %}
- **{{ strategy.name }}:** {{ strategy.reason }}
{% endfor %}

---
{% endif %}

{% if phase3 %}
## Phase 3: ML Integration Results

### Raw vs ML-Filtered Performance

{{ phase3.comparison_table }}

### Production Recommendations

{% for strategy in phase3.production_ready %}
#### {{ strategy.name }}

**Recommendation:** {{ strategy.recommendation }}

**Key Metrics:**
- ML Sharpe Improvement: {{ strategy.ml_sharpe }} / {{ strategy.raw_sharpe }} = {{ strategy.improvement_ratio | round(2) }}x
- Trade Efficiency: {{ (strategy.trades_kept_pct) | round(0) }}% of trades kept
- Drawdown Reduction: {{ strategy.dd_reduction | round(0) }}%

**ML Model Performance:**
- Precision: {{ (strategy.ml_precision * 100) | round(1) }}%
- Recall: {{ (strategy.ml_recall * 100) | round(1) }}%
- F1 Score: {{ (strategy.ml_f1 * 100) | round(1) }}%

**Top 5 Features:**
{% for feature in strategy.top_features[:5] %}
{{ loop.index }}. {{ feature.name }} ({{ (feature.importance * 100) | round(1) }}%)
{% endfor %}

---
{% endfor %}
{% endif %}

## Next Steps

{% if not phase2 %}
- [ ] Review Phase 1 results and select strategies for Phase 2
- [ ] Run `strategy_factory.py --phase 2 --strategies SELECTED_LIST`
{% elif not phase3 %}
- [ ] Review Phase 2 results and confirm strategies for ML pipeline
- [ ] Run `strategy_factory.py --phase 3 --strategies SELECTED_LIST`
{% else %}
- [ ] Review ML integration results
- [ ] For STRONG_YES recommendations: Deploy to paper trading
- [ ] For YES/MAYBE recommendations: Additional validation
- [ ] Update meta-learning database for next research cycle
{% endif %}

---

*Generated automatically by Strategy Factory on {{ run_timestamp }}*
```

---

## 10. Feedback Loop System

### 10.1 Meta-Learning Database

Track which strategy characteristics predict ML success:

```python
def update_meta_learning(strategy_results: Dict):
    """
    After Phase 3, capture learnings for future research
    """
    meta_entry = {
        'strategy_name': strategy_results['name'],
        'archetype': strategy_results['archetype'],
        
        # Input characteristics (Phase 1)
        'raw_sharpe': strategy_results['phase1']['sharpe'],
        'trade_count': strategy_results['phase1']['trades'],
        'feature_count': strategy_results['phase1']['feature_count'],
        'feature_diversity': strategy_results['phase1']['feature_diversity'],
        'regime_consistency': strategy_results['phase1']['regime_score'],
        'param_stability': strategy_results['phase1']['param_stability'],
        
        # Cross-symbol (Phase 2)
        'cross_symbol_avg_sharpe': strategy_results['phase2']['avg_sharpe'],
        'correlation_with_portfolio': strategy_results['phase2']['avg_correlation'],
        
        # ML enhancement (Phase 3)
        'ml_sharpe': strategy_results['phase3']['ml_sharpe'],
        'ml_boost_ratio': strategy_results['phase3']['ml_sharpe'] / strategy_results['phase1']['sharpe'],
        'trade_efficiency': strategy_results['phase3']['trades_kept_pct'],
        
        # Timestamp
        'date': datetime.now().isoformat()
    }
    
    # Save to meta-learning database
    conn = sqlite3.connect('results/meta_learning.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO meta_learning (
            strategy_name, archetype,
            raw_sharpe, trade_count, feature_count, feature_diversity,
            regime_consistency, param_stability,
            cross_symbol_avg_sharpe, correlation_with_portfolio,
            ml_sharpe, ml_boost_ratio, trade_efficiency,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(meta_entry.values()))
    
    conn.commit()
    conn.close()
```

### 10.2 Archetype Performance Analysis

```python
def analyze_archetype_performance():
    """
    Which strategy archetypes benefit most from ML?
    """
    conn = sqlite3.connect('results/meta_learning.db')
    
    query = """
        SELECT 
            archetype,
            COUNT(*) as n_strategies,
            AVG(raw_sharpe) as avg_raw_sharpe,
            AVG(ml_sharpe) as avg_ml_sharpe,
            AVG(ml_boost_ratio) as avg_boost,
            AVG(trade_efficiency) as avg_efficiency
        FROM meta_learning
        WHERE ml_sharpe IS NOT NULL
        GROUP BY archetype
        ORDER BY avg_boost DESC
    """
    
    results = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n=== Archetype Performance Analysis ===\n")
    print(results.to_string(index=False))
    
    # Insights for next research cycle
    print("\n=== Recommendations for Next Cycle ===\n")
    
    top_archetype = results.iloc[0]
    print(f"✅ Focus on '{top_archetype['archetype']}' strategies")
    print(f"   - Highest ML boost: {top_archetype['avg_boost']:.2f}x")
    print(f"   - Current count: {int(top_archetype['n_strategies'])} strategies")
    print(f"   - Recommendation: Develop 5-10 more variations\n")
    
    low_archetype = results.iloc[-1]
    print(f"⚠️  De-prioritize '{low_archetype['archetype']}' strategies")
    print(f"   - Lower ML boost: {low_archetype['avg_boost']:.2f}x")
    print(f"   - Consider: Different parameter ranges or skip archetype\n")
    
    return results

# Example Output:
"""
=== Archetype Performance Analysis ===

archetype         n_strategies  avg_raw_sharpe  avg_ml_sharpe  avg_boost  avg_efficiency
cross_asset       4             0.38            1.52           4.00       0.34
mean_reversion    7             0.42            1.38           3.29       0.37
volatility        3             0.35            1.15           3.29       0.31
momentum          5             0.40            1.05           2.63       0.29
seasonality       2             0.33            0.82           2.48       0.41

=== Recommendations for Next Cycle ===

✅ Focus on 'cross_asset' strategies
   - Highest ML boost: 4.00x
   - Current count: 4 strategies
   - Recommendation: Develop 5-10 more variations

⚠️  De-prioritize 'seasonality' strategies
   - Lower ML boost: 2.48x
   - Consider: Different parameter ranges or skip archetype
"""
```

### 10.3 Feature Importance Trends

```python
def analyze_feature_importance_trends():
    """
    Track which features consistently matter across strategies
    """
    # Aggregate feature importance from all ML models
    conn = sqlite3.connect('results/meta_learning.db')
    
    query = """
        SELECT 
            strategy_name,
            archetype,
            top_10_features
        FROM ml_results
        WHERE top_10_features IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse JSON feature importance
    all_features = {}
    for _, row in df.iterrows():
        features = json.loads(row['top_10_features'])
        for feature, importance in features.items():
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(importance)
    
    # Calculate statistics
    feature_stats = {
        feature: {
            'mean_importance': np.mean(importances),
            'median_importance': np.median(importances),
            'frequency': len(importances),
            'max_importance': max(importances)
        }
        for feature, importances in all_features.items()
    }
    
    # Sort by frequency (how often it's top-10)
    sorted_features = sorted(
        feature_stats.items(),
        key=lambda x: (x[1]['frequency'], x[1]['mean_importance']),
        reverse=True
    )
    
    print("\n=== Most Consistent Important Features ===\n")
    print(f"{'Feature':<40} {'Frequency':<12} {'Mean Importance':<18}")
    print("-" * 70)
    
    for feature, stats in sorted_features[:20]:
        print(f"{feature:<40} {stats['frequency']:<12} {stats['mean_importance']:.4f}")
    
    # Insights
    print("\n=== Feature Engineering Recommendations ===\n")
    print("High-value feature types to include in new strategies:")
    
    top_features = sorted_features[:10]
    feature_types = {}
    for feature, _ in top_features:
        if 'correlation' in feature.lower() or 'vix' in feature.lower() or 'tlt' in feature.lower():
            feature_types['cross_asset'] = feature_types.get('cross_asset', 0) + 1
        elif 'rsi' in feature.lower() or 'macd' in feature.lower() or 'bb' in feature.lower():
            feature_types['technical'] = feature_types.get('technical', 0) + 1
        elif 'volume' in feature.lower():
            feature_types['volume'] = feature_types.get('volume', 0) + 1
        elif 'volatility' in feature.lower() or 'atr' in feature.lower():
            feature_types['volatility'] = feature_types.get('volatility', 0) + 1
    
    for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {ftype}: {count} features in top-10")
    
    return sorted_features

# Example Output:
"""
=== Most Consistent Important Features ===

Feature                                  Frequency    Mean Importance
----------------------------------------------------------------------
vix_es_correlation_30                    12           0.0847
rsi_14_zscore                            11           0.0723
tlt_es_divergence                        10           0.0691
volume_surge                             10           0.0654
atr_percentile                           9            0.0598
price_zscore_20                          9            0.0576
volatility_regime                        8            0.0545
bb_position                              8            0.0512
...

=== Feature Engineering Recommendations ===

High-value feature types to include in new strategies:
  - cross_asset: 4 features in top-10
  - technical: 3 features in top-10
  - volatility: 2 features in top-10
  - volume: 1 features in top-10
"""
```

### 10.4 Strategy Generation Recommendations

```python
def generate_next_cycle_recommendations():
    """
    Based on meta-learning, suggest new strategies for next research cycle
    """
    # Analyze current coverage
    archetype_performance = analyze_archetype_performance()
    feature_importance = analyze_feature_importance_trends()
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': []
    }
    
    # Rule 1: More variations of top-performing archetypes
    top_archetype = archetype_performance.iloc[0]
    if top_archetype['avg_boost'] > 3.0:
        recommendations['high_priority'].append({
            'action': 'EXPAND_ARCHETYPE',
            'archetype': top_archetype['archetype'],
            'reason': f"Highest ML boost ({top_archetype['avg_boost']:.2f}x)",
            'suggestion': f"Develop 5-10 new {top_archetype['archetype']} strategy variations",
            'focus_features': [f[0] for f in feature_importance[:10] if 'correlation' in f[0] or 'vix' in f[0]]
        })
    
    # Rule 2: Test underexplored archetypes
    tested_archetypes = set(archetype_performance['archetype'].values)
    all_archetypes = ['mean_reversion', 'momentum', 'volatility', 'cross_asset', 
                      'seasonality', 'microstructure', 'multi_timeframe', 'regime_adaptive']
    untested = set(all_archetypes) - tested_archetypes
    
    for archetype in untested:
        recommendations['medium_priority'].append({
            'action': 'TEST_NEW_ARCHETYPE',
            'archetype': archetype,
            'reason': 'Untested archetype (potential for diversification)',
            'suggestion': f"Develop 3-5 {archetype} strategies with high feature diversity"
        })
    
    # Rule 3: Combine successful features from different strategies
    top_features = [f[0] for f in feature_importance[:15]]
    recommendations['high_priority'].append({
        'action': 'FEATURE_COMBINATION',
        'reason': 'Combine consistently important features',
        'suggestion': f"Create strategies that combine: {', '.join(top_features[:5])}",
        'expected_boost': 3.5
    })
    
    # Rule 4: De-prioritize low-performing archetypes
    low_archetype = archetype_performance.iloc[-1]
    if low_archetype['avg_boost'] < 2.0:
        recommendations['low_priority'].append({
            'action': 'DEPRIORITIZE',
            'archetype': low_archetype['archetype'],
            'reason': f"Low ML boost ({low_archetype['avg_boost']:.2f}x)",
            'suggestion': f"Reduce {low_archetype['archetype']} strategy development unless new hypothesis emerges"
        })
    
    # Generate report
    print("\n" + "="*80)
    print("STRATEGY GENERATION RECOMMENDATIONS FOR NEXT CYCLE")
    print("="*80 + "\n")
    
    print("🔥 HIGH PRIORITY\n")
    for rec in recommendations['high_priority']:
        print(f"  [{rec['action']}]")
        print(f"  Archetype: {rec.get('archetype', 'N/A')}")
        print(f"  Reason: {rec['reason']}")
        print(f"  → {rec['suggestion']}\n")
    
    print("📋 MEDIUM PRIORITY\n")
    for rec in recommendations['medium_priority']:
        print(f"  [{rec['action']}]")
        print(f"  Archetype: {rec.get('archetype', 'N/A')}")
        print(f"  Reason: {rec['reason']}")
        print(f"  → {rec['suggestion']}\n")
    
    print("⚠️  LOW PRIORITY\n")
    for rec in recommendations['low_priority']:
        print(f"  [{rec['action']}]")
        print(f"  Archetype: {rec.get('archetype', 'N/A')}")
        print(f"  Reason: {rec['reason']}")
        print(f"  → {rec['suggestion']}\n")
    
    return recommendations
```

---

## 9. Tier 1 Strategy Implementations

### 9.1 Pragmatic Implementation Approach

**Philosophy**: Build great infrastructure with 10 high-priority strategies first, then expand.

**Why These 10 Strategies:**
- ✅ Low-Medium complexity (implementable in 1 week)
- ✅ High trade volume potential (10k+ trades expected)
- ✅ Diverse archetypes (mean reversion, momentum, breakout)
- ✅ Well-documented in literature (less risk of implementation bugs)
- ✅ Parameter grids manageable (~4-81 combos each with fixed exits)

### 9.2 Selected Strategies

| ID | Name | Archetype | Combos | Expected Trades | Priority |
|----|------|-----------|--------|-----------------|----------|
| 21 | **RSI(2) Mean Reversion** | Mean Reversion | 36 | 15,000+ | ⭐️⭐️⭐️ |
| 1 | **Bollinger Bands** | Mean Reversion | 16 | 12,000+ | ⭐️⭐️⭐️ |
| 36 | **RSI(2) + 200 SMA Filter** | Mean Reversion | 81 | 15,000+ | ⭐️⭐️⭐️ |
| 37 | **Double 7s** | Mean Reversion | 27 | 10,000+ | ⭐️⭐️⭐️ |
| 17 | **MA Cross** | Trend Following | 16 | Variable | ⭐️⭐️ |
| 24 | **VWAP Reversion** | Mean Reversion | 4 | 8,000+ | ⭐️⭐️ |
| 23 | **Gap Fill** | Mean Reversion | 7 | 3,000-5,000 | ⭐️⭐️ |
| 25 | **Opening Range Breakout** | Breakout | 9 | 2,000-4,000 | ⭐️⭐️ |
| 19 | **MACD** | Momentum | 27 | 4,000-6,000 | ⭐️⭐️ |
| 15 | **Price Channel Breakout** | Breakout | 12 | 2,000-3,000 | ⭐️⭐️ |

**Total: ~235 parameter combinations** (vs 14,000 if testing all exit combinations)

### 9.3 Fixed Exit Parameters (Phase 1)

To reduce search space in Phase 1, we use **fixed exits**:

```yaml
stop_loss_atr: 1.0    # 1x ATR(14) stop loss
take_profit_atr: 1.0  # 1x ATR(14) profit target
max_bars_held: 20     # Maximum 20 bars in position
auto_close_time: "16:00 EST"  # Close all positions at 4pm
```

**Rationale:**
- Reduces combinations from ~14,000 to ~235 (60x speedup)
- 1.0 ATR is literature-standard baseline
- Can optimize exits in Phase 2 for winning strategies
- Focus Phase 1 on finding signal edges, not exit optimization

### 9.4 Parameter Grids (Fixed Exits)

#### Strategy #21: RSI(2) Mean Reversion
```yaml
rsi_length: [2, 3, 4]
rsi_oversold: [5, 10, 15]
rsi_overbought: [60, 65, 70, 75]
# Total: 3 × 3 × 4 = 36 combinations
```

#### Strategy #1: Bollinger Bands
```yaml
bb_length: [15, 20, 25, 30]
bb_stddev: [1.5, 2.0, 2.5, 3.0]
# Total: 4 × 4 = 16 combinations
```

#### Strategy #36: RSI(2) + 200 SMA Filter
```yaml
rsi_length: [2, 3, 4]
rsi_oversold: [3, 5, 10]
rsi_overbought: [65, 70, 75]
sma_filter: [150, 200, 250]
# Total: 3 × 3 × 3 × 3 = 81 combinations
```

#### Strategy #37: Double 7s
```yaml
percentile_window: [5, 7, 10]
entry_pct: [3, 5, 10]
exit_pct: [90, 95, 97]
# Total: 3 × 3 × 3 = 27 combinations
```

#### Strategy #17: MA Cross
```yaml
ma_fast: [5, 10, 15, 20]
ma_slow: [30, 50, 75, 100]
# Total: 4 × 4 = 16 combinations (filtered: fast < slow)
```

#### Strategy #24: VWAP Reversion
```yaml
vwap_std_threshold: [1.5, 2.0, 2.5, 3.0]
# Total: 4 combinations
```

#### Strategy #23: Gap Fill
```yaml
gap_threshold: [0.5, 1.0, 1.5, 2.0]
gap_fill_target: [0.3, 0.5, 0.7]
# Total: 4 × 3 = 12 combinations (reduced from original)
```

#### Strategy #25: Opening Range Breakout
```yaml
or_duration_minutes: [15, 30, 60]
or_breakout_pct: [0.0, 0.1, 0.2]
# Total: 3 × 3 = 9 combinations
```

#### Strategy #19: MACD
```yaml
macd_fast: [8, 12, 16]
macd_slow: [21, 26, 31]
macd_signal: [7, 9, 11]
# Total: 3 × 3 × 3 = 27 combinations
```

#### Strategy #15: Price Channel Breakout
```yaml
channel_length: [15, 20, 25, 30]
channel_breakout_pct: [0.0, 0.25, 0.5]
# Total: 4 × 3 = 12 combinations
```

### 9.5 Implementation Timeline (1 Week)

**Day 1-2: Infrastructure**
- ✅ BaseStrategy class with entry/exit/feature interfaces
- ✅ ParameterGrid generator (itertools.product)
- ✅ Vectorized backtester (fast numpy operations)
- ✅ SQLite results database
- ✅ Parallel execution engine (multiprocessing, 16 workers)
- ✅ Real-time progress tracking (tqdm)

**Day 3-4: Strategy Implementations**
- ✅ Implement all 10 strategies
- ✅ Unit tests for each strategy
- ✅ Indicator calculations (RSI, Bollinger, MACD, VWAP, ATR)
- ✅ Data loading integration

**Day 5: Phase 1 Execution**
- ✅ Run all 235 backtests on ES 15-min (2010-2024)
- ✅ Apply filters: trade count, Sharpe, walk-forward, regime, stability
- ✅ Generate Phase 1 report with top 5-10 strategies

**Day 6: Phase 2 Multi-Symbol**
- ✅ Run winners on all symbols (NQ, YM, RTY, GC, etc.)
- ✅ Correlation analysis
- ✅ Portfolio simulation
- ✅ Select 2-4 strategies for ML

**Day 7: Phase 3 ML Integration**
- ✅ Pipe winners to extract_training_data.py
- ✅ Run train_rf_cpcv_bo.py on each
- ✅ Compare raw vs ML performance
- ✅ Final report + recommendations

---

## 10. Execution Guide

### 10.1 Phase 1: Raw Strategy Testing

```bash
# Run Phase 1: Test 10 strategies × ~24 params = 235 backtests on ES
python research/strategy_factory/main.py \
    --phase 1 \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --timeframe 15min \
    --workers 16 \
    --output results/strategy_factory_phase1_$(date +%Y%m%d_%H%M%S)

# Expected runtime: ~30-60 minutes with 16 cores
# Output: SQLite database + markdown report + charts
```

**Filters Applied (Sequential):**
1. ✅ Trade count ≥ 10,000
2. ✅ Sharpe ≥ 0.2
3. ✅ Profit Factor ≥ 1.15
4. ✅ Max Drawdown ≤ 30%
5. ✅ Walk-forward validation (Test Sharpe / Train Sharpe ≥ 0.5)
6. ✅ Regime consistency (Sharpe ≥ 0.2 in 2 of 3 regimes)
7. ✅ Parameter stability (variation < 40%)
8. ✅ Statistical significance (FDR-corrected p < 0.05)

**Expected Output:**
- 5-10 strategies pass all filters
- Ranked by composite score (robustness, consistency, diversification)

### 10.2 Phase 2: Multi-Symbol Validation

```bash
# Run Phase 2: Test Phase 1 winners on all symbols
python research/strategy_factory/main.py \
    --phase 2 \
    --input results/strategy_factory_phase1_TIMESTAMP/phase1_winners.csv \
    --symbols ES NQ YM RTY GC SI HG CL NG 6A 6B 6C 6J 6M 6N 6S TLT \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --timeframe 15min \
    --workers 16 \
    --output results/strategy_factory_phase2_$(date +%Y%m%d_%H%M%S)

# Expected runtime: ~1-2 hours
# Output: Correlation matrix, portfolio simulations, alpha tests
```

**Filters Applied:**
1. ✅ ≥3 of 4 equity symbols (ES/NQ/YM/RTY) with Sharpe ≥ 0.3
2. ✅ Average Sharpe ≥ 0.5 across all tested symbols
3. ✅ Max correlation < 0.7 with existing strategies
4. ✅ Portfolio Sharpe improvement > 0.05

**Expected Output:**
- 2-4 strategies ready for ML pipeline
- Recommendation: which strategies to deploy

### 10.3 Phase 3: ML Pipeline Integration

```bash
# For each Phase 2 winner, run feature extraction
python research/extract_training_data.py \
    --strategy RSI2_MeanRev_v1 \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31

# Then train ML model with existing pipeline
python research/train_rf_cpcv_bo.py \
    --symbol ES \
    --strategy RSI2_MeanRev_v1 \
    --n_trials 100 \
    --cv_splits 5

# Backtest with ML filter
python research/backtest_runner.py \
    --strategy RSI2_MeanRev_v1 \
    --symbol ES \
    --ml_model models/ES_RSI2_MeanRev_v1_rf_model.pkl \
    --start 2022-01-01 \
    --end 2024-12-31
```

**Expected Performance Improvement:**
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-2.0+
- Improvement: 2-4x
- Trades kept: 30-40% of raw signals

---

## Conclusion

This Strategy Factory guide establishes a rigorous, systematic approach to strategy research that feeds your world-class ML pipeline. Key takeaways:

1. **Volume is King:** 10,000+ trades are essential for effective ML training
2. **Robustness > Raw Performance:** Let ML enhance weak but robust edges
3. **Portfolio-Level Thinking:** Diversification matters more than single-strategy Sharpe
4. **Statistical Rigor:** Multiple testing corrections prevent false discoveries
5. **Pragmatic Implementation:** Start with 10 proven strategies, build infrastructure for future expansion

**Implementation Summary (Week 1):**
- 🎯 10 Tier 1 strategies implemented
- 🎯 ~235 parameter combinations tested (fixed exits)
- 🎯 ES 15-min bars (2010-2024) → Multi-symbol validation → ML integration
- 🎯 16-core parallel execution (~2-3 hours total runtime)
- 🎯 2-4 production-ready strategies for live trading

**Next Steps:**
1. Run data inspection commands to identify data format
2. Build core infrastructure (Days 1-2)
3. Implement 10 strategies (Days 3-4)
4. Execute Phases 1-3 (Days 5-7)
3. Run Phase 1 screening
4. Iterate based on meta-learning insights

This system transforms strategy research from art to science, matching the sophistication of your existing CPCV + Bayesian optimization ML pipeline.
