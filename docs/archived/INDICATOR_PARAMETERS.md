# Indicator Parameters Reference

This document lists all indicators used in the IBS strategy and their data requirements.

## Summary

Most indicators require **252 bars** to be fully "warm" (one trading year). This is the primary bottleneck for warmup completion.

---

## 1. Cross-Asset Z-Score Indicators

These calculate correlations with other instruments and require the most data.

### Symbols Tracked
- ES, NQ, RTY, YM (equity futures)
- GC, SI, HG, PL (metals)
- CL, NG (energy)
- 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S (currencies)
- TLT (bonds)
- VIX (volatility)

### Timeframes
Each symbol tracks:
- **Hourly** z-scores
- **Daily** z-scores

### Parameters (per symbol/timeframe)
- **SMA Length**: 20 bars (default)
- **StdDev Window**: **252 bars** (default) ⚠️ **PRIMARY BOTTLENECK**
- **Z-Score Range**: -2.0 to 2.0 (entry thresholds)

### Data Requirements
- **Hourly feeds**: Need 252 hourly bars = **252 hours** = **10.5 days** of live data
- **Daily feeds**: Need 252 daily bars = **252 days** = **~1 year** of live data

**Status**: Currently only 4-5 bars available (not warm)

---

## 2. ATR Z-Score Indicators

Measures volatility relative to historical norms.

### Intraday ATR Z-Score (enableATRZ)
- **ATR Length**: 5 bars
- **Z-Score Window**: **252 bars** ⚠️
- **Timeframe**: Hourly
- **Range**: -2.0 to 2.0

### Daily ATR Z-Score (enableDATRZ)
- **ATR Length**: 5 bars
- **Z-Score Window**: **252 bars** ⚠️
- **Timeframe**: Daily
- **Range**: -2.0 to 2.0

### Daily ATR Percentile (enableDailyATRPercentile)
- **ATR Length**: 14 bars
- **Percentile Window**: **252 bars** ⚠️
- **Timeframe**: Daily
- **Range**: 0-100

### Hourly ATR Percentile (enableHourlyATRPercentile)
- **ATR Length**: 14 bars
- **Percentile Window**: **252 bars** ⚠️
- **Timeframe**: Hourly
- **Range**: 0-100

**Data Requirements**: 252 bars on respective timeframe

---

## 3. Volume Z-Score Indicators

Measures volume relative to historical norms.

### Intraday Volume Z-Score (enableVolZ)
- **Volume SMA Length**: 5 bars
- **Z-Score Window**: **252 bars** ⚠️
- **Timeframe**: Hourly
- **Range**: -2.0 to 2.0

### Daily Volume Z-Score (enableDVolZ)
- **Volume SMA Length**: 5 bars
- **Z-Score Window**: **252 bars** ⚠️
- **Timeframe**: Daily
- **Range**: -2.0 to 2.0

**Data Requirements**: 252 bars on respective timeframe

---

## 4. RSI (Relative Strength Index) Indicators

Momentum oscillators - these are already WARM.

### Hourly RSI (enableRSIEntry)
- **Period**: 5 bars
- **Range**: 40-60
- **Timeframe**: Hourly
- **Data Required**: 5 bars ✅

### Hourly RSI 2-period (enableRSIEntry2Len)
- **Period**: 2 bars
- **Range**: 0-40
- **Timeframe**: Hourly
- **Data Required**: 2 bars ✅

### Hourly RSI 14-period (enableRSIEntry14Len)
- **Period**: 14 bars
- **Range**: 40-60
- **Timeframe**: Hourly
- **Data Required**: 14 bars ✅

### Daily RSI (enableDailyRSI)
- **Period**: 5 bars
- **Range**: 0-40
- **Timeframe**: Daily
- **Data Required**: 5 bars ✅

### Daily RSI 2-period (enableDailyRSI2Len)
- **Period**: 2 bars
- **Range**: 0-40
- **Timeframe**: Daily
- **Data Required**: 2 bars ✅

### Daily RSI 14-period (enableDailyRSI14Len)
- **Period**: 14 bars
- **Range**: 40-60
- **Timeframe**: Daily
- **Data Required**: 14 bars ✅

**Status**: All RSI indicators are WARM (low data requirements)

---

## 5. IBS (Internal Bar Strength) Indicators

Primary signal - these are WARM.

### Current IBS
- **Calculation**: `(Close - Low) / (High - Low)`
- **Data Required**: 1 bar ✅

### Previous IBS
- **Data Required**: 2 bars ✅

### Daily IBS
- **Data Required**: 1 daily bar ✅

**Status**: IBS indicators are WARM

---

## 6. Moving Average Indicators

These are WARM (short lookback periods).

### EMA 8 (enableEMA8)
- **Period**: 8 bars
- **Timeframe**: Hourly
- **Data Required**: 8 bars ✅

### EMA 20 (enableEMA20)
- **Period**: 20 bars
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

### EMA 50 (enableEMA50)
- **Period**: 50 bars
- **Timeframe**: Hourly
- **Data Required**: 50 bars ✅

### EMA 200 (enableEMA200)
- **Period**: 200 bars
- **Timeframe**: Hourly
- **Data Required**: 200 bars ✅

### Daily EMAs (enableEMA20D, enableEMA50D, enableEMA200D)
- **Periods**: 20, 50, 200 days
- **Timeframe**: Daily
- **Data Required**: 200 bars ✅

**Status**: All EMAs are WARM (historical warmup provided sufficient data)

---

## 7. Bollinger Bands

These are WARM (short lookback).

### Intraday Bollinger Bands (enableBB)
- **Period**: 20 bars
- **Standard Deviations**: 2.0
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

### Daily Bollinger Bands (enableBBHighD)
- **Period**: 20 bars
- **Standard Deviations**: 2.0
- **Timeframe**: Daily
- **Data Required**: 20 bars ✅

**Status**: Bollinger Bands are WARM

---

## 8. Additional Indicators (All WARM)

### Donchian Channel (enableDonch)
- **Period**: 20 bars
- **Timeframe**: Daily
- **Data Required**: 20 bars ✅

### Pivot Points
- **Left Bars**: 5
- **Right Bars**: 5
- **Data Required**: 11 bars ✅

### Range Compression (enableRangeCompressionATR)
- **ATR Period**: 14 bars
- **Data Required**: 14 bars ✅

### VIX Regime (enable_vix_reg)
- **SMA Period**: 100 bars
- **Timeframe**: Daily
- **Data Required**: 100 bars ✅

---

## 9. ADX (Average Directional Index)

Measures trend strength.

### Hourly ADX (enableADX)
- **Period**: 14 bars
- **Range**: 20.0-40.0
- **Timeframe**: Hourly
- **Data Required**: 14 bars ✅

### Daily ADX (enableDADX)
- **Period**: 14 bars
- **Range**: 20.0-40.0
- **Timeframe**: Daily
- **Data Required**: 14 bars ✅

**Status**: ADX indicators are WARM

---

## 10. Parabolic SAR

Trailing stop and reversal indicator.

### Parabolic SAR (enablePSAR)
- **Timeframe**: Hourly
- **Acceleration Factor**: 0.02
- **AF Maximum**: 0.2
- **Range**: -1.0 to 1.0
- **Data Required**: ~14 bars ✅

**Features**: parabolic_sar_distance, parabolic_sar_distance_daily

**Status**: Parabolic SAR is WARM

---

## 11. Moving Average Slope & Spread

Measures trend direction and MA convergence/divergence.

### MA Slope (enableMASlope)
- **EMA Period**: 20 bars
- **Shift**: 5 bars
- **ATR Period**: 14 bars
- **Range**: -1.0 to 1.0
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

### Daily MA Slope (enableDailySlope)
- **EMA Period**: 20 bars
- **Shift**: 5 bars
- **ATR Period**: 14 bars
- **Range**: -1.0 to 1.0
- **Timeframe**: Daily
- **Data Required**: 20 bars ✅

### MA Spread/Ribbon Tightness (enableMASpread)
- **Fast EMA**: 20 bars
- **Slow EMA**: 50 bars
- **ATR Period**: 14 bars (optional)
- **Range**: 0.0 to 1.0
- **Timeframe**: Hourly (hourly & daily variants)
- **Data Required**: 50 bars ✅

**Features**: ma_slope_fast, daily_slope_fast, ma_spread_ribbon_tightness, ma_spread_daily_ribbon_tightness

**Status**: MA slope/spread indicators are WARM

---

## 12. Momentum Z-Score

Measures momentum relative to historical norms.

### Momentum Z-Score (enableMom3, mom3_z_pct)
- **Momentum Length**: 3 bars
- **Z-Score Window**: 20 bars
- **Range**: -2.0 to 2.0
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

**Features**: mom3_z_pct, momentum_z_entry_daily, momentum_z_percentile

**Status**: Momentum indicators are WARM

---

## 13. Directional Drift

Measures directional momentum over time.

### Hourly Directional Drift (enableDirDrift)
- **Period**: 50 bars
- **Range**: -2.0 to 2.0
- **Timeframe**: Hourly
- **Data Required**: 50 bars ✅

### Daily Directional Drift (enableDirDriftD)
- **Period**: 20 bars
- **Range**: -2.0 to 2.0
- **Timeframe**: Daily
- **Data Required**: 20 bars ✅

**Features**: hourly_directional_draft, hourly_directional_drift, daily_directional_drift

**Status**: Directional drift indicators are WARM

---

## 14. Pair Indicators

Calculates IBS and z-scores relative to a paired symbol (e.g., ES paired with NQ).

### Pair IBS (enablePairIBS)
- **Paired Symbol**: Defined in PAIR_MAP config
- **Timeframe**: Hourly
- **Range**: 0.0 to 0.2
- **Data Required**: 1-2 bars ✅

### Pair Z-Score (enablePairZ)
- **Paired Symbol**: Defined in PAIR_MAP config
- **Z-Score Window**: 20 bars
- **Range**: -2.0 to 2.0
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

**Features**: pair_ibs, pair_ibs_pct, pair_ibs_daily, pair_ibs_percentile, pair_z, pair_z_pct, pair_z_score_daily

**Status**: Pair indicators are WARM

---

## 15. Spiral Efficiency Ratio

Measures price path efficiency using multiple lookback periods.

### Spiral ER (enableSpiralER)
- **Lookbacks**: 5, 20, 60 bars
- **Weights**: 0.5, 0.25, 0.125
- **Range**: 0.0 to 1.0
- **Timeframe**: Hourly (hourly & daily variants)
- **Data Required**: 60 bars ✅

**Features**: spiral_efficiency_ratio, spiral_efficiency_ratio_daily

**Status**: Spiral indicators are WARM

---

## 16. Bollinger Bandwidth

Measures the width of Bollinger Bands (volatility expansion/contraction).

### Bollinger Bandwidth (enableBBW)
- **Period**: 20 bars
- **Standard Deviations**: 2.0
- **Range**: 0.0 to 1.0
- **Timeframe**: Hourly (daily variant available)
- **Data Required**: 20 bars ✅

**Features**: bollinger_bandwidth_daily

**Status**: Bollinger Bandwidth is WARM

---

## 17. Donchian Proximity

Measures proximity to nearest Donchian Channel band.

### Donchian Proximity (enableDonchProx)
- **Period**: 20 bars
- **ATR Period**: 14 bars
- **Range**: 0.0 to 1.0
- **Timeframe**: Hourly (hourly & daily variants)
- **Data Required**: 20 bars ✅

**Features**: donchian_proximity_to_nearest_band, donchian_proximity_daily_to_nearest_band

**Status**: Donchian Proximity is WARM

---

## 18. Range Compression (Additional)

Daily variant of range compression using ATR.

### Daily Range Compression (enableDailyRangeCompression)
- **ATR Period**: 14 bars
- **Threshold**: 0.5
- **Timeframe**: Daily
- **Data Required**: 14 bars ✅

**Features**: daily_range_compression

**Status**: Daily Range Compression is WARM

---

## 19. TR/ATR Ratio

Measures current True Range relative to Average True Range.

### TR/ATR Percentile (enableTRATR, tratr_pct)
- **ATR Period**: 20 bars
- **Range**: 1.0 to 2.0
- **Timeframe**: Hourly
- **Data Required**: 20 bars ✅

**Features**: tr_atr_percentile, tratr_pct

**Status**: TR/ATR indicators are WARM

---

## 20. Previous Bar & Day Returns

Simple return calculations from previous bars.

### Previous Day Return (enablePrevDayPct, prev_day_pct)
- **Range**: -2.0 to 2.0
- **Timeframe**: Daily
- **Data Required**: 2 bars (current + previous) ✅

### Previous Bar Return (enablePrevBarPct, prev_bar_pct)
- **Range**: -1.0 to 1.0
- **Timeframe**: Hourly (configurable)
- **Data Required**: 2 bars (current + previous) ✅

**Features**: prev_day_pctxvalue, prev_bar_pct, prev_bar_pct_pct

**Status**: Return indicators are WARM

---

## 21. Price & Value Indicators

Simple price and value snapshots.

### Price USD (price_usd)
- **Description**: Current price in USD
- **Data Required**: 1 bar ✅

### Open/Close (enableOpenClose)
- **Description**: Open vs close relationship
- **Data Required**: 1 bar ✅

### Value Filter (useValFilter, value_pct)
- **Description**: Position value filter
- **Data Required**: 1 bar ✅

**Features**: price_usd, open_close, value, value_pct

**Status**: Price/Value indicators are WARM

---

## 22. Cross-Asset Returns

Return calculations from cross-asset z-score feeds.

These features calculate simple returns for all tracked symbols:
- **Equity futures**: ES, NQ, RTY, YM
- **Metals**: GC, SI, HG, PL
- **Energy**: CL, NG
- **Currencies**: 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
- **Bonds**: TLT
- **Volatility**: VIX

### Timeframes
- **Hourly returns**: `{symbol}_hourly_return`
- **Daily returns**: `{symbol}_daily_return`

**Data Required**: 1-2 bars (just the return calculation) ✅

**Example Features**: 6a_daily_return, 6a_hourly_return, es_hourly_return, nq_daily_return, tlt_daily_return, vix_hourly_return

**Status**: Cross-asset returns are WARM (calculated from z-score feeds)

---

## 23. Distance Z-Score

Distance-based z-score indicator.

**Features**: distance_z_entry_daily, dist_z_pct

**Note**: Appears in ML features but implementation details need verification.

**Status**: Likely WARM (short lookback)

---

## Warmup Status Summary

### ✅ Fully Warm (Have Sufficient Data)
- IBS indicators (all variants including pair IBS)
- RSI indicators (all variants)
- Moving averages (EMAs, MA slope, MA spread/ribbon tightness)
- Bollinger Bands & Bollinger Bandwidth
- Donchian Channels & Donchian Proximity
- Pivot Points
- Range compression indicators (hourly & daily)
- VIX regime filter
- ADX (hourly & daily)
- Parabolic SAR
- Momentum z-scores
- Directional drift (hourly & daily)
- Pair indicators (pair IBS, pair z-score)
- Spiral efficiency ratio
- TR/ATR ratio
- Previous bar & day returns
- Price & value indicators
- Cross-asset returns
- Distance z-scores

### ⚠️ NOT Warm (Insufficient Data)
**Cross-Asset Z-Scores** (Need 252 bars per timeframe):
- All hourly cross z-scores: **4-5 / 252 bars** (need 10.5 days)
- All daily cross z-scores: **0 / 252 bars** (need 1 year)

**ATR/Volume Z-Scores** (Need 252 bars):
- Hourly ATR Z-Score: **5 / 252 bars**
- Daily ATR Z-Score: **0 / 252 bars**
- Hourly Volume Z-Score: **5 / 252 bars**
- Daily Volume Z-Score: **0 / 252 bars**
- ATR Percentiles: **5 / 252 bars**

---

## Why 252 Bars?

**252 = Trading Days in One Year**

Many statistical indicators use a full year of data to:
- Calculate "normal" volatility ranges
- Identify how unusual current conditions are
- Provide stable statistical distributions

This is a finance industry standard for:
- Z-scores (standard deviations from mean)
- Percentile rankings
- Volatility measures

---

## Options to Reduce Warmup Time

### Option 1: Reduce Window Size
Change the 252-bar window to a smaller value (e.g., 60, 90, 120 bars):

**Pros:**
- Faster warmup (60 bars = 2.5 days vs 10.5 days)
- System ready to trade sooner

**Cons:**
- Uses less historical data (may affect strategy performance)
- Statistical measures less stable
- Needs backtesting to validate

### Option 2: Accept Natural Warmup
Let the system run for 10.5 days to accumulate hourly data naturally:

**Pros:**
- Strategy works exactly as designed
- No changes to tested parameters

**Cons:**
- 10.5 day wait for full functionality
- Partial ML features during warmup

### Option 3: Disable Long-Lookback Indicators
Temporarily disable indicators requiring 252 bars:

**Pros:**
- Immediate full warmup
- Can re-enable after 10 days

**Cons:**
- Reduced ML feature set during warmup
- Different strategy behavior temporarily

---

## Current Configuration

Based on ML model features loaded across all 12 symbols (ES, NQ, RTY, YM, GC, SI, HG, CL, NG, 6A, 6B, 6E), the strategy uses **133 unique features** including:

### Statistical Indicators (Require 252 bars - NOT WARM)
- Cross-asset z-scores (hourly & daily) for 24 symbols
- ATR z-scores and percentiles (hourly & daily)
- Volume z-scores and percentiles (hourly & daily)

### Technical Indicators (WARM)
- **RSI variants**: 2, 5, 14 period (hourly & daily)
- **IBS indicators**: Current, previous, daily, pair IBS
- **Moving averages**: EMA 8/20/50/200, MA slope, MA spread/ribbon tightness
- **Bollinger Bands**: Standard bands + bandwidth
- **Donchian**: Channels + proximity to bands
- **ADX**: Hourly & daily trend strength
- **Parabolic SAR**: Trailing stop indicator
- **Momentum**: 3-period momentum z-scores
- **Directional drift**: Hourly (50-bar) & daily (20-bar)
- **Range compression**: Hourly & daily ATR compression
- **TR/ATR ratio**: True range percentiles
- **Pivot points**: 5-bar left/right pivots
- **Spiral efficiency**: Multi-period path efficiency
- **Pair indicators**: Pair IBS & pair z-scores

### Derived Features (WARM)
- **Cross-asset returns**: Hourly & daily returns for 24 symbols
- **Previous returns**: Previous bar & previous day
- **Price/Value**: Price USD, open/close, value filter
- **Distance**: Distance z-scores

### Summary
- **Total features**: 133
- **Warm indicators**: ~110+ features (short lookback periods)
- **NOT warm indicators**: ~20-25 features (252-bar statistical indicators)

**Estimated Time to Full Warmup:** 10.5 days (252 hours of hourly bars) for all cross-asset z-scores, ATR percentiles, and volume percentiles
