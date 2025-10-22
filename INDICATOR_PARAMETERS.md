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

## Warmup Status Summary

### ✅ Fully Warm (Have Sufficient Data)
- IBS indicators
- RSI indicators (all variants)
- Moving averages (EMAs)
- Bollinger Bands
- Donchian Channels
- Pivot Points
- Range compression indicators
- VIX regime filter

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

Based on ML model features loaded, the active indicators include:
- Cross z-scores for various symbols (ES, NQ, TLT, VIX, currencies, metals)
- ATR z-scores and percentiles
- Volume z-scores
- RSI variants (2, 14 period)
- IBS and related metrics
- Return-based features
- Directional indicators

**Estimated Time to Full Warmup:** 10.5 days (252 hours of hourly bars)
