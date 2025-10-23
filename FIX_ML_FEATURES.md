# CRITICAL FINDINGS: All 26 Missing Features ARE Required by Your Models

## TL;DR
**All 26 missing features are in your ML model feature lists.** They must be fixed - you cannot ignore them.

---

## ROOT CAUSE IDENTIFIED

After analyzing your code, I found the issue at `src/strategy/ibs_strategy.py:1340-1341`:

```python
data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
if data_feed is None:
    continue  # Silently skips creating the z-score pipeline!
```

**The cross-symbol z-score pipelines aren't being created because `_get_cross_feed()` is returning `None` for these symbols.**

At `src/strategy/ibs_strategy.py:2802-2810`, when a feed is missing:
```python
message = f"Missing data feed {feed_name} for {symbol} {feed_suffix} data"
if not previously_missing:
    logging.warning(message)  # Logs a warning
missing_cache.add(key)
return None  # Returns None, causing pipeline to be skipped
```

---

## DIAGNOSTIC STEPS

### Step 1: Check for "Missing data feed" warnings

Run this command to see which feeds are missing:

```bash
sudo journalctl -u pine-runner.service --no-pager | grep -i "missing data feed" | tail -50
```

This will show you EXACTLY which cross-symbol feeds are not available when the strategy initializes.

### Step 2: Verify feed names

The code expects feeds to be named exactly as:
- `6A_hour`, `6A_day`
- `6B_hour`, `6B_day`
- `ES_hour`, `ES_day`
- `VIX_hour`, `VIX_day`
- etc.

Your logs show:
```
Converted 6A historical data to 361 1h bars (queuing to feed: 6A_hour) ✓
Converted 6A historical data to 253 1d bars (queuing to feed: 6A_day) ✓
```

So the naming is correct. The issue is likely **timing** or **feed registration**.

---

## LIKELY CAUSES

### Cause #1: VIX Not in Contract Map (CONFIRMED)
VIX is NOT in your `/home/user/rooney-capital-v1/Data/Databento_contract_map.yml` file.

**Affects:** 4 features (vix_daily_return, vix_hourly_return)

**Fix:** Add VIX to contract map (see below)

### Cause #2: Feed Registration Timing Issue
The strategy might be initializing BEFORE all cross-symbol feeds are registered with Cerebro.

**Affects:** Most z-score features

**Debug:** Check when `self.getdatabyname(name)` is called vs. when feeds are added

### Cause #3: Feed Availability Check Failing
The feeds exist but `self.getdatabyname(name)` at line 2783 is throwing a KeyError.

**Affects:** All cross-symbol z-scores

**Debug:** Add logging to see which feeds are available during __init__

---

## FIXES

### FIX #1: Add VIX to Contract Map (IMMEDIATE)

Edit `/home/user/rooney-capital-v1/Data/Databento_contract_map.yml` and add VIX to the `reference_feeds` section:

```json
{
  "symbol": "VIX",
  "databento": {
    "dataset": "GLBX.MDP3",
    "product_id": "VX.FUT"
  }
}
```

**Note:** Verify the correct Databento product_id for VIX futures. You may need to check Databento docs.

After adding, restart the service:
```bash
sudo systemctl restart pine-runner.service
```

This will fix 4 features across 3 symbols (CL, HG, RTY).

---

### FIX #2: Enable Momentum Indicator for RTY

Add to your RTY strategy configuration:
```python
enableMom3=True
```

This will fix 2 features: `mom3_z_pct`, `momentum_z_entry_daily`

---

### FIX #3: Ensure Base Features for Derived Calculations

For symbols that need `rsixatrz` or `rsixvolz`, ensure these are enabled:
```python
enableRSI=True
enableATRZ=True
enableVOLZ=True
```

**Affected symbols:**
- **rsixatrz:** HG, NQ
- **rsixvolz:** NQ, SI, YM

---

### FIX #4: Debug Cross-Symbol Feed Availability (CRITICAL)

The main issue is the 18 missing cross-symbol z-scores. To debug:

**Option A: Add debug logging**

Edit `src/strategy/ibs_strategy.py` at line 1336 (right before the feed check):

```python
if not need_indicator:
    continue

# ADD THIS DEBUG LOGGING:
import logging
logger = logging.getLogger(__name__)
logger.info(
    f"Attempting to get cross feed: {symbol}_{feed_suffix} for enable_param={enable_param}, need_indicator={need_indicator}"
)

data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)

# ADD THIS TOO:
if data_feed is None:
    logger.warning(f"Cross feed {symbol}_{feed_suffix} returned None - z-score pipeline will NOT be created")
    continue
else:
    logger.info(f"Successfully got cross feed {symbol}_{feed_suffix} - creating z-score pipeline")
```

Then restart and check logs to see which feeds are failing.

**Option B: Force enable z-score parameters**

As a temporary workaround, manually enable all needed z-score parameters in your strategy configuration:

```python
# For CL model:
enable6AZScoreHour=True,
enable6CZScoreHour=True,
enable6EZScoreHour=True,
enable6MZScoreHour=True,
enableESZScoreHour=True,
enableRTYZScoreHour=True,
enableYMZScoreHour=True,

# For HG model:
enable6AZScoreHour=True,
enable6CZScoreHour=True,
enable6EZScoreHour=True,
enable6NZScoreHour=True,
enableESZScoreHour=True,
enableRTYZScoreHour=True,

# For NQ model:
enable6EZScoreHour=True,
enable6JZScoreDay=True,
enable6JZScoreHour=True,
enable6SZScoreHour=True,
enableGCZScoreHour=True,
enableTLTZScoreDay=True,

# For RTY model:
enable6BZScoreHour=True,
enable6CZScoreHour=True,
enableNGZScoreDay=True,
enableTLTZScoreDay=True,

# For SI model:
enable6AZScoreHour=True,
enable6BZScoreHour=True,
enable6CZScoreHour=True,
enable6JZScoreHour=True,
enable6MZScoreHour=True,
enable6SZScoreHour=True,
enableHGZScoreHour=True,

# For YM model:
enable6CZScoreHour=True,
enable6NZScoreDay=True,
enable6NZScoreHour=True,
enableCLZScoreHour=True,
enableNGZScoreHour=True,
enableSIZScoreDay=True,
enableTLTZScoreDay=True,
```

When these parameters are explicitly set to `True`, the code at line 1332 will trigger:
```python
getattr(self.p, enable_param, False)  # This will be True
```

And force the pipeline creation even if the ML feature matching fails.

---

## VALIDATION

After applying fixes, run your verification script again:

```bash
./verify_features.py
```

You should see:
- **Before:** 21/30 features for CL (70%)
- **After:** 30/30 features for CL (100%) ✓

---

## WHY THIS MATTERS

Your ML models were trained on 30 features each. If only 21/30 features are available:
- The model receives incomplete input data
- Prediction quality degrades significantly
- The missing features might be the most important ones!

For example, CL's model expects `es_hourly_z_score` but isn't getting it. If ES z-score is a key predictor for CL movements, your model is essentially blind to that signal.

---

## NEXT STEPS

1. **Immediate:** Add VIX to contract map and restart
2. **Quick:** Enable mom3 for RTY
3. **Quick:** Verify rsi/atrz/volz are enabled for symbols needing derived features
4. **Debug:** Add logging to find which cross-feeds are failing
5. **Workaround:** Force enable z-score parameters
6. **Validate:** Run verification script to confirm 100% feature coverage

Once you fix these, all 137 features should calculate properly and your ML models will operate at full capacity.
