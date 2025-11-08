# IBS Strategy Split Guide

## Overview

Splitting the IBS strategy into two independent strategies (IBS A and IBS B) to:
- Better utilize capital ($150k each instead of shared)
- Reduce correlation between strategies
- Increase diversification
- Each has independent daily stop loss ($2,500 each)

---

## Asset Class Distribution

**Total: 18 symbols with ML models**

- **Equities (4):** ES, NQ, RTY, YM
- **Currencies (8):** 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
- **Commodities (6):** CL, NG, GC, SI, HG, PL

---

## Selected Grouping: Balanced Mixed Split ✅

Each strategy gets an equal mix of all asset classes:
- 2 Equities each
- 4 Currencies each
- 3 Commodities each

### **IBS A (9 symbols):**
```
ES, RTY, 6A, 6C, 6E, 6M, CL, GC, HG
```
- **Equities:** ES (S&P 500), RTY (Russell 2000)
- **Currencies:** 6A (AUD), 6C (CAD), 6E (EUR), 6M (MXN)
- **Commodities:** CL (Crude Oil), GC (Gold), HG (Copper)

### **IBS B (9 symbols):**
```
NQ, YM, 6B, 6J, 6N, 6S, NG, SI, PL
```
- **Equities:** NQ (Nasdaq), YM (Dow Jones)
- **Currencies:** 6B (GBP), 6J (JPY), 6N (NZD), 6S (CHF)
- **Commodities:** NG (Natural Gas), SI (Silver), PL (Platinum)

### **Rationale:**
- Equal 9+9 split for balanced capital allocation
- Both strategies have same asset class distribution (2+4+3)
- Similar risk profiles - both will behave as diversified portfolios
- Neither strategy concentrated in single asset class

---

## Commands to Run Greedy Optimizer

**Location:** `/opt/pine/rooney-capital-v1`

### IBS A - Balanced Mix (9 symbols)
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols ES RTY 6A 6C 6E 6M CL GC HG \
    --output-suffix ibs_a \
    --strategy-name ibs_a \
    --output-dir results
```

**Symbols:** ES, RTY, 6A, 6C, 6E, 6M, CL, GC, HG
**Output:**
- Individual: `results/greedy_optimization_ibs_a_TIMESTAMP.json`
- Consolidated: `results/all_optimizations.json` (updated)

---

### IBS B - Balanced Mix (9 symbols)
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols NQ YM 6B 6J 6N 6S NG SI PL \
    --output-suffix ibs_b \
    --strategy-name ibs_b \
    --output-dir results
```

**Symbols:** NQ, YM, 6B, 6J, 6N, 6S, NG, SI, PL
**Output:**
- Individual: `results/greedy_optimization_ibs_b_TIMESTAMP.json`
- Consolidated: `results/all_optimizations.json` (updated)

---

## Expected Outputs

After running both optimizers, you'll have:

1. **Individual Optimization Results:**
   - `results/greedy_optimization_ibs_a_TIMESTAMP.json`
   - `results/greedy_optimization_ibs_b_TIMESTAMP.json`

2. **Consolidated Results (NEW!):**
   - `results/all_optimizations.json` - All strategies in one file, ranked by test Sharpe

### Consolidated Results Format

The `all_optimizations.json` file contains all strategy optimizations, sorted by test Sharpe (best first):

```json
[
  {
    "strategy_name": "ibs_a",
    "timestamp": "20251108_143022",
    "date": "2025-11-08 14:30:22",
    "train_period": "2023-01-01 to 2023-12-31",
    "test_period": "2024-01-01 to 2024-12-31",
    "candidate_symbols": ["ES", "RTY", "6A", "6C", "6E", "6M", "CL", "GC", "HG"],
    "optimal_symbols": ["ES", "RTY", "6A", "CL", "GC"],
    "n_symbols": 5,
    "max_positions": 2,
    "initial_capital": 150000.0,
    "daily_stop_loss": 2500.0,
    "train_sharpe": 12.45,
    "train_cagr": 0.78,
    "train_max_dd": 4850.0,
    "test_sharpe": 11.23,
    "test_cagr": 0.72,
    "test_max_dd": 4200.0,
    "generalization": 0.902,
    "result_file": "greedy_optimization_ibs_a_20251108_143022.json"
  },
  {
    "strategy_name": "ibs_b",
    ...
  }
]
```

### Benefits

- **Compare All Strategies:** See IBS A, IBS B, Breakout, etc. side-by-side
- **Ranked by Performance:** Sorted by test Sharpe ratio (highest first)
- **Track Over Time:** See how strategies evolve with re-optimization
- **Quick Overview:** All key metrics in one place

---

## Next Steps (After Running Optimizer)

Once both optimizers complete, we'll:

1. **Create portfolio tracking files:**
   - `config/portfolio_optimization_ibs_a.json`
   - `config/portfolio_optimization_ibs_b.json`

2. **Update multi-alpha config:**
   - Add `ibs_a` to `config.multi_alpha.yml`
   - Add `ibs_b` to `config.multi_alpha.yml`

3. **Deploy as separate strategies:**
   - Each with own starting cash ($150k)
   - Each with own daily stop loss ($2,500)
   - Each with own TradersPost webhook
   - Each runs as independent process

---

## Configuration Decisions ✅

All decisions finalized:

1. **Grouping:** Balanced 9+9 split (each gets 2 equities, 4 currencies, 3 commodities)
2. **Train/Test Dates:** 2023 train, 2024 test
3. **Constraints:** max_dd_limit=5000, max_positions=1-4
4. **Capital:** $150k each (independent)
5. **Daily Stop Loss:** $2,500 each (independent)

---

## New Flags Added

The greedy optimizer now supports:

- `--symbols ES NQ CL ...` - Filter to specific symbols only
- `--output-suffix ibs_a` - Add suffix to output filename
- `--strategy-name ibs_a` - Strategy name for consolidated results tracking

These flags enable:
1. Splitting strategies while using the same optimization script
2. Tracking all strategy results in one consolidated file (`results/all_optimizations.json`)
3. Easy comparison across IBS A, IBS B, and future strategies
