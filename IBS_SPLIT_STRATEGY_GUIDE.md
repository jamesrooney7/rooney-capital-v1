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

## Proposed Grouping Options

### Option 1: Equities+Commodities vs Currencies (RECOMMENDED)

**IBS A - Equities & Commodities (10 symbols):**
```
ES, NQ, RTY, YM, CL, NG, GC, SI, HG, PL
```

**IBS B - Currencies (8 symbols):**
```
6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
```

**Rationale:**
- Separates currency risk completely
- IBS A has mix of equity and commodity exposure
- IBS B pure currency play
- Natural diversification between groups

---

### Option 2: Equities+Currencies vs Commodities

**IBS A - Equities & Currencies (12 symbols):**
```
ES, NQ, RTY, YM, 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S
```

**IBS B - Commodities (6 symbols):**
```
CL, NG, GC, SI, HG, PL
```

**Rationale:**
- Separates commodity risk
- IBS A more diversified (larger pool)
- IBS B concentrated on commodities

---

### Option 3: Balanced Mixed Split

**IBS A - Mixed (9 symbols):**
```
ES, NQ, RTY, YM, CL, NG, GC, SI, HG
```

**IBS B - Mixed (9 symbols):**
```
6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S, PL
```

**Rationale:**
- Equal 9+9 split
- Both have some commodity exposure
- Balanced size

---

## Commands to Run Greedy Optimizer

**Location:** `/opt/pine/rooney-capital-v1`

### Option 1: Equities+Commodities vs Currencies

#### IBS A - Equities & Commodities
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols ES NQ RTY YM CL NG GC SI HG PL \
    --output-suffix ibs_a \
    --output-dir results
```

**Output:** `results/greedy_optimization_ibs_a_TIMESTAMP.json`

#### IBS B - Currencies
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols 6A 6B 6C 6E 6J 6M 6N 6S \
    --output-suffix ibs_b \
    --output-dir results
```

**Output:** `results/greedy_optimization_ibs_b_TIMESTAMP.json`

---

### Option 2: Equities+Currencies vs Commodities

#### IBS A - Equities & Currencies
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols ES NQ RTY YM 6A 6B 6C 6E 6J 6M 6N 6S \
    --output-suffix ibs_a \
    --output-dir results
```

#### IBS B - Commodities
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols CL NG GC SI HG PL \
    --output-suffix ibs_b \
    --output-dir results
```

---

### Option 3: Balanced Mixed

#### IBS A - Mixed
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols ES NQ RTY YM CL NG GC SI HG \
    --output-suffix ibs_a \
    --output-dir results
```

#### IBS B - Mixed
```bash
python research/portfolio_optimizer_greedy_train_test.py \
    --results-dir results \
    --train-start 2023-01-01 --train-end 2023-12-31 \
    --test-start 2024-01-01 --test-end 2024-12-31 \
    --min-positions 1 --max-positions 4 \
    --max-dd-limit 5000 \
    --initial-capital 150000 \
    --daily-stop-loss 2500 \
    --symbols 6A 6B 6C 6E 6J 6M 6N 6S PL \
    --output-suffix ibs_b \
    --output-dir results
```

---

## Expected Outputs

After running both optimizers, you'll have:

1. **Optimization Results:**
   - `results/greedy_optimization_ibs_a_TIMESTAMP.json`
   - `results/greedy_optimization_ibs_b_TIMESTAMP.json`

2. **Each file contains:**
   - Optimal instruments list (subset of input symbols)
   - Optimal max_positions (e.g., 2, 3, or 4)
   - Train metrics (Sharpe, CAGR, Max DD)
   - Test metrics (Sharpe, CAGR, Max DD)
   - Generalization score

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

## Questions to Answer

Before running, please choose:

1. **Which grouping option?** (1, 2, or 3)
2. **Same train/test dates?** (2023 train, 2024 test)
3. **Same constraints?** (max_dd_limit=5000, max_positions=1-4)

---

## New Flags Added

The greedy optimizer now supports:

- `--symbols ES NQ CL ...` - Filter to specific symbols only
- `--output-suffix ibs_a` - Add suffix to output filename

These flags enable splitting strategies while using the same optimization script.
