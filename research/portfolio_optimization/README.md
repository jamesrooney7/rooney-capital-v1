# Portfolio Optimization System

**Equal-weighted portfolio optimizer with risk constraints**

Selects the optimal combination of ML-enhanced strategies to maximize Sharpe ratio while respecting strict risk limits.

## Features

✅ **Simple Position Sizing**: 1 contract per strategy (no complex weighting)
✅ **Risk Constraints**: Max drawdown ≤ $6,000, Daily loss limit ≤ $3,000
✅ **Train/Test Split**: Optimize on 2022-2023, validate on 2024
✅ **Greedy Optimization**: Fast combinatorial search
✅ **Real-time Risk Management**: Emergency shutdown on daily loss limit
✅ **Deployment Ready**: Generates manifest for live trading

---

## Quick Start

### Step 1: Run Portfolio Optimization

After completing ML pipeline for all strategies:

```bash
python research/portfolio_optimization/portfolio_optimizer.py \
    --validation-dir research/ml_meta_labeling/results \
    --max-drawdown 6000 \
    --daily-loss-limit 3000 \
    --output research/portfolio_optimization/portfolio_manifest.json
```

**Output:**
```
GREEDY OPTIMIZATION (2022-2023)
================================================================================
Max Drawdown Constraint: $6,000
Daily Loss Limit: $3,000

Finding best starting strategy...
  ES_21: Sharpe=2.94, DD=$2,100, MaxDailyLoss=$450
  NQ_21: Sharpe=3.21, DD=$2,800, MaxDailyLoss=$520
  ...

Starting with: NQ_21 (Sharpe=3.21)

Iteration 1: Testing 44 candidates...
  ✅ Added ES_21: Sharpe=3.45, DD=$3,200, MaxDailyLoss=$750

Iteration 2: Testing 43 candidates...
  ✅ Added GC_42: Sharpe=3.68, DD=$4,100, MaxDailyLoss=$980

...

OPTIMIZATION COMPLETE
================================================================================
Selected 5 strategies:
  - NQ_21
  - ES_21
  - GC_42
  - CL_45
  - 6E_37

Portfolio Metrics (2022-2023):
  Sharpe Ratio: 3.85
  Max Drawdown: $5,850
  Max Daily Loss: $2,950
  Total Return: $18,450
  Avg Daily P&L: $185
  Win Rate: 68.5%

OUT-OF-SAMPLE VALIDATION (2024)
================================================================================
Portfolio Metrics (2024):
  Sharpe Ratio: 3.22
  Max Drawdown: $4,200
  Max Daily Loss: $2,100
  Total Return: $12,300
  Avg Daily P&L: $125
  Win Rate: 64.2%

✅ Deployment manifest saved to: portfolio_manifest.json
```

### Step 2: Review Deployment Manifest

The optimizer generates a deployment manifest with selected strategies:

```json
{
  "version": "1.0",
  "created_at": "2025-01-22T10:30:00",
  "optimization_config": {
    "max_drawdown": 6000,
    "daily_loss_limit": 3000,
    "optimization_period": "2022-01-01 to 2023-12-31",
    "test_period": "2024-01-01 to 2024-12-31"
  },
  "selected_strategies": [
    {
      "strategy_id": "NQ_21",
      "symbol": "NQ",
      "strategy_number": 21,
      "position_size": 1
    },
    {
      "strategy_id": "ES_21",
      "symbol": "ES",
      "strategy_number": 21,
      "position_size": 1
    }
  ],
  "performance": {
    "optimization_period": {
      "sharpe_ratio": 3.85,
      "max_drawdown": 5850,
      "total_return": 18450
    },
    "test_period": {
      "sharpe_ratio": 3.22,
      "max_drawdown": 4200,
      "total_return": 12300
    }
  }
}
```

---

## Live Trading Integration

### Using the Risk Manager

```python
from research.portfolio_optimization.risk_manager import PortfolioRiskManager

# Initialize risk manager
risk_mgr = PortfolioRiskManager(
    daily_loss_limit=3000,
    max_drawdown_alert=6000,
    shutdown_callbacks=[
        close_all_positions,
        cancel_all_orders,
        send_alert_email
    ]
)

# In your trading loop
while trading:
    # Check if shutdown
    if risk_mgr.is_shutdown:
        logger.warning("Risk manager shutdown - no new trades")
        break

    # Process signals and take trades
    for strategy in active_strategies:
        if strategy.has_signal():
            trade = strategy.execute_trade()

            # Update risk manager with trade P&L
            risk_mgr.update_pnl(
                trade_pnl=trade.pnl,
                strategy_id=strategy.id
            )

    # At end of day
    risk_mgr.print_daily_summary()

    # At start of next day
    risk_mgr.reset_daily()
```

### Emergency Shutdown Flow

When daily loss limit is hit:

1. **Detect**: Daily P&L ≤ -$3,000
2. **Log**: Write critical event to risk log
3. **Execute Callbacks**:
   - Close all open positions
   - Cancel all pending orders
   - Send alerts to trading desk
4. **Block New Trades**: Set `is_shutdown = True`
5. **Wait**: Remains shutdown until `reset_daily()` called

---

## Advanced Options

### Specify Subset of Strategies

```bash
# Only consider specific strategies
python research/portfolio_optimization/portfolio_optimizer.py \
    --strategies ES_21 NQ_21 GC_42 CL_45 \
    --output portfolio_manifest.json
```

### Custom Risk Limits

```bash
# More conservative limits
python research/portfolio_optimization/portfolio_optimizer.py \
    --max-drawdown 4000 \
    --daily-loss-limit 2000 \
    --output conservative_portfolio.json
```

---

## Optimization Algorithm

**Greedy Search with Constraints:**

1. **Start**: Find best single strategy that meets constraints
2. **Iterate**: For each remaining strategy:
   - Test portfolio with added strategy
   - Keep it if Sharpe improves AND constraints still met
3. **Stop**: When no more strategies improve Sharpe

**Complexity**: O(n²) where n = number of strategies
**Runtime**: ~1 minute for 50 strategies

**Why Greedy?**
- Fast: O(n²) vs O(2ⁿ) for exhaustive search
- Effective: Usually finds near-optimal solution
- Transparent: Easy to understand which strategies were added and why

---

## Files

| File | Purpose |
|------|---------|
| `portfolio_optimizer.py` | Main optimization engine |
| `risk_manager.py` | Real-time risk management for live trading |
| `portfolio_manifest.json` | Deployment config (generated) |
| `README.md` | This file |

---

## Integration with ML Pipeline

The portfolio optimizer is the **final step** in the automated pipeline:

```
Phase 1: Strategy Factory
  ↓
Phase 2: Extract Winners (extract_winners.py)
  ↓
Phase 3: ML Meta-Labeling (run_ml_pipeline.py)
  ↓
Phase 4: Portfolio Optimization (portfolio_optimizer.py)  ← YOU ARE HERE
  ↓
Phase 5: Live Trading Deployment
```

**Inputs**: ML validation results from `research/ml_meta_labeling/results/`
**Outputs**: Deployment manifest for live trading

---

## Risk Management Philosophy

### Two-Layer Protection:

**Layer 1: Portfolio Selection (Optimization)**
- Choose strategies that historically kept DD < $6k
- Diversification across instruments and strategy types
- Validation on out-of-sample data (2024)

**Layer 2: Real-Time Monitoring (Risk Manager)**
- Daily loss limit = hard stop ($3k)
- Emergency shutdown with position closing
- Prevents catastrophic losses

### Why $3k Daily / $6k Total?

- **Daily limit** prevents single bad day from destroying account
- **Total DD limit** ensures controlled risk over longer periods
- **1:2 ratio** allows for 2 bad days before hitting total DD
- **Conservative** for futures trading (ES contract ~$50/point)

---

## Example Output

See `portfolio_manifest.json` for deployment configuration.

Risk events logged to: `logs/risk_management/risk_events_YYYY-MM-DD.jsonl`

---

## Next Steps

1. ✅ **Review manifest**: Check selected strategies make sense
2. ✅ **Paper trade**: Test in simulation first
3. ✅ **Monitor closely**: Watch first week carefully
4. ✅ **Scale gradually**: Start with 1 contract, scale up if stable

---

## Support

For issues or questions, contact Rooney Capital trading desk.

**Remember**: This is automated portfolio selection, not automated money printing. Markets change. Monitor carefully. Trade responsibly.
