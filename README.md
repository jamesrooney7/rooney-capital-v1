# Rooney Capital Trading Strategy v1.0

Automated futures trading system using IBS strategy with ML optimization.

## Architecture
- **Data Source**: Databento (CME Globex GLBX.MDP3)
- **Strategy**: IBS (Internal Bar Strength) with Random Forest classifier
- **Execution**: TradersPost → Tradovate
- **Symbols**: ES, NQ, RTY, YM, GC, SI, HG, CL, NG, 6A, 6B, 6E

## Structure
src/
├── strategy/     # Trading logic & ML models
├── runner/       # Execution engine
└── models/       # ML models (12 instrument-specific RF models)
deployment/
├── config/       # Contract maps, policies
├── env/          # Environment templates
└── systemd/      # Service definitions

## Development Workflow
1. Edit locally on Mac
2. Commit & push to GitHub
3. Pull on server: `cd /opt/pine/rooney-capital-v1 && git pull`
4. Restart: `sudo systemctl restart pine-runner.service`

## Safety
- Never commit `.env` files with real credentials
- Always test in paper trading first
- Use `POLICY_KILLSWITCH=true` to halt all trading

## ML Models
Each instrument has a dedicated Random Forest model trained on instrument-specific features:
- 6A_rf_model.pkl (21MB) - Australian Dollar
- ES_rf_model.pkl (2B) - E-mini S&P 500
- NQ_rf_model.pkl (1.9MB) - E-mini Nasdaq-100
- ... (12 models total)

Use ``src.models.load_model_bundle`` to hydrate the trained model and
probability threshold when wiring the strategy:

```python
from models import load_model_bundle
from strategy.ibs_strategy import IbsStrategy

bundle = load_model_bundle("ES")
cerebro.addstrategy(IbsStrategy, **bundle.strategy_kwargs())
```

> **Note:** The ``*_rf_model.pkl`` artefacts are stored with Git LFS.  Ensure
> you have pulled the large files (``git lfs pull``) before attempting to load
> the models in a fresh clone.

## Documentation

- [Component overview](docs/components_readme.md) – Detailed walk-through of
  every module, how data flows from Databento into Backtrader, and how
  executions reach TradersPost.
