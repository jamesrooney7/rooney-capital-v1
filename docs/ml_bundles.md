# Machine Learning Model Bundles

The optimisation pipeline in `bin/` exports a **pair of artefacts per instrument**.
You need *both* files in order to hydrate the ML filter used by `IbsStrategy`:

1. `SYMBOL_best.json`
   - JSON blob produced by `bin/rf_tuner.py` (see "Main orchestration" section in the
     script).
   - Captures the tuned hyper-parameters, CPCV diagnostics, production
     probability threshold, and the exact feature column list that the model was
     trained with.

2. `SYMBOL_rf_model.pkl`
   - `joblib` serialisation of the trained `RandomForestClassifier` (and, as a
     fallback, the feature names via `feature_names_in_`).
   - Stored with Git LFS because the payloads can be hundreds of megabytes.

Both files live in `src/models/` by default when produced by the tuning script.
If you run the optimiser elsewhere, place the outputs in a common directory and
point the loader at it via the `base_dir` argument.

## Loading a bundle in code

```python
from models import load_model_bundle
from strategy.ibs_strategy import IbsStrategy

bundle = load_model_bundle("ES")  # looks for ES_best.json & ES_rf_model.pkl
kwargs = bundle.strategy_kwargs()  # {'ml_model': ..., 'ml_features': (...,), 'ml_threshold': 0.62}

cerebro.addstrategy(IbsStrategy, **kwargs)
```

Under the hood, `load_model_bundle`:

1. Normalises the symbol (`es` → `ES`).
2. Reads `SYMBOL_best.json` to recover the production probability threshold and
   canonical feature list.
3. Loads `SYMBOL_rf_model.pkl` with `joblib`, validating that it is not a Git LFS
   pointer. If you see the error `git lfs pointer`, run `git lfs pull` to fetch
   the artefact.
4. Returns a `ModelBundle` dataclass containing the hydrated estimator,
   features, threshold, and the raw metadata dict for any custom logic.

## Quick health check

```python
from models import load_model_bundle

for sym in ["ES", "NQ", "RTY"]:
    bundle = load_model_bundle(sym)
    print(sym, bundle.threshold, len(bundle.features))
```

If any model file is missing or still a Git LFS pointer the loader will raise a
`FileNotFoundError` or `RuntimeError` describing what needs to be pulled.

## Artefacts created by the tuner

When you run `bin/rf_tuner.py` you will additionally find:

- `SYMBOL_rf_best_trades.csv` – trade-by-trade output with model probabilities.
- `SYMBOL_trades.csv` – daily aggregated returns for portfolio stitching.
- `SYMBOL_rf_best_era_table.csv` – per-era diagnostics.
- `SYMBOL_rf_cpcv_random.csv` / `SYMBOL_rf_cpcv_bo.csv` – random search and BO logs.
- `best.json` – summary of the top configuration across all instruments.

Only `SYMBOL_best.json` and `SYMBOL_rf_model.pkl` are required at runtime. The
other files provide offline diagnostics and reporting.
