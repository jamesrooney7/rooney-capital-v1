# ML Bundle and Snapshot Review

## Loader behaviour
- `load_model_bundle` reads `<SYMBOL>_best.json` metadata and prefers the listed `Features` when hydrating strategy kwargs, so the live strategy receives whatever feature names are recorded in the JSON files.

## Metadata feature names
- Example: `6A_best.json` enumerates features such as `open_close`, `secondary_rsi_entry_daily`, `hg_daily_return_pipeline`, and `6m_daily_z_score`. None of these names include the `enable…` prefix used by the live strategy.
- Example: `6E_best.json` includes features like `cl_hourly_return_pipeline`, `pair_ibs_daily`, and `price_z_score_daily`, confirming the metadata expects rich, snake_case feature identifiers for every instrument.

## Snapshot contents in `IbsStrategy`
- `collect_filter_values` records cross-instrument Z-score and return metrics solely under their `enable…` parameter names (e.g., `enable6AZScoreDay`, `enable6AReturnDay`) rather than emitting snake_case signal names.
- The method only emits snake_case keys for a limited set of core indicators such as `prev_day_pct`, `daily_ibs`, and `pair_z`, so the majority of feature names referenced by the metadata never appear in the snapshot.

## rf_model.pkl availability
- The stored `*_rf_model.pkl` artefacts are Git LFS pointers; the loader will refuse to hydrate a model until the binary payloads are fetched (`_is_git_lfs_pointer` guard).

## Impact
- When `_evaluate_ml_score` normalises the snapshot it cannot find metadata feature names such as `6m_daily_z_score` because only the `enable…` variants exist, so every lookup falls back to `_ml_default_for_feature`.

## Next steps
- Extend `collect_filter_values` (and related helpers) to emit snake_case feature keys aligned with the metadata feature list (e.g., `6m_daily_z_score`, `6m_daily_z_pipeline`, `daily_atr_percentile`).
- Audit the metadata feature inventory per instrument to make sure every requested feature can be sourced from live data, updating either the metadata or the snapshot export where necessary.
- Add regression coverage that hydrates a model bundle and asserts every advertised feature appears in the normalised snapshot so future changes keep the artefacts in sync.
