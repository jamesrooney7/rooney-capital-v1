#!/usr/bin/env python3
# Feature screening -> Random Search -> Bayesian Optimization (Optuna) tuner for RF under CPCV.
# Writes best.json (Params + Features + metrics + production threshold) and exports trade selections.

import argparse
import functools
import inspect
import json
import logging
import math
import os
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import joblib


from daily_utils import ensure_daily_index
from feature_utils import build_normalised_feature_matrix


ERA_BINS_2010_2024 = [
    ("2010-01-01", "2012-12-31"),
    ("2013-01-01", "2015-12-31"),
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2024-12-31"),
]

# ========== Metrics ==========
def sharpe_ratio_from_daily(d: pd.Series) -> float:
    d = pd.Series(d).dropna()
    if d.empty: return 0.0
    mu, sd = d.mean(), d.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else (mu/sd)*np.sqrt(252)


def equity_curve_from_daily(daily: pd.Series, initial_equity: float = 100_000.0) -> pd.Series:
    """Return the compounded equity curve from a series of daily returns."""

    daily = pd.Series(daily, dtype=float).dropna()
    if daily.empty:
        return pd.Series(dtype=float)

    if not isinstance(daily.index, pd.DatetimeIndex):
        try:
            daily.index = pd.to_datetime(daily.index)
        except Exception:
            daily = daily.reset_index(drop=True)

    daily = daily.sort_index()
    equity = (1.0 + daily).cumprod() * float(initial_equity)
    equity.name = "Equity"
    return equity


def portfolio_metrics_from_daily(daily: pd.Series, initial_equity: float = 100_000.0) -> dict:
    """Compute portfolio-level diagnostics from daily return series."""

    daily = pd.Series(daily, dtype=float).dropna()
    if not isinstance(daily.index, pd.DatetimeIndex):
        try:
            daily.index = pd.to_datetime(daily.index)
        except Exception:
            daily = daily.reset_index(drop=True)

    daily = daily.sort_index()
    equity_curve = equity_curve_from_daily(daily, initial_equity)

    start_equity = float(initial_equity)
    if equity_curve.empty:
        final_equity = start_equity
        total_return_usd = 0.0
        total_return_pct = 0.0
        cagr = 0.0
        worst_day_pnl = 0.0
        worst_day_return = 0.0
        max_dd_usd = 0.0
        max_dd_pct = 0.0
    else:
        final_equity = float(equity_curve.iloc[-1])
        total_return_usd = final_equity - start_equity
        total_return_pct = total_return_usd / start_equity if start_equity else 0.0

        equity_values = np.concatenate([[start_equity], equity_curve.to_numpy(dtype=float)])
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = running_max - equity_values
        max_dd_usd = float(drawdowns.max()) if drawdowns.size else 0.0
        peak_equity = running_max[np.argmax(drawdowns)] if drawdowns.size else start_equity
        max_dd_pct = (max_dd_usd / peak_equity) if peak_equity else 0.0

        pnl_changes = np.diff(equity_values)
        worst_day_pnl = float(pnl_changes.min()) if pnl_changes.size else 0.0
        worst_day_return = float(daily.min()) if not daily.empty else 0.0

        if isinstance(daily.index, pd.DatetimeIndex) and len(daily) > 1:
            span_days = (daily.index.max() - daily.index.min()).days
            years = span_days / 365.25
            if years > 0 and start_equity > 0:
                cagr = (final_equity / start_equity) ** (1 / years) - 1 if final_equity > 0 else -1.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0

    sharpe = sharpe_ratio_from_daily(daily)
    downside = daily[daily < 0]
    if downside.empty:
        sortino = 0.0
    else:
        downside_dev = downside.std(ddof=0)
        sortino = 0.0 if downside_dev == 0 else (daily.mean() / downside_dev) * np.sqrt(252)

    return {
        "Equity_Curve": equity_curve,
        "Start_Equity": start_equity,
        "Final_Equity": final_equity,
        "Total_Return_USD": total_return_usd,
        "Total_Return_Pct": total_return_pct,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown_USD": max_dd_usd,
        "Max_Drawdown_Pct": max_dd_pct,
        "Worst_Day_PnL_USD": worst_day_pnl,
        "Worst_Day_Return": worst_day_return,
    }


def trade_diagnostics(selected: pd.DataFrame, universe: pd.DataFrame, daily: pd.Series) -> dict:
    """Compute trade-level diagnostics for the selected trades."""

    selected = selected.copy()
    universe = universe.copy()
    win_rate = float((selected["pnl_usd"] > 0).mean()) if not selected.empty else 0.0
    expectancy = float(selected["pnl_usd"].mean()) if not selected.empty else 0.0
    trade_count = int(selected.shape[0])
    total_profit = float(selected["pnl_usd"].sum()) if not selected.empty else 0.0
    pf = profit_factor(selected["pnl_usd"]) if not selected.empty else 0.0

    daily = pd.Series(daily, dtype=float).dropna()
    if isinstance(daily.index, pd.DatetimeIndex) and not daily.empty:
        yearly = daily.groupby(daily.index.year).apply(lambda r: (1.0 + r).prod() - 1.0)
        winning_year_ratio = float((yearly > 0).sum() / len(yearly)) if len(yearly) else 0.0
    else:
        winning_year_ratio = 0.0

    entries = pd.to_datetime(selected.get("Date/Time"), errors="coerce")
    exits = pd.to_datetime(selected.get("Exit Date/Time"), errors="coerce")
    durations = (exits - entries).dropna()
    durations = durations[durations >= pd.Timedelta(0)]
    avg_duration = durations.mean() if not durations.empty else pd.Timedelta(0)

    avg_hours = float(avg_duration / pd.Timedelta(hours=1)) if avg_duration is not pd.NaT else 0.0
    avg_days = float(avg_duration / pd.Timedelta(days=1)) if avg_duration is not pd.NaT else 0.0

    intervals = []
    for start, end in zip(entries, exits):
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            continue
        intervals.append((start, end))
    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    total_time_seconds = sum((end - start).total_seconds() for start, end in merged)

    all_entries = pd.to_datetime(universe.get("Date/Time"), errors="coerce")
    all_exits = pd.to_datetime(universe.get("Exit Date/Time"), errors="coerce") if "Exit Date/Time" in universe.columns else pd.Series(dtype="datetime64[ns]")
    bt_start = all_entries.min()
    bt_end = all_entries.max()
    if not all_exits.empty:
        bt_end = max(bt_end, all_exits.max()) if bt_end is not pd.NaT else all_exits.max()

    if pd.isna(bt_start) or pd.isna(bt_end) or bt_end <= bt_start:
        time_in_market_pct = 0.0
    else:
        span_seconds = (bt_end - bt_start).total_seconds()
        time_in_market_pct = (total_time_seconds / span_seconds) * 100.0 if span_seconds > 0 else 0.0

    return {
        "Win_Rate": win_rate,
        "Profit_Factor": float(pf),
        "Expectancy_USD": expectancy,
        "Trade_Count": trade_count,
        "Total_Profit_USD": total_profit,
        "Winning_Year_Ratio": winning_year_ratio,
        "Average_Hold_Timedelta": str(avg_duration),
        "Average_Hold_Hours": avg_hours,
        "Average_Hold_Days": avg_days,
        "Time_In_Market_Pct": float(time_in_market_pct),
    }

def profit_factor(tr: pd.Series) -> float:
    tr = pd.Series(tr).dropna()
    if tr.empty: return 0.0
    pos, neg = tr[tr>0].sum(), -tr[tr<0].sum()
    if neg == 0: return np.inf if pos>0 else 0.0
    return pos/neg

def _norm_ppf(p: float) -> float:
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    plow=0.02425; phigh=1-plow
    if p<=0 or p>=1: return np.nan
    if p<plow:
        q=np.sqrt(-2*np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p>phigh:
        q=np.sqrt(-2*np.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+1))

def _psr(sr: float, n: int, kurt_excess: float = 0.0, sr_bench: float = 0.0) -> float:
    if n<=1: return np.nan
    variance = 1 + 0.25*kurt_excess*(sr**2)
    if variance<=0: return np.nan
    z = (sr - sr_bench) * np.sqrt(n - 1) / np.sqrt(variance)
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))

def deflated_sharpe_ratio(sr: float, n: int, kurt_excess: float = 0.0, m: int = 1) -> float:
    sr_star = _norm_ppf(1 - 1.0/m) / np.sqrt(max(n - 1, 1)) if m>1 else 0.0
    return _psr(sr, n, kurt_excess, sr_star)

def build_core_features(Xy: pd.DataFrame) -> pd.DataFrame:
    return build_normalised_feature_matrix(Xy)

def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    def col(name):
        if f"{name}_pct" in Xn.columns: return f"{name}_pct"
        return name if name in Xn.columns else None
    # lightweight interactions that helped earlier
    for a,b in [("ibs","atrz"),("ibs","volz"),("rsi","volz"),("rsi","atrz"),
                ("prev_day_pct","value"),("daily_rsi","value")]:
        ca, cb = col(a), col(b)
        if ca and cb: Xn[f"{a}x{b}"] = Xn[ca] * Xn[cb]
    return Xn

# ========== CPCV ==========
def embargoed_cpcv_splits(dates, n_folds=5, k_test=2, embargo_days=2):
    d = pd.to_datetime(dates).dt.date
    unique_dates = np.array(sorted(pd.Series(d).unique()))
    split_points = np.linspace(0, len(unique_dates), n_folds + 1, dtype=int)
    folds = [unique_dates[split_points[i]:split_points[i+1]] for i in range(n_folds)]
    d_ord = pd.to_datetime(d).map(lambda x: x.toordinal()).to_numpy()
    for test_idx in combinations(range(n_folds), k_test):
        test_dates = np.concatenate([folds[i] for i in test_idx])
        if test_dates.size==0: continue
        test_ord = np.array([pd.to_datetime(td).toordinal() for td in test_dates])
        dist = np.min(np.abs(d_ord[:, None] - test_ord[None, :]), axis=1)
        te_mask = np.isin(d, test_dates)
        tr_mask = (~te_mask) & (dist > embargo_days)
        yield tr_mask, te_mask

# ========== Feature screening ==========
logger = logging.getLogger(__name__)


def _fold_importances(method: str, X_tr: pd.DataFrame, y_tr: pd.Series, seed: int, fold_idx: int) -> pd.Series:
    """Return feature importances for a single CPCV fold."""

    rng_seed = seed + fold_idx
    if method == "mdi":
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=100,
            max_features="sqrt",
            n_jobs=-1,
            random_state=rng_seed,
        )
        rf.fit(X_tr, y_tr)
        return pd.Series(rf.feature_importances_, index=X_tr.columns)

    if method == "permutation":
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=rng_seed,
        )
        rf.fit(X_tr, y_tr)
        perm = permutation_importance(
            rf,
            X_tr,
            y_tr,
            n_repeats=10,
            random_state=rng_seed,
            n_jobs=1,
        )
        return pd.Series(np.maximum(perm.importances_mean, 0.0), index=X_tr.columns)

    if method == "l1":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tr)
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            random_state=rng_seed,
            class_weight="balanced",
        )
        clf.fit(X_scaled, y_tr)
        coefs = np.abs(clf.coef_)
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        return pd.Series(coefs, index=X_tr.columns)

    raise ValueError(f"Unsupported screen_method: {method}")


def _clustered_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_clusters: int = 15,
    features_per_cluster: int = 2,
    max_correlation: float = 0.7,
):
    """Select diverse features using correlation-based clustering.

    Algorithm:
    1. Calculate correlation matrix of all features
    2. Cluster features by correlation (hierarchical clustering)
    3. For each cluster, select top K features by permutation importance
    4. Validate diversity (ensure selected features have correlation < max_correlation)

    Args:
        X: Feature matrix
        y: Target labels
        seed: Random seed
        n_clusters: Number of clusters to create (default: 15)
        features_per_cluster: Features to select per cluster (default: 2)
        max_correlation: Maximum allowed correlation between selected features (default: 0.7)

    Returns:
        List of selected feature names
    """
    logger.info(f"Starting clustered feature selection: {len(X.columns)} features → {n_clusters} clusters")

    # Step 0: Clean data - remove problematic features
    logger.info("  Step 0/4: Cleaning features...")
    # Remove features with zero variance
    variances = X.var()
    valid_features = variances[variances > 1e-10].index.tolist()
    X_clean = X[valid_features].copy()

    # Replace any inf/nan with 0
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"   Removed {len(X.columns) - len(valid_features)} zero-variance features, {len(valid_features)} remaining")

    # Step 1: Calculate correlation matrix
    logger.info("  Step 1/4: Calculating correlation matrix...")
    corr_matrix = X_clean.corr().abs()  # Absolute correlation

    # Replace any NaN correlations with 0 (uncorrelated)
    corr_matrix = corr_matrix.fillna(0)

    # Step 2: Hierarchical clustering based on correlation distance
    logger.info("  Step 2/4: Performing hierarchical clustering...")
    # Distance = 1 - correlation (higher correlation = lower distance)
    distance_matrix = 1 - corr_matrix

    # Ensure all values are finite
    distance_matrix = distance_matrix.clip(lower=0, upper=2)  # Distance should be in [0, 2]

    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix.values, checks=False)
    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    # Cut tree to get clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Group features by cluster (use X_clean columns, not original X)
    clusters = {}
    for feature, cluster_id in zip(X_clean.columns, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(feature)

    logger.info(f"  Created {len(clusters)} clusters (sizes: {[len(c) for c in clusters.values()]})")

    # Step 3: Select best features from each cluster using permutation importance
    logger.info("  Step 3/4: Selecting features from each cluster...")
    selected_features = []

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )

    for cluster_id, cluster_features in clusters.items():
        if len(cluster_features) == 0:
            continue

        # Get permutation importance for features in this cluster
        X_cluster = X_clean[cluster_features]

        try:
            rf.fit(X_cluster, y)
            perm = permutation_importance(
                rf,
                X_cluster,
                y,
                n_repeats=5,  # Fewer repeats for speed (vs 10 for full permutation)
                random_state=seed,
                n_jobs=1,
            )
            importances = pd.Series(
                np.maximum(perm.importances_mean, 0.0),
                index=cluster_features
            ).sort_values(ascending=False)

            # Select top K from this cluster
            top_k = min(features_per_cluster, len(importances))
            cluster_selected = importances.head(top_k).index.tolist()
            selected_features.extend(cluster_selected)

            logger.info(
                f"    Cluster {cluster_id}: {len(cluster_features)} features → "
                f"selected {cluster_selected} (importance: {importances[cluster_selected].values})"
            )

        except Exception as e:
            logger.warning(f"    Cluster {cluster_id}: Failed to evaluate - {e}")
            # Fallback: just take first feature from cluster
            selected_features.append(cluster_features[0])

    # Step 4: Validate diversity (remove highly correlated features)
    logger.info("  Step 4/4: Validating diversity...")
    final_features = []
    for feature in selected_features:
        # Check correlation with already-selected features
        if not final_features:
            final_features.append(feature)
            continue

        # Calculate max correlation with existing features
        existing_corrs = corr_matrix.loc[feature, final_features].values
        max_corr = np.max(existing_corrs)

        if max_corr < max_correlation:
            final_features.append(feature)
        else:
            logger.debug(
                f"    Rejected {feature}: max_corr={max_corr:.3f} >= {max_correlation:.3f}"
            )

    logger.info(
        f"✅ Clustered selection complete: {len(final_features)} diverse features "
        f"(from {len(selected_features)} candidates)"
    )

    # Log correlation statistics
    if len(final_features) > 1:
        final_corr_matrix = corr_matrix.loc[final_features, final_features]
        # Get upper triangle (exclude diagonal)
        upper_triangle = final_corr_matrix.where(
            np.triu(np.ones(final_corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_triangle.stack().mean()
        max_pair_corr = upper_triangle.stack().max()
        logger.info(
            f"   Diversity stats: avg_corr={avg_corr:.3f}, max_corr={max_pair_corr:.3f}"
        )

    return final_features


def screen_features(
    Xy,
    X,
    seed,
    method="importance",
    folds=5,
    k_test=2,
    embargo_days=2,
    top_n=None,
    n_clusters=15,
    features_per_cluster=2,
):
    """Rank features via the requested screening method and keep the top set.

    Methods:
    - mdi/importance: Mean Decrease Impurity (fast, traditional)
    - permutation: Permutation importance (slower, more robust)
    - l1: L1 regularization
    - clustered: Correlation-based clustering with diverse selection (expert-recommended)
    - none: Use all features

    For traditional methods (mdi/permutation/l1):
    The per-fold screening keeps at most ``top_n`` features per CPCV split, logs the
    retained names with their importance scores, and then aggregates by taking the
    union of all kept columns.  The final selection averages the scores for each
    feature across the folds where it survived and returns the top ``top_n`` global
    features.

    For clustered method:
    Uses correlation-based clustering to group similar features, then selects best
    features from each cluster to ensure diversity (no redundant features).
    """

    method = (method or "importance").lower()
    method = {"importance": "mdi"}.get(method, method)

    if method == "none":
        return list(X.columns)

    # Handle clustered method separately (doesn't use per-fold aggregation)
    if method == "clustered":
        logger.info("Using CLUSTERED feature selection (correlation-based)")
        # Use first fold's data for clustering (or could aggregate, but clustering is deterministic)
        for fold_idx, (tr_mask, _te_mask) in enumerate(
            embargoed_cpcv_splits(Xy["Date"], n_folds=folds, k_test=k_test, embargo_days=embargo_days),
            start=1,
        ):
            X_tr = X.loc[tr_mask]
            y_tr = Xy.loc[tr_mask, "y_binary"]
            if X_tr.empty or y_tr.nunique() < 2:
                continue

            # Run clustered selection on first valid fold
            try:
                selected = _clustered_feature_selection(
                    X_tr,
                    y_tr,
                    seed,
                    n_clusters=n_clusters,
                    features_per_cluster=features_per_cluster,
                    max_correlation=0.7,
                )
                logger.info(f"Clustered selection retained {len(selected)} features")
                return selected
            except Exception as e:
                logger.error(f"Clustered selection failed: {e}")
                logger.info("Falling back to MDI importance method")
                method = "mdi"  # Fallback
                break

    if method not in {"mdi", "permutation", "l1"}:
        raise ValueError(f"Unsupported screen_method: {method}")

    if top_n is None or top_n <= 0:
        top_n = X.shape[1]

    per_split_cap = min(top_n, X.shape[1])

    agg_scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    splits_used = 0

    for fold_idx, (tr_mask, _te_mask) in enumerate(
        embargoed_cpcv_splits(Xy["Date"], n_folds=folds, k_test=k_test, embargo_days=embargo_days),
        start=1,
    ):
        X_tr = X.loc[tr_mask]
        y_tr = Xy.loc[tr_mask, "y_binary"]
        if X_tr.empty or y_tr.nunique() < 2:
            continue

        try:
            scores = _fold_importances(method, X_tr, y_tr, seed, fold_idx)
        except Exception:
            continue

        if scores.empty:
            continue

        scores = scores.astype(float)
        ordered = scores.sort_values(ascending=False).head(per_split_cap)
        splits_used += 1

        log_msg = ", ".join(f"{feat}:{val:.4f}" for feat, val in ordered.items())
        logger.info(
            "Screen split %d using %s kept %d/%d features: %s",
            fold_idx,
            method,
            len(ordered),
            per_split_cap,
            log_msg,
        )

        for feat, val in ordered.items():
            agg_scores[feat] = agg_scores.get(feat, 0.0) + float(val)
            counts[feat] = counts.get(feat, 0) + 1

    if not agg_scores or splits_used == 0:
        return list(X.columns)

    averaged = {feat: agg_scores[feat] / counts[feat] for feat in agg_scores}
    final = pd.Series(averaged).sort_values(ascending=False)
    keep = min(per_split_cap, len(final))
    selected = final.head(keep).index.tolist()
    logger.info("Aggregated screening retained %d features: %s", len(selected), ", ".join(selected))
    return selected

# ========== Evaluation ==========
def _cpcv_evaluate(
    Xy,
    X,
    rf_params,
    folds=5,
    k_test=2,
    embargo_days=2,
    min_train=200,
    min_test=50,
    thr_grid=None,
    *,
    fixed_thr=None,
    collect_details=False,
    n_trials_total=1,
):
    if thr_grid is None:
        thr_grid = list(np.round(np.arange(0.45, 0.75 + 1e-9, 0.05), 2))

    n_rows = X.shape[0]
    prob_sum = np.zeros(n_rows, dtype=float)
    vote_count = np.zeros(n_rows, dtype=int)
    chosen = []
    fold_details = []

    params = rf_params.copy()
    if not params.get("bootstrap", False):
        params["max_samples"] = None

    for fold_idx, (tr_mask, te_mask) in enumerate(
        embargoed_cpcv_splits(Xy["Date"], folds, k_test, embargo_days),
        start=1,
    ):
        if tr_mask.sum() < min_train or te_mask.sum() < min_test:
            continue

        y_tr = Xy.loc[tr_mask, "y_binary"]
        if y_tr.nunique() < 2:
            vote_count[te_mask] += 1
            continue

        rf = RandomForestClassifier(**params)
        rf.fit(X.loc[tr_mask], y_tr)

        if fixed_thr is None:
            p_tr = rf.predict_proba(X.loc[tr_mask])[:, 1]
            best_sr, best_thr = -1e9, 0.5
            for thr in thr_grid:
                sel = p_tr >= thr
                if sel.sum() < 30:
                    continue
                dr = pd.DataFrame(
                    {
                        "d": Xy.loc[tr_mask, "Date"],
                        "r": Xy.loc[tr_mask, "y_return"].where(sel, 0.0),
                    }
                ).groupby(pd.Grouper(key="d", freq="D"))["r"].sum()
                dr = ensure_daily_index(dr, Xy.loc[tr_mask, "Date"])
                sr = sharpe_ratio_from_daily(dr)
                if sr > best_sr:
                    best_sr, best_thr = sr, thr
        else:
            best_thr = float(fixed_thr)

        chosen.append(best_thr)

        p_te = rf.predict_proba(X.loc[te_mask])[:, 1]
        prob_sum[te_mask] += p_te
        vote_count[te_mask] += 1

        if collect_details:
            sel = p_te >= best_thr
            holdout_returns = Xy.loc[te_mask, "y_return"].astype(float)
            holdout_dates = Xy.loc[te_mask, "Date"]
            daily_fold = (
                pd.DataFrame({"d": holdout_dates, "r": holdout_returns.where(sel, 0.0)})
                .groupby(pd.Grouper(key="d", freq="D"))["r"]
                .sum()
            )
            daily_fold = ensure_daily_index(daily_fold, holdout_dates)
            fold_port = portfolio_metrics_from_daily(daily_fold)
            fold_details.append(
                {
                    "Fold": fold_idx,
                    "Sharpe": float(fold_port.get("Sharpe", 0.0)),
                    "Sortino": float(fold_port.get("Sortino", 0.0)),
                    "Max_Drawdown_Pct": float(fold_port.get("Max_Drawdown_Pct", 0.0)),
                    "Max_Drawdown_USD": float(fold_port.get("Max_Drawdown_USD", 0.0)),
                    "Trades": int(sel.sum()),
                    "Threshold": float(best_thr),
                    "Rows": int(te_mask.sum()),
                }
            )

    if not np.any(vote_count):
        empty = {
            "Sharpe": 0.0,
            "DSR": 0.0,
            "PF": 0.0,
            "Trades": 0,
            "Thr": float(fixed_thr) if fixed_thr is not None else 0.5,
            "Total_PnL_USD": 0.0,
            "Era_Positive": False,
            "Era_Positive_Count": 0,
            "Era_Count": 0,
            "Sortino": 0.0,
            "Max_Drawdown_USD": 0.0,
            "Max_Drawdown_Pct": 0.0,
        }
        if collect_details:
            return empty, {
                "probabilities": np.full(n_rows, np.nan, dtype=float),
                "valid_mask": np.zeros(n_rows, dtype=bool),
                "selected_mask": np.zeros(n_rows, dtype=bool),
                "vote_count": vote_count,
                "fold_metrics": fold_details,
                "thresholds": chosen,
            }
        return empty

    valid_idx = vote_count > 0
    aggregated_prob = np.full(n_rows, np.nan, dtype=float)
    aggregated_prob[valid_idx] = prob_sum[valid_idx] / vote_count[valid_idx]

    if fixed_thr is None:
        thr = float(np.median(chosen)) if chosen else 0.5
    else:
        thr = float(fixed_thr)

    selected_mask = np.zeros(n_rows, dtype=bool)
    selected_mask[valid_idx] = aggregated_prob[valid_idx] >= thr

    valid_returns = Xy.loc[valid_idx, "y_return"].astype(float)
    selected_valid = pd.Series(selected_mask[valid_idx], index=Xy.index[valid_idx])
    daily = (
        pd.DataFrame(
            {"d": Xy.loc[valid_idx, "Date"], "r": valid_returns.where(selected_valid, 0.0)}
        )
        .groupby(pd.Grouper(key="d", freq="D"))["r"]
        .sum()
    )
    daily = ensure_daily_index(daily, Xy.loc[valid_idx, "Date"])
    trades_usd = Xy.loc[selected_mask, "y_pnl_usd"].astype(float).to_numpy()

    bins = ERA_BINS_2010_2024
    era_flags = []
    for a, b in bins:
        a_ts = pd.to_datetime(a)
        b_ts = pd.to_datetime(b)
        m = (daily.index >= a_ts) & (daily.index <= b_ts)
        if m.any():
            era_flags.append(sharpe_ratio_from_daily(daily[m]) > 0)
    era_positive_count = int(np.sum(era_flags))
    era_count = int(len(era_flags))
    era_positive = False
    if era_count:
        era_positive = (era_positive_count / era_count) >= 0.8

    sr = sharpe_ratio_from_daily(daily)

    # Calculate effective number of trials for multiple testing correction
    # Account for correlation between trials (models are similar - same data, different hyperparams)
    # Conservative estimate: rho_avg ≈ 0.7 for correlated trials
    rho_avg = 0.7
    n_effective = max(1, int(n_trials_total / (1 + (n_trials_total - 1) * rho_avg)))

    # Deflated Sharpe Ratio with proper multiple testing correction
    dsr = deflated_sharpe_ratio(sr, n=daily.shape[0], kurt_excess=daily.kurt(), m=n_effective)
    pf = profit_factor(pd.Series(trades_usd))
    total_profit = float(pd.Series(trades_usd).dropna().sum())

    portfolio = portfolio_metrics_from_daily(daily)

    summary = {
        "Sharpe": sr,
        "DSR": dsr,
        "PF": pf,
        "Trades": int(selected_mask.sum()),
        "Thr": thr,
        "Total_PnL_USD": total_profit,
        "Era_Positive": era_positive,
        "Era_Positive_Count": era_positive_count,
        "Era_Count": era_count,
        "Sortino": float(portfolio.get("Sortino", 0.0)),
        "Max_Drawdown_USD": float(portfolio.get("Max_Drawdown_USD", 0.0)),
        "Max_Drawdown_Pct": float(portfolio.get("Max_Drawdown_Pct", 0.0)),
    }

    if not collect_details:
        return summary

    details = {
        "probabilities": aggregated_prob,
        "valid_mask": valid_idx,
        "selected_mask": selected_mask,
        "vote_count": vote_count,
        "fold_metrics": fold_details,
        "thresholds": chosen,
        "daily": daily,
    }
    return summary, details


def evaluate_rf_cpcv(
    Xy,
    X,
    rf_params,
    folds=5,
    k_test=2,
    embargo_days=2,
    min_train=200,
    min_test=50,
    thr_grid=None,
    n_trials_total=1,
):
    return _cpcv_evaluate(
        Xy,
        X,
        rf_params,
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        min_train=min_train,
        min_test=min_test,
        thr_grid=thr_grid,
        n_trials_total=n_trials_total,
    )

# ========== Random search space ==========
def sample_rf_params(rng: random.Random):
    bootstrap = rng.choice([True, False])
    params = {
        "n_estimators": rng.choice([300, 600, 900, 1200]),
        "max_depth": rng.choice([3, 5, 7, None]),
        "min_samples_leaf": rng.choice([50, 100, 200]),
        "max_features": rng.choice(["sqrt", "log2", 0.3, 0.5]),
        "bootstrap": bootstrap,
        "class_weight": rng.choice([None, "balanced_subsample"]),
        "n_jobs": -1,
        "random_state": rng.randint(1, 10_000)
    }
    params["max_samples"] = rng.choice([0.7, 0.8, 0.9]) if bootstrap else None
    return params


def _with_q_param(candidates_func, bo_batch):
    if bo_batch is None or bo_batch <= 1:
        return candidates_func
    try:
        sig = inspect.signature(candidates_func)
    except (TypeError, ValueError):  # pragma: no cover - fallback for built-ins
        return candidates_func
    if "q" not in sig.parameters:
        return candidates_func
    return functools.partial(candidates_func, q=bo_batch)


def make_bo_sampler(optuna_module, seed, bo_acq, bo_batch, turbo):
    """Return an Optuna sampler honoring the BO configuration."""

    sampler = None
    acq = (bo_acq or "tpe").lower()

    if acq in {"qei", "ei", "qpi", "pi", "qucb", "ucb"} or turbo:
        try:
            from optuna.integration import botorch
        except Exception as exc:  # pragma: no cover - optional dependency
            warnings.warn(
                f"BoTorch sampler unavailable ({exc}); falling back to TPE.")
        else:
            target = {
                "qei": "qEI",
                "ei": "qEI",
                "qpi": "qPI",
                "pi": "qPI",
                "qucb": "qUCB",
                "ucb": "qUCB",
            }.get(acq)
            candidates_func = getattr(botorch, target, None)
            if candidates_func is not None:
                candidates_func = _with_q_param(candidates_func, bo_batch)
            sampler_kwargs = {}
            sampler_sig = inspect.signature(botorch.BoTorchSampler)
            if "seed" in sampler_sig.parameters:
                sampler_kwargs["seed"] = seed
            if turbo and "use_turbo" in sampler_sig.parameters:
                sampler_kwargs["use_turbo"] = True
            if candidates_func is not None and "candidates_func" in sampler_sig.parameters:
                sampler_kwargs["candidates_func"] = candidates_func
            try:
                sampler = botorch.BoTorchSampler(**sampler_kwargs)
            except Exception as exc:  # pragma: no cover - runtime guard
                warnings.warn(
                    f"Failed to initialise BoTorchSampler ({exc}); using TPE instead."
                )
                sampler = None

    if sampler is not None:
        return sampler

    if acq in {"random", "uniform"}:
        sampler_cls = getattr(optuna_module.samplers, "RandomSampler")
        return sampler_cls(seed=seed)
    if acq in {"cmaes", "cma-es"}:
        sampler_cls = getattr(optuna_module.samplers, "CmaEsSampler")
        return sampler_cls(seed=seed)

    sampler_cls = getattr(optuna_module.samplers, "TPESampler")
    sampler_sig = inspect.signature(sampler_cls)
    sampler_kwargs = {"seed": seed}
    if "multivariate" in sampler_sig.parameters:
        sampler_kwargs["multivariate"] = True
    if "group" in sampler_sig.parameters:
        sampler_kwargs["group"] = True
    if bo_batch and bo_batch > 1 and "constant_liar" in sampler_sig.parameters:
        sampler_kwargs["constant_liar"] = True
    return sampler_cls(**sampler_kwargs)

# ========== Export selected trades with fixed threshold ==========
def export_selected_trades(
    Xy,
    X,
    params,
    prod_thr,
    folds,
    k_test,
    embargo_days,
    symbol,
    outdir,
    *,
    precomputed=None,
    min_train=200,
    min_test=50,
):
    fixed_thr = None if prod_thr is None else float(prod_thr)
    if precomputed is None:
        summary, details = _cpcv_evaluate(
            Xy,
            X,
            params,
            folds=folds,
            k_test=k_test,
            embargo_days=embargo_days,
            min_train=min_train,
            min_test=min_test,
            fixed_thr=fixed_thr,
            collect_details=True,
        )
    else:
        summary, details = precomputed
        if fixed_thr is not None and not np.isclose(summary.get("Thr", fixed_thr), fixed_thr):
            summary = summary.copy()
            summary["Thr"] = fixed_thr

    probs = details.get("probabilities")
    valid_mask = pd.Series(details.get("valid_mask"), index=X.index)
    selected_mask = pd.Series(details.get("selected_mask"), index=X.index)
    vote_count = pd.Series(details.get("vote_count"), index=X.index)

    base_dates = pd.to_datetime(Xy["Date/Time"], errors="coerce")
    exit_dates = (
        pd.to_datetime(Xy["Exit Date/Time"], errors="coerce")
        if "Exit Date/Time" in Xy.columns
        else pd.Series(pd.NaT, index=Xy.index)
    )

    core = pd.DataFrame(index=Xy.index)
    core["Date/Time"] = base_dates
    core["Exit Date/Time"] = exit_dates
    if "Date" in Xy.columns:
        core["Date"] = pd.to_datetime(Xy["Date"], errors="coerce").dt.date
    else:
        core["Date"] = base_dates.dt.date
    core["symbol"] = symbol
    core["prob"] = probs
    core["selected"] = selected_mask.astype(int)
    core["threshold"] = summary.get("Thr", prod_thr if prod_thr is not None else 0.0)
    core["votes"] = vote_count.fillna(0).astype(int)
    core["pnl"] = pd.to_numeric(Xy["y_return"], errors="coerce")
    core["pnl_when_selected"] = np.where(core["selected"] == 1, core["pnl"], 0.0)
    core["pnl_usd"] = pd.to_numeric(Xy["y_pnl_usd"], errors="coerce")
    core["pnl_usd_when_selected"] = np.where(core["selected"] == 1, core["pnl_usd"], 0.0)

    rename_labels = {
        "y_return": "Label_Return",
        "y_binary": "Label_Binary",
        "y_pnl_usd": "Label_PnL_USD",
    }
    detailed = Xy.rename(columns={k: v for k, v in rename_labels.items() if k in Xy.columns}).copy()

    def _assign_or_insert(df: pd.DataFrame, loc: int, column: str, values) -> None:
        if column in df.columns:
            df[column] = values
            current_loc = df.columns.get_loc(column)
            if current_loc != loc:
                series = df.pop(column)
                df.insert(loc, column, series)
        else:
            df.insert(loc, column, values)

    _assign_or_insert(detailed, 0, "Symbol", symbol)
    _assign_or_insert(detailed, 1, "Strategy", "rf_cpcv_random_then_bo")
    _assign_or_insert(detailed, 2, "Model", "RandomForestClassifier")
    _assign_or_insert(detailed, 3, "Trade_ID", np.arange(1, len(detailed) + 1))
    detailed["Entry_Date"] = base_dates.dt.date
    detailed["Exit_Date"] = exit_dates.dt.date
    used_thr = float(summary.get("Thr", prod_thr if prod_thr is not None else 0.0))
    detailed["Model_Probability"] = probs
    detailed["Model_Selected"] = selected_mask.astype(int)
    detailed["Model_Threshold"] = used_thr
    detailed["prob"] = probs
    detailed["selected"] = selected_mask.astype(int)
    detailed["threshold"] = used_thr
    detailed["Model_Vote_Count"] = vote_count.fillna(0).astype(int)
    detailed["Model_Return"] = core["pnl"]
    detailed["Model_Return_When_Selected"] = core["pnl_when_selected"]
    detailed["Model_PnL_USD"] = core["pnl_usd"]
    detailed["Model_PnL_USD_When_Selected"] = core["pnl_usd_when_selected"]
    if "Label_Binary" in detailed.columns:
        detailed["Model_Label_Binary"] = pd.to_numeric(detailed["Label_Binary"], errors="coerce")
    detailed["pnl"] = core["pnl"]
    detailed["pnl_when_selected"] = core["pnl_when_selected"]
    detailed["pnl_usd"] = core["pnl_usd"]
    detailed["pnl_usd_when_selected"] = core["pnl_usd_when_selected"]

    feature_cols = [c for c in X.columns if c not in {"Date", "Date/Time"}]
    if feature_cols:
        features_prefixed = X[feature_cols].copy()
        features_prefixed.columns = [f"Feature__{col}" for col in features_prefixed.columns]
        detailed = detailed.join(features_prefixed, how="left")

    valid_mask = pd.Series(details.get("valid_mask"), index=core.index)
    daily = core.loc[valid_mask].groupby("Date")["pnl_when_selected"].sum()
    daily = ensure_daily_index(daily, core.loc[valid_mask, "Date"])
    portfolio = portfolio_metrics_from_daily(daily)
    equity_curve = portfolio.get("Equity_Curve", pd.Series(dtype=float))
    selected_trades = core.loc[core["selected"]==1].copy()
    trade_stats = trade_diagnostics(selected_trades, core, daily)
    total_profit = trade_stats.get("Total_Profit_USD", 0.0)
    n = int(trade_stats.get("Trade_Count", 0))

    bins = ERA_BINS_2010_2024
    rows=[]
    for a,b in bins:
        era_start = pd.to_datetime(a).date()
        era_end = pd.to_datetime(b).date()
        m = (core["Date"]>=era_start) & (core["Date"]<=era_end)
        mv = m & valid_mask
        daily_e = core.loc[mv].groupby("Date")["pnl_when_selected"].sum()
        daily_e = ensure_daily_index(daily_e, core.loc[mv, "Date"])
        sr_e = sharpe_ratio_from_daily(daily_e)
        pf_e = profit_factor(core.loc[m & (core["selected"]==1), "pnl_usd"])
        n_e  = int((m & (core["selected"]==1)).sum())
        rows.append({"Era": f"{a[:4]}–{b[:4]}", "Sharpe": sr_e, "Profit_Factor": pf_e, "Trades": n_e})
    era_df = pd.DataFrame(rows)

    # detailed per-trade output
    trades_csv = os.path.join(outdir, f"{symbol}_rf_best_trades.csv")
    detailed.to_csv(trades_csv, index=False)

    # daily returns file for portfolio aggregation
    daily_df = daily.rename_axis("Date").reset_index().rename(columns={"pnl_when_selected": "Return"})
    daily_df["Date"] = pd.to_datetime(daily_df["Date"]).dt.date
    daily_df.insert(1, "Symbol", symbol)
    simple_csv = os.path.join(outdir, f"{symbol}_trades.csv")
    daily_df.to_csv(simple_csv, index=False)

    era_csv = os.path.join(outdir, f"{symbol}_rf_best_era_table.csv")
    era_df.to_csv(era_csv, index=False)
    summary_path = os.path.join(outdir, f"{symbol}_rf_best_summary.txt")
    portfolio_for_json = {k: v for k, v in portfolio.items() if k != "Equity_Curve"}
    equity_curve_records = [
        {"Date": (idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)), "Equity": float(val)}
        for idx, val in (equity_curve.items() if hasattr(equity_curve, "items") else [])
    ]

    def _clean_numeric(val):
        if isinstance(val, (np.floating, float, np.integer, int)):
            return None if pd.isna(val) else float(val)
        return val

    portfolio_json = {k: _clean_numeric(v) for k, v in portfolio_for_json.items()}
    portfolio_json["Equity_Curve"] = equity_curve_records

    trade_json = {k: _clean_numeric(v) if k not in {"Average_Hold_Timedelta"} else v for k, v in trade_stats.items()}

    with open(summary_path, "w") as f:
        json.dump({
            "Symbol": symbol,
            "Prod_Threshold": float(prod_thr) if prod_thr is not None else used_thr,
            "Sharpe_OOS_CPCV": float(portfolio.get("Sharpe", 0.0)),
            "Sortino_OOS_CPCV": float(portfolio.get("Sortino", 0.0)),
            "Max_Drawdown_Pct": float(portfolio.get("Max_Drawdown_Pct", 0.0)),
            "Max_Drawdown_USD": float(portfolio.get("Max_Drawdown_USD", 0.0)),
            "Profit_Factor": float(trade_stats.get("Profit_Factor", 0.0)),
            "Total_Profit_USD": float(total_profit),
            "Trades_Selected": int(n),
            "Portfolio": portfolio_json,
            "Trade_Diagnostics": trade_json,
            "Params": params,
        }, f, indent=2)

    return {
        "Sharpe": float(portfolio.get("Sharpe", 0.0)),
        "Sortino": float(portfolio.get("Sortino", 0.0)),
        "Max_Drawdown_Pct": float(portfolio.get("Max_Drawdown_Pct", 0.0)),
        "Max_Drawdown_USD": float(portfolio.get("Max_Drawdown_USD", 0.0)),
        "PF": float(trade_stats.get("Profit_Factor", 0.0)),
        "Trades": n,
        "Threshold": used_thr,
        "Total_Profit_USD": total_profit,
        "Portfolio": portfolio,
        "Trade_Diagnostics": trade_stats,
        "Equity_Curve": equity_curve,
        "paths": {
            "trades": trades_csv,
            "trades_daily": simple_csv,
            "era": era_csv,
            "summary": summary_path,
        },
    }


def evaluate_holdout_performance(
    holdout_X,
    holdout_Xy,
    params,
    *,
    folds,
    k_test,
    embargo_days,
    prod_thr=None,
    thr_grid=None,
    min_train=200,
    min_test=50,
):
    """Run CPCV exclusively on the holdout segment using the tuned parameters."""

    if holdout_Xy is None or holdout_X is None:
        return {}, None
    if holdout_X.empty or holdout_Xy.empty:
        return {}, None

    fixed_thr = None if prod_thr is None else float(prod_thr)
    summary, details = _cpcv_evaluate(
        holdout_Xy,
        holdout_X,
        params,
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        min_train=min_train,
        min_test=min_test,
        thr_grid=thr_grid,
        fixed_thr=fixed_thr,
        collect_details=True,
    )

    return summary, details


# ========== Main pipeline ==========
"""Main orchestration for feature screening, random search, and BO."""

def main(
    inp,
    outdir,
    symbol,
    seed=42,
    rs_trials=25,
    bo_trials=65,
    folds=5,
    k_test=2,
    embargo_days=2,
    holdout_start=None,
    bo_batch=1,
    bo_acq="tpe",
    turbo=False,
    k_features=30,
    screen_method="importance",
    n_clusters=15,
    features_per_cluster=2,
    feature_selection_end="2020-12-31",
    score_metric="Sharpe",
): 
    os.makedirs(outdir, exist_ok=True)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = random.Random(seed)

    # load & features
    Xy = pd.read_csv(inp)
    for req in ["Date/Time", "y_return", "y_binary", "y_pnl_usd"]:
        if req not in Xy.columns:
            raise ValueError(f"Missing required column: {req}")
    Xy["Date"] = pd.to_datetime(Xy["Date/Time"])
    Xy["y_pnl_usd"] = pd.to_numeric(Xy["y_pnl_usd"], errors="coerce")

    X = add_engineered(build_core_features(Xy))
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    pct_like = [c for c in X.columns if c.endswith("_pct")]
    if pct_like:
        X[pct_like] = X[pct_like].fillna(0.50)
    rest = [c for c in X.columns if c not in pct_like]
    if rest:
        X[rest] = X[rest].fillna(0.0)

    # Parse feature selection end date (for time-based split to prevent data leakage)
    feat_sel_end_ts = None
    if feature_selection_end:
        try:
            feat_sel_end_ts = pd.to_datetime(feature_selection_end)
        except Exception as exc:
            raise ValueError(f"Invalid feature_selection_end '{feature_selection_end}': {exc}")
        if isinstance(feat_sel_end_ts, pd.DatetimeIndex):
            feat_sel_end_ts = feat_sel_end_ts[0]
        feat_sel_end_ts = pd.Timestamp(feat_sel_end_ts).tz_localize(None)

    # Parse holdout start date (for final holdout evaluation)
    holdout_ts = None
    if holdout_start:
        try:
            holdout_ts = pd.to_datetime(holdout_start)
        except Exception as exc:
            raise ValueError(f"Invalid holdout_start '{holdout_start}': {exc}")
        if isinstance(holdout_ts, pd.DatetimeIndex):
            holdout_ts = holdout_ts[0]
        holdout_ts = pd.Timestamp(holdout_ts).tz_localize(None)

    dates = Xy["Date"]

    # Three-way split: feature_selection_period | optimization_period | holdout_period
    # This prevents data leakage where test folds influence feature selection
    if feat_sel_end_ts is not None:
        feat_sel_mask = dates <= feat_sel_end_ts
        if holdout_ts is not None:
            optimization_mask = (dates > feat_sel_end_ts) & (dates < holdout_ts)
            holdout_mask = dates >= holdout_ts
        else:
            optimization_mask = dates > feat_sel_end_ts
            holdout_mask = pd.Series(False, index=Xy.index)

        feat_sel_days = int(pd.to_datetime(Xy.loc[feat_sel_mask, "Date"]).dt.normalize().nunique())
        opt_days = int(pd.to_datetime(Xy.loc[optimization_mask, "Date"]).dt.normalize().nunique())
        holdout_days = int(pd.to_datetime(Xy.loc[holdout_mask, "Date"]).dt.normalize().nunique())

        print("=" * 80)
        print("DATA LEAKAGE PREVENTION: Time-Based Split Active")
        print("=" * 80)
        print(f"Feature Selection Period: data <= {feat_sel_end_ts.date()}")
        print(f"  Rows: {int(feat_sel_mask.sum())} ({feat_sel_days} trading days)")
        print(f"Optimization Period: data > {feat_sel_end_ts.date()}" + (f" and < {holdout_ts.date()}" if holdout_ts else ""))
        print(f"  Rows: {int(optimization_mask.sum())} ({opt_days} trading days)")
        if holdout_ts:
            print(f"Holdout Period: data >= {holdout_ts.date()}")
            print(f"  Rows: {int(holdout_mask.sum())} ({holdout_days} trading days)")
        print()
        print("This ensures test data NEVER influences feature selection.")
        print("=" * 80 + "\n")

        train_mask = optimization_mask  # Optimization period is the "training" for hyperparameters
    else:
        # Fallback to old behavior (no feature selection split - NOT RECOMMENDED)
        print("⚠️  WARNING: No feature_selection_end specified - data leakage possible!")
        print("   Recommend using --feature_selection_end 2020-12-31\n")
        if holdout_ts is not None:
            train_mask = dates < holdout_ts
            holdout_mask = dates >= holdout_ts
            print(
                f"Using holdout_start={holdout_ts.date()} "
                f"({int(holdout_mask.sum())} rows reserved for holdout)"
            )
        else:
            train_mask = pd.Series(True, index=Xy.index)
            holdout_mask = pd.Series(False, index=Xy.index)

        feat_sel_mask = train_mask  # Use same data (old, leaky behavior)

    if not train_mask.any():
        raise ValueError("No training data available after applying date filters")
    if feat_sel_end_ts is not None and not feat_sel_mask.any():
        raise ValueError("No feature selection data available - check feature_selection_end date")

    # Prepare data splits
    X_feat_sel = X.loc[feat_sel_mask].copy()  # Feature selection period (early)
    Xy_feat_sel = Xy.loc[feat_sel_mask].copy()

    X_train_full = X.loc[train_mask].copy()  # Optimization period (late)
    Xy_train = Xy.loc[train_mask].copy()
    Xy_holdout = Xy.loc[holdout_mask].copy()

    # Feature selection on EARLY period ONLY (prevents data leakage)
    max_feat = X_feat_sel.shape[1]
    if k_features and k_features > 0:
        top_n = min(k_features, max_feat)
    else:
        top_n = max_feat

    print(f"Running feature selection on {len(X_feat_sel)} samples (feature selection period)")
    feats = screen_features(
        Xy_feat_sel,  # Use ONLY feature selection period data
        X_feat_sel,
        seed,
        method=screen_method,
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        top_n=top_n,
        n_clusters=n_clusters,
        features_per_cluster=features_per_cluster,
    )

    # Lock features and apply to ALL periods
    X = X[feats]
    X_train = X.loc[train_mask].copy()
    X_holdout = X.loc[holdout_mask].copy()
    print(f"✅ Selected {len(feats)} features (from early period): {', '.join(feats)}")
    print(f"   Applying these features to optimization period ({len(X_train)} samples)\n")

    # ---------- Random Search ----------
    # Calculate total trials for Deflated Sharpe correction
    n_trials_total = rs_trials + bo_trials

    rs_rows = []
    for t in range(1, rs_trials + 1):
        params = sample_rf_params(rng)
        res = evaluate_rf_cpcv(Xy_train, X_train, params, folds, k_test, embargo_days, n_trials_total=n_trials_total)
        rs_rows.append({**res, **params, "Trial": t, "Phase": "Random"})
        era_status = "Y" if res["Era_Positive"] else "N"
        print(
            f"[RS {t:03d}] SR={res['Sharpe']:.3f} PF={res['PF']:.3f} "
            f"Trades={res['Trades']} Thr={res['Thr']:.2f} "
            f"EraOK={era_status} ({res['Era_Positive_Count']}/{res['Era_Count']}) "
            f"params={params}"
        )

    rs_df = pd.DataFrame(rs_rows).sort_values("Sharpe", ascending=False)
    rs_csv = os.path.join(outdir, f"{symbol}_rf_cpcv_random.csv")
    rs_df.to_csv(rs_csv, index=False)

    if rs_df.empty:
        raise RuntimeError("Random search produced no results.")

    # FIXED: Use FULL hyperparameter space for Bayesian optimization
    # Previous implementation constrained to top Random Search results,
    # which could miss the global optimum if Random Search found a local maximum.
    # Now using the same full space as Random Search (from sample_rf_params).
    est_range = [300, 600, 900, 1200]
    depth_opts = [3, 5, 7, None]
    leaf_range = [50, 100, 200]
    mf_opts = ["sqrt", "log2", 0.3, 0.5]
    boot_opts = [True, False]
    cs_opts = [None, "balanced_subsample"]

    import optuna

    sampler = make_bo_sampler(optuna, seed, bo_acq, bo_batch, turbo)

    def objective(trial: optuna.Trial):
        n_estimators = trial.suggest_categorical("n_estimators", est_range or [600, 900, 1200])
        max_depth = trial.suggest_categorical("max_depth", depth_opts or [3, 5, 7, None])
        min_leaf = trial.suggest_categorical("min_samples_leaf", leaf_range or [50, 100, 200])
        max_features = trial.suggest_categorical("max_features", mf_opts or ["sqrt", "log2", 0.3, 0.5])
        bootstrap = trial.suggest_categorical("bootstrap", boot_opts or [True, False])
        class_weight = trial.suggest_categorical("class_weight", cs_opts or [None, "balanced_subsample"])
        max_samples = trial.suggest_float("max_samples", 0.6, 0.95) if bootstrap else None

        params = dict(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth in [None, "None", np.nan] else int(max_depth),
            min_samples_leaf=int(min_leaf),
            max_features=max_features,
            bootstrap=bool(bootstrap),
            class_weight=None if (class_weight in [None, "None", "none"]) else str(class_weight),
            max_samples=max_samples,
            n_jobs=-1,
            random_state=seed + trial.number,
        )

        res = evaluate_rf_cpcv(Xy_train, X_train, params, folds, k_test, embargo_days, n_trials_total=n_trials_total)
        trial.set_user_attr("PF", float(res["PF"]))
        trial.set_user_attr("Trades", int(res["Trades"]))
        trial.set_user_attr("Thr", float(res["Thr"]))
        trial.set_user_attr("Era_Positive", bool(res["Era_Positive"]))
        trial.set_user_attr("Era_Positive_Count", int(res["Era_Positive_Count"]))
        trial.set_user_attr("Era_Count", int(res["Era_Count"]))
        trial.set_user_attr("DSR", float(res.get("DSR", np.nan)))
        trial.set_user_attr("Total_PnL_USD", float(res.get("Total_PnL_USD", np.nan)))
        return float(res["Sharpe"])

    n_jobs = bo_batch if bo_batch and bo_batch > 1 else 1
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=bo_trials, show_progress_bar=False, n_jobs=n_jobs)

    bo_rows = []
    for tr in study.get_trials(deepcopy=False):
        p = tr.params.copy()
        p["max_depth"] = None if p.get("max_depth") in [None, "None", "nan"] else p.get("max_depth")
        bo_rows.append({
            "Trial": tr.number,
            "Phase": "BO",
            "Sharpe": tr.value,
            "PF": tr.user_attrs.get("PF", np.nan),
            "Trades": tr.user_attrs.get("Trades", np.nan),
            "Thr": tr.user_attrs.get("Thr", np.nan),
            "Era_Positive": tr.user_attrs.get("Era_Positive", False),
            "Era_Positive_Count": tr.user_attrs.get("Era_Positive_Count", np.nan),
            "Era_Count": tr.user_attrs.get("Era_Count", np.nan),
            "DSR": tr.user_attrs.get("DSR", np.nan),
            "Total_PnL_USD": tr.user_attrs.get("Total_PnL_USD", np.nan),
            **p,
        })
    bo_df = pd.DataFrame(bo_rows).sort_values("Sharpe", ascending=False)
    bo_csv = os.path.join(outdir, f"{symbol}_rf_cpcv_bo.csv")
    bo_df.to_csv(bo_csv, index=False)

    def with_dsr(df):
        if df.empty:
            return df
        out = df.copy()
        out.rename(
            columns={
                "Thr": "Prod_Threshold",
                "PF": "Profit_Factor",
                "DSR": "Deflated_Sharpe",
                "Total_PnL_USD": "Total_PnL_USD",
            },
            inplace=True,
        )
        if "Deflated_Sharpe" not in out.columns:
            out["Deflated_Sharpe"] = np.nan
        if "Total_PnL_USD" not in out.columns:
            out["Total_PnL_USD"] = np.nan
        return out

    MIN_TRADES = 500
    MIN_ERA_POSITIVE = 4
    REQUIRED_ERAS = 5

    cand = pd.concat(
        [
            with_dsr(
                rs_df[
                    [
                        "Sharpe",
                        "PF",
                        "Trades",
                        "Thr",
                        "DSR",
                        "Total_PnL_USD",
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                        "max_features",
                        "bootstrap",
                        "max_samples",
                        "class_weight",
                        "Era_Positive",
                        "Era_Positive_Count",
                        "Era_Count",
                    ]
                ]
            ),
            with_dsr(
                bo_df[
                    [
                        "Sharpe",
                        "PF",
                        "Trades",
                        "Thr",
                        "DSR",
                        "Total_PnL_USD",
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                        "max_features",
                        "bootstrap",
                        "max_samples",
                        "class_weight",
                        "Era_Positive",
                        "Era_Positive_Count",
                        "Era_Count",
                    ]
                ]
            ),
        ],
        ignore_index=True,
    )
    cand = cand[cand["Era_Positive"]]
    if cand.empty:
        print("No candidates met the era Sharpe guardrail (>=80% positive windows)")
        return

    metric_aliases = {
        "sharpe": "Sharpe",
        "deflated_sharpe": "Deflated_Sharpe",
        "dsr": "Deflated_Sharpe",
        "profit_factor": "Profit_Factor",
        "pf": "Profit_Factor",
        "trades": "Trades",
        "total_pnl": "Total_PnL_USD",
        "total_pnl_usd": "Total_PnL_USD",
        "total_profit": "Total_PnL_USD",
        "total_profit_usd": "Total_PnL_USD",
    }
    metric_col = metric_aliases.get(score_metric.lower(), score_metric)
    if metric_col not in cand.columns:
        raise ValueError(
            f"Score metric '{score_metric}' not available. Available columns: {', '.join(sorted(cand.columns))}"
        )
    cand["Score"] = cand[metric_col]
    print(f"Ranking candidates by {metric_col}")
    cand = cand.sort_values([metric_col, "Sharpe", "Profit_Factor"], ascending=[False, False, False])

    guardrail_msg = (
        f"Trades>={MIN_TRADES}, Era_Positive_Count>={MIN_ERA_POSITIVE}, Era_Count>={REQUIRED_ERAS}"
    )
    best_row = cand.iloc[0]
    guardrails_pass = (
        int(best_row.get("Trades", 0)) >= MIN_TRADES
        and int(best_row.get("Era_Positive_Count", 0)) >= MIN_ERA_POSITIVE
        and int(best_row.get("Era_Count", 0)) >= REQUIRED_ERAS
    )
    if not guardrails_pass:
        print("WARNING: Best candidate did not meet guardrails:", guardrail_msg)
    else:
        print("Guardrails satisfied:", guardrail_msg)

    prod_thr = float(best_row.get("Prod_Threshold", 0.60))
    print("\nBest candidate (pre-export):")
    print(best_row)
    print(f"Pre-holdout production threshold: {prod_thr:.2f}")

    def coerce_params(row):
        r = dict(row)
        p = {}
        p["n_estimators"] = int(r["n_estimators"])
        p["min_samples_leaf"] = int(r["min_samples_leaf"])
        p["max_depth"] = None if (pd.isna(r["max_depth"]) or str(r["max_depth"]) == "None") else int(r["max_depth"])
        try:
            p["max_features"] = float(r["max_features"])
        except Exception:
            p["max_features"] = str(r["max_features"])
        p["bootstrap"] = bool(r["bootstrap"])
        if p["bootstrap"]:
            try:
                p["max_samples"] = float(r["max_samples"])
            except Exception:
                p["max_samples"] = 0.8
        else:
            p["max_samples"] = None
        cw = r["class_weight"]
        p["class_weight"] = None if (str(cw) in ["None", "none", "nan"]) else str(cw)
        p["n_jobs"] = -1
        p["random_state"] = seed
        return p

    best_params = coerce_params(best_row)
    prod_thr = float(best_row.get("Prod_Threshold", 0.60))

    feat_list = list(map(str, X.columns))
    best_json_path = os.path.join(outdir, "best.json")

    holdout_summary, holdout_details = evaluate_holdout_performance(
        X_holdout,
        Xy_holdout,
        best_params,
        folds=folds,
        k_test=k_test,
        embargo_days=embargo_days,
        prod_thr=None,
    )

    with open(best_json_path, "w") as f:
        payload = {
            "Symbol": symbol,
            "Sharpe": float(best_row["Sharpe"]),
            "Deflated_Sharpe": None,
            "Profit_Factor": float(best_row["Profit_Factor"]),
            "Trades": int(best_row["Trades"]),
            "Prod_Threshold": float(prod_thr),
            "Guardrails": {
                "Trades": MIN_TRADES,
                "Era_Positive_Count": MIN_ERA_POSITIVE,
                "Era_Count": REQUIRED_ERAS,
            },
            "Params": best_params,
            "Features": feat_list,
            "Score_Metric": metric_col,
        }
        if holdout_summary:
            payload["Holdout"] = {
                k: (None if pd.isna(v) else float(v)) for k, v in holdout_summary.items()
            }
        if holdout_details and holdout_details.get("fold_metrics"):
            payload["Holdout_Folds"] = holdout_details["fold_metrics"]
        json.dump(payload, f, indent=2)
    print(f"\nWrote overall best config to: {best_json_path}")
    print(
        "Best Params:",
        best_params,
        "Prod_Threshold:",
        prod_thr,
        "Guardrails:",
        guardrail_msg,
    )

    final_params = best_params.copy()
    if not final_params.get("bootstrap", False):
        final_params["max_samples"] = None
    final_rf = RandomForestClassifier(**final_params)
    final_rf.fit(X_train, Xy_train["y_binary"])
    model_path = os.path.join(outdir, f"{symbol}_rf_model.pkl")
    joblib.dump({"model": final_rf, "features": feat_list}, model_path)
    print(f"Saved trained model to: {model_path}")

    if holdout_details and holdout_details.get("fold_metrics"):
        print("\nHoldout CPCV fold diagnostics:")
        for fm in holdout_details.get("fold_metrics", []):
            print(
                "  Fold {Fold:02d}: SR={Sharpe:.3f} Sortino={Sortino:.3f} "
                "MaxDD={Max_Drawdown_Pct:.2%} Trades={Trades} Thr={Threshold:.2f} Rows={Rows}".format(**fm)
            )

    if holdout_details:
        exp = export_selected_trades(
            Xy_holdout,
            X_holdout,
            best_params,
            None,
            folds,
            k_test,
            embargo_days,
            symbol,
            outdir,
            precomputed=(holdout_summary, holdout_details),
        )
    else:
        exp = export_selected_trades(
            Xy_train,
            X_train,
            best_params,
            prod_thr,
            folds,
            k_test,
            embargo_days,
            symbol,
            outdir,
        )
    port = exp.get("Portfolio", {})
    trades = exp.get("Trade_Diagnostics", {})
    sortino = port.get("Sortino", 0.0)
    cagr = port.get("CAGR", 0.0)
    total_ret_pct = port.get("Total_Return_Pct", 0.0)
    win_rate = trades.get("Win_Rate", 0.0)
    export_basis = "holdout" if holdout_details else "pre-holdout"
    print(
        f"\nExport complete ({export_basis} CPCV). "
        f"SR={exp['Sharpe']:.3f} "
        f"Sortino={sortino:.3f} "
        f"CAGR={cagr*100:.2f}% "
        f"TotalRet={total_ret_pct*100:.2f}% "
        f"PF={exp['PF']:.3f} "
        f"WinRate={win_rate*100:.1f}% "
        f"Trades={exp['Trades']}"
    )
    if holdout_summary:
        print(
            "Holdout (CPCV aggregation): "
            f"SR={holdout_summary.get('Sharpe', 0.0):.3f} "
            f"Sortino={holdout_summary.get('Sortino', 0.0):.3f} "
            f"DSR={holdout_summary.get('DSR', 0.0):.3f} "
            f"PF={holdout_summary.get('PF', 0.0):.3f} "
            f"Trades={int(holdout_summary.get('Trades', 0))} "
            f"Thr={holdout_summary.get('Thr', 0.0):.2f} "
            f"TotalPnL={holdout_summary.get('Total_PnL_USD', 0.0):.0f} "
            f"MaxDD%={holdout_summary.get('Max_Drawdown_Pct', 0.0)*100:.2f}%"
        )
    print(f"Files ({export_basis} diagnostics):")
    for k, v in exp["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to transformed_features.csv")
    ap.add_argument("--outdir", required=True, help="Directory for all tuner outputs")
    ap.add_argument("--symbol", default="ES", help="Instrument symbol label")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--holdout_start", help="YYYY-MM-DD to reserve data for holdout scoring")
    ap.add_argument("--rs_trials", type=int, default=25, help="Random search trial count (reduced from 120 for meta-labeling)")
    ap.add_argument("--bo_trials", type=int, default=65, help="Bayesian optimisation trial count (reduced from 300 for meta-labeling)")
    ap.add_argument("--bo_batch", type=int, default=1, help="Parallel Optuna jobs during BO")
    ap.add_argument(
        "--bo_acq",
        default="tpe",
        help="Acquisition/sampler for BO (e.g. tpe, random, cmaes, qei)",
    )
    ap.add_argument("--turbo", action="store_true", help="Enable TuRBO trust-region BO when available")
    ap.add_argument("--k_features", type=int, default=30, help="Top-k features to retain after screening")
    ap.add_argument(
        "--screen_method",
        default="importance",
        choices=["importance", "mdi", "permutation", "l1", "clustered", "none"],
        help="Feature screening strategy: importance/mdi (fast), permutation (robust), l1 (sparse), clustered (diverse, expert-recommended), none (all features)",
    )
    ap.add_argument(
        "--n_clusters",
        type=int,
        default=15,
        help="Number of clusters for clustered feature selection (default: 15)",
    )
    ap.add_argument(
        "--features_per_cluster",
        type=int,
        default=2,
        help="Features to select per cluster for clustered method (default: 2, giving ~30 total features)",
    )
    ap.add_argument(
        "--feature_selection_end",
        default="2020-12-31",
        help="YYYY-MM-DD split date: feature selection uses data BEFORE this date, optimization uses data AFTER (fixes data leakage)",
    )
    ap.add_argument("--embargo_days", type=int, default=2, help="Embargo window between train/test eras (reduced from 5 for meta-labeling)")
    ap.add_argument("--score_metric", default="Sharpe", help="Metric used to rank tuned candidates")
    ap.add_argument("--folds", type=int, default=5, help="Number of CPCV folds")
    ap.add_argument("--k_test", type=int, default=2, help="Test fold count per CPCV split")
    args = ap.parse_args()
    main(
        args.input,
        args.outdir,
        args.symbol,
        seed=args.seed,
        rs_trials=args.rs_trials,
        bo_trials=args.bo_trials,
        folds=args.folds,
        k_test=args.k_test,
        embargo_days=args.embargo_days,
        holdout_start=args.holdout_start,
        bo_batch=args.bo_batch,
        bo_acq=args.bo_acq,
        turbo=args.turbo,
        k_features=args.k_features,
        screen_method=args.screen_method,
        n_clusters=args.n_clusters,
        features_per_cluster=args.features_per_cluster,
        feature_selection_end=args.feature_selection_end,
        score_metric=args.score_metric,
    )
