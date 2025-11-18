#!/usr/bin/env python3
"""
Feature Stability Analysis for ML Trading Models.

This script analyzes feature stability across different time periods within the
training data to identify regime-dependent or unreliable features BEFORE running
full hyperparameter optimization.

Approach:
1. Split training period (2010-2018) into rolling windows
2. Train simple RF models on each window
3. Track feature importance and selection consistency across windows
4. Flag features with high variance or low consistency
5. Generate report for manual review

This helps identify features that don't generalize across different market regimes
without touching threshold or test data (no lookahead bias).

Usage:
    python research/analyze_feature_stability.py --symbol ES --k-features 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.rf_optimization_utils import (
    screen_features,
    build_core_features,
    evaluate_rf_cpcv,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_problematic_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter out known problematic features (duplicates, VIX, etc.).

    Returns:
        Tuple of (filtered DataFrame, dict of filter counts)
    """
    valid_cols = []
    counts = {
        'title_case': 0,
        'enable_params': 0,
        'vix': 0,
        'redundant': 0,
    }

    cross_symbols = ["ES", "NQ", "RTY", "YM", "GC", "SI", "HG", "CL", "NG", "PL",
                     "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S", "TLT", "VIX"]

    for col in X.columns:
        # Skip Title Case cross-instrument features
        if " " in col and any(col.startswith(sym + " ") for sym in cross_symbols):
            counts['title_case'] += 1
            continue

        # Skip enable parameter columns
        if col.startswith("enable") and any(sym in col for sym in cross_symbols):
            if ("Return" in col or "ZScore" in col) and ("Hour" in col or "Day" in col):
                counts['enable_params'] += 1
                continue

        # Skip VIX features
        if "vix" in col.lower():
            counts['vix'] += 1
            continue

        # Skip redundant features
        if col == "enableIBSExit":
            counts['redundant'] += 1
            continue

        valid_cols.append(col)

    return X[valid_cols].copy(), counts


def create_rolling_windows(
    df: pd.DataFrame,
    window_years: int = 4,
    step_years: int = 1,
) -> List[Tuple[str, pd.DataFrame]]:
    """Split dataframe into rolling time windows.

    Args:
        df: DataFrame with 'date' column
        window_years: Size of each window in years
        step_years: Step size between windows in years

    Returns:
        List of (window_name, window_df) tuples
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    min_year = df['date'].dt.year.min()
    max_year = df['date'].dt.year.max()

    windows = []
    start_year = min_year

    while start_year + window_years <= max_year + 1:
        end_year = start_year + window_years

        window_df = df[
            (df['date'].dt.year >= start_year) &
            (df['date'].dt.year < end_year)
        ].copy()

        if len(window_df) > 100:  # Minimum data requirement
            window_name = f"{start_year}-{end_year-1}"
            windows.append((window_name, window_df))

        start_year += step_years

    return windows


def analyze_window(
    window_name: str,
    Xy: pd.DataFrame,
    X: pd.DataFrame,
    k_features: int,
    screen_method: str,
    seed: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Analyze feature importance for a single time window.

    Args:
        window_name: Name of the time window
        Xy: Full feature matrix with target
        X: Feature-only matrix
        k_features: Number of features to select
        screen_method: Feature screening method
        seed: Random seed

    Returns:
        Tuple of (selected_features, importance_dict)
    """
    logger.info(f"\nAnalyzing window: {window_name} ({len(Xy)} samples)")

    # Filter problematic features
    X_filtered, filter_counts = filter_problematic_features(X)
    logger.info(
        f"  Filtered: {filter_counts['title_case']} title_case, "
        f"{filter_counts['enable_params']} enable_params, "
        f"{filter_counts['vix']} vix, {filter_counts['redundant']} redundant"
    )

    # Screen features
    selected_features = screen_features(
        Xy,
        X_filtered,
        seed=seed,
        method=screen_method,
        folds=5,
        k_test=1,
        embargo_days=5,
        top_n=k_features,
    )
    logger.info(f"  Selected {len(selected_features)} features")

    # Train simple RF to get importance
    X_selected = X[selected_features].copy()
    y = Xy['target'].values

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=100,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_selected.fillna(0), y)

    # Get feature importances
    importance_dict = dict(zip(selected_features, rf.feature_importances_))

    logger.info(f"  Top 5 features: {list(importance_dict.keys())[:5]}")

    return selected_features, importance_dict


def calculate_stability_metrics(
    window_results: List[Tuple[str, List[str], Dict[str, float]]],
    k_features: int,
) -> pd.DataFrame:
    """Calculate stability metrics for all features across windows.

    Args:
        window_results: List of (window_name, selected_features, importance_dict)
        k_features: Number of features that were selected per window

    Returns:
        DataFrame with stability metrics per feature
    """
    # Collect all features that appeared in at least one window
    all_features = set()
    for _, selected, importance in window_results:
        all_features.update(selected)

    # Build importance matrix: features x windows
    importance_matrix = []
    selection_matrix = []

    for feature in sorted(all_features):
        importances = []
        selections = []

        for window_name, selected, importance_dict in window_results:
            # Importance (0 if not selected)
            imp = importance_dict.get(feature, 0.0)
            importances.append(imp)

            # Binary selection indicator
            selections.append(1 if feature in selected else 0)

        importance_matrix.append(importances)
        selection_matrix.append(selections)

    # Convert to arrays
    importance_matrix = np.array(importance_matrix)
    selection_matrix = np.array(selection_matrix)

    # Calculate metrics for each feature
    metrics = []

    for i, feature in enumerate(sorted(all_features)):
        importances = importance_matrix[i]
        selections = selection_matrix[i]

        # Only calculate stats when feature was actually selected
        selected_importances = importances[importances > 0]

        if len(selected_importances) > 0:
            mean_importance = selected_importances.mean()
            std_importance = selected_importances.std()
            cv_importance = std_importance / mean_importance if mean_importance > 0 else 0
        else:
            mean_importance = 0
            std_importance = 0
            cv_importance = 0

        # Selection consistency: how often was it selected?
        selection_rate = selections.mean()

        # Rank stability: how much does rank vary?
        ranks = []
        for window_name, selected, importance_dict in window_results:
            if feature in importance_dict:
                # Get rank within this window (higher importance = lower rank number)
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                rank = [f for f, _ in sorted_features].index(feature) + 1
                ranks.append(rank)

        rank_std = np.std(ranks) if len(ranks) > 1 else 0

        metrics.append({
            'feature': feature,
            'selection_rate': selection_rate,
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'cv_importance': cv_importance,
            'rank_std': rank_std,
            'n_windows_selected': int(selections.sum()),
            'n_windows_total': len(window_results),
        })

    df = pd.DataFrame(metrics)
    df = df.sort_values('cv_importance', ascending=False)

    return df


def categorize_features(stability_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize features by stability and provide recommendations.

    Args:
        stability_df: DataFrame with stability metrics

    Returns:
        Dictionary with categorized feature lists
    """
    categories = {
        'high_variance': [],
        'inconsistent': [],
        'stable': [],
        'regime_dependent_currencies': [],
        'regime_dependent_metals': [],
    }

    for _, row in stability_df.iterrows():
        feature = row['feature']
        cv = row['cv_importance']
        selection_rate = row['selection_rate']

        # High variance in importance when selected
        if cv > 0.5 and selection_rate > 0.3:
            categories['high_variance'].append(feature)

        # Inconsistently selected (sometimes in top K, sometimes not)
        if 0.2 < selection_rate < 0.8:
            categories['inconsistent'].append(feature)

        # Stable (consistently selected with low variance)
        if selection_rate >= 0.8 and cv < 0.3:
            categories['stable'].append(feature)

        # Currency features (known to be regime-dependent)
        if any(curr in feature for curr in ['6a', '6b', '6c', '6e', '6j', '6m', '6n', '6s']):
            categories['regime_dependent_currencies'].append(feature)

        # Metals/commodities (can be regime-dependent)
        if any(metal in feature for metal in ['gc_', 'si_', 'hg_', 'cl_', 'ng_', 'pl_']):
            categories['regime_dependent_metals'].append(feature)

    return categories


def generate_report(
    stability_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    window_results: List[Tuple[str, List[str], Dict[str, float]]],
    output_path: Path,
):
    """Generate human-readable stability analysis report.

    Args:
        stability_df: DataFrame with stability metrics
        categories: Categorized feature lists
        window_results: Raw window analysis results
        output_path: Path to save report
    """
    report = []

    report.append("=" * 80)
    report.append("FEATURE STABILITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total features analyzed: {len(stability_df)}")
    report.append(f"Time windows: {len(window_results)}")
    report.append("")

    for window_name, selected, _ in window_results:
        report.append(f"  {window_name}: {len(selected)} features selected")
    report.append("")

    # High variance features (REVIEW THESE)
    report.append("HIGH VARIANCE FEATURES (REVIEW RECOMMENDED)")
    report.append("-" * 80)
    report.append("These features show high coefficient of variation in importance")
    report.append("across different time periods, suggesting regime-dependence.")
    report.append("")

    if categories['high_variance']:
        high_var_df = stability_df[stability_df['feature'].isin(categories['high_variance'])]
        report.append(f"Found {len(high_var_df)} high variance features:")
        report.append("")
        for _, row in high_var_df.iterrows():
            report.append(
                f"  {row['feature']:40s} | CV={row['cv_importance']:.2f} | "
                f"Selection={row['selection_rate']:.1%} | "
                f"MeanImp={row['mean_importance']:.4f}"
            )
    else:
        report.append("  No high variance features found.")
    report.append("")

    # Inconsistent features
    report.append("INCONSISTENTLY SELECTED FEATURES")
    report.append("-" * 80)
    report.append("These features are sometimes selected, sometimes not,")
    report.append("suggesting they may not be reliably predictive.")
    report.append("")

    if categories['inconsistent']:
        inconsistent_df = stability_df[stability_df['feature'].isin(categories['inconsistent'])]
        report.append(f"Found {len(inconsistent_df)} inconsistent features:")
        report.append("")
        for _, row in inconsistent_df.head(20).iterrows():
            report.append(
                f"  {row['feature']:40s} | Selected in {row['n_windows_selected']}/{row['n_windows_total']} windows | "
                f"CV={row['cv_importance']:.2f}"
            )
        if len(inconsistent_df) > 20:
            report.append(f"  ... and {len(inconsistent_df) - 20} more")
    else:
        report.append("  No inconsistent features found.")
    report.append("")

    # Stable features (GOOD)
    report.append("STABLE FEATURES (HIGH CONFIDENCE)")
    report.append("-" * 80)
    report.append("These features are consistently selected with low variance,")
    report.append("suggesting robust predictive power across regimes.")
    report.append("")

    if categories['stable']:
        stable_df = stability_df[stability_df['feature'].isin(categories['stable'])]
        report.append(f"Found {len(stable_df)} stable features:")
        report.append("")
        for _, row in stable_df.iterrows():
            report.append(
                f"  {row['feature']:40s} | Selection={row['selection_rate']:.1%} | "
                f"CV={row['cv_importance']:.2f} | "
                f"MeanImp={row['mean_importance']:.4f}"
            )
    else:
        report.append("  No highly stable features found.")
    report.append("")

    # Currency features (domain knowledge)
    report.append("CURRENCY FEATURES (POTENTIALLY REGIME-DEPENDENT)")
    report.append("-" * 80)
    report.append("Currencies are known to change behavior across carry trade,")
    report.append("crisis, and QE regimes. Consider removing if unstable.")
    report.append("")

    if categories['regime_dependent_currencies']:
        currency_df = stability_df[stability_df['feature'].isin(categories['regime_dependent_currencies'])]
        report.append(f"Found {len(currency_df)} currency features:")
        report.append("")
        for _, row in currency_df.iterrows():
            report.append(
                f"  {row['feature']:40s} | Selection={row['selection_rate']:.1%} | "
                f"CV={row['cv_importance']:.2f}"
            )
    else:
        report.append("  No currency features found.")
    report.append("")

    # Metal/commodity features
    report.append("METAL/COMMODITY FEATURES")
    report.append("-" * 80)
    report.append("Metals and commodities can be regime-dependent (e.g., inflation cycles).")
    report.append("")

    if categories['regime_dependent_metals']:
        metal_df = stability_df[stability_df['feature'].isin(categories['regime_dependent_metals'])]
        report.append(f"Found {len(metal_df)} metal/commodity features:")
        report.append("")
        for _, row in metal_df.iterrows():
            report.append(
                f"  {row['feature']:40s} | Selection={row['selection_rate']:.1%} | "
                f"CV={row['cv_importance']:.2f}"
            )
    else:
        report.append("  No metal/commodity features found.")
    report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("")
    report.append("1. REMOVE: Features with CV > 0.7 and selection rate < 0.5")
    report.append("   → These are highly unstable and unreliable")
    report.append("")

    remove_candidates = stability_df[
        (stability_df['cv_importance'] > 0.7) &
        (stability_df['selection_rate'] < 0.5)
    ]
    if len(remove_candidates) > 0:
        report.append("   Suggested removals:")
        for _, row in remove_candidates.iterrows():
            report.append(f"     - {row['feature']}")
    else:
        report.append("   No clear removal candidates.")
    report.append("")

    report.append("2. REVIEW: Currency features with selection rate < 0.6")
    report.append("   → Known regime-dependence, consider removing")
    report.append("")

    currency_review = stability_df[
        stability_df['feature'].isin(categories['regime_dependent_currencies']) &
        (stability_df['selection_rate'] < 0.6)
    ]
    if len(currency_review) > 0:
        report.append("   Currency features to review:")
        for _, row in currency_review.iterrows():
            report.append(f"     - {row['feature']} (selected {row['selection_rate']:.1%})")
    else:
        report.append("   No currency features need review.")
    report.append("")

    report.append("3. KEEP: Stable features with selection rate >= 0.8 and CV < 0.3")
    report.append("   → These are robust across regimes")
    report.append("")

    # Full detailed table
    report.append("")
    report.append("=" * 80)
    report.append("FULL FEATURE STABILITY TABLE")
    report.append("=" * 80)
    report.append("")
    report.append(
        f"{'Feature':40s} | {'SelRate':>8s} | {'CV':>6s} | {'MeanImp':>8s} | "
        f"{'StdImp':>8s} | {'RankStd':>8s} | {'N/Total':>8s}"
    )
    report.append("-" * 120)

    for _, row in stability_df.iterrows():
        report.append(
            f"{row['feature']:40s} | "
            f"{row['selection_rate']:8.1%} | "
            f"{row['cv_importance']:6.2f} | "
            f"{row['mean_importance']:8.4f} | "
            f"{row['std_importance']:8.4f} | "
            f"{row['rank_std']:8.2f} | "
            f"{row['n_windows_selected']:3d}/{row['n_windows_total']:3d}"
        )

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"\nReport saved to: {output_path}")

    # Also print key sections to console
    print("\n" + "\n".join(report[:50]))  # Print first 50 lines
    print("\n... (see full report in output file) ...\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature stability across time periods")
    parser.add_argument("--symbol", type=str, default="ES", help="Trading symbol")
    parser.add_argument("--data-dir", type=str, default="research/training_data", help="Training data directory")
    parser.add_argument("--k-features", type=int, default=30, help="Number of features to select per window")
    parser.add_argument("--screen-method", type=str, default="importance",
                       choices=["importance", "permutation", "l1"],
                       help="Feature screening method")
    parser.add_argument("--window-years", type=int, default=4, help="Window size in years")
    parser.add_argument("--step-years", type=int, default=1, help="Step size between windows in years")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_stability",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    csv_path = Path(args.data_dir) / f"{args.symbol}_transformed_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}\n"
            f"Run: python research/extract_training_data.py --symbol {args.symbol}"
        )

    logger.info(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} trades with {len(df.columns)} columns")

    # Ensure date column
    if "Date/Time" in df.columns:
        df['Date'] = pd.to_datetime(df['Date/Time'])
    elif "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No date column found")

    # Filter to training period only (2010-2018)
    df = df[df['Date'] < pd.Timestamp('2019-01-01')].copy()
    logger.info(f"Using training period: {len(df)} samples (2010-2018)")

    # Build core features (normalize)
    logger.info("Building core features...")
    Xy_train, fitted_scaler = build_core_features(df, scaler=None, fit_scaler=True)

    # Separate features from target
    X_train = Xy_train.drop(columns=['target', 'trade_return'])
    logger.info(f"Feature matrix: {X_train.shape}")

    # Create rolling windows
    logger.info(f"\nCreating rolling windows ({args.window_years} year windows, {args.step_years} year steps)...")
    windows = create_rolling_windows(Xy_train, args.window_years, args.step_years)
    logger.info(f"Created {len(windows)} windows")

    # Analyze each window
    window_results = []

    for window_name, window_df in windows:
        # Separate features from target for this window
        X_window = window_df.drop(columns=['target', 'trade_return', 'date'])

        selected_features, importance_dict = analyze_window(
            window_name,
            window_df,
            X_window,
            args.k_features,
            args.screen_method,
            args.seed,
        )

        window_results.append((window_name, selected_features, importance_dict))

    # Calculate stability metrics
    logger.info("\nCalculating stability metrics...")
    stability_df = calculate_stability_metrics(window_results, args.k_features)

    # Categorize features
    logger.info("Categorizing features...")
    categories = categorize_features(stability_df)

    # Save detailed results
    stability_csv = output_dir / f"{args.symbol}_feature_stability.csv"
    stability_df.to_csv(stability_csv, index=False)
    logger.info(f"Stability metrics saved to: {stability_csv}")

    # Save window results
    window_json = output_dir / f"{args.symbol}_window_results.json"
    window_data = {
        window_name: {
            'selected_features': selected,
            'importance': {k: float(v) for k, v in importance.items()},
        }
        for window_name, selected, importance in window_results
    }
    with open(window_json, 'w') as f:
        json.dump(window_data, f, indent=2)
    logger.info(f"Window results saved to: {window_json}")

    # Generate human-readable report
    report_path = output_dir / f"{args.symbol}_stability_report.txt"
    generate_report(stability_df, categories, window_results, report_path)

    logger.info("\n" + "="*80)
    logger.info("FEATURE STABILITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Stability metrics: {stability_csv}")
    logger.info(f"  - Window details: {window_json}")
    logger.info(f"  - Human report: {report_path}")
    logger.info("\nReview the report before running model comparison!")


if __name__ == "__main__":
    main()
