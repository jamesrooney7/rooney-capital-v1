#!/usr/bin/env python3
"""
Parameter Stability Analysis for Walk-Forward Optimization Results.

This script analyzes how stable the optimal parameters are across
different walk-forward windows. Stable parameters indicate robust
strategy behavior that generalizes well.

Calculates:
- Coefficient of Variation (CV) for each parameter
- Robust parameter ranges (median ± 1 std)
- Stability ratings
- Parameter recommendations

Usage:
    python research/analyze_strategy_stability.py --symbol ES
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# STABILITY THRESHOLDS
# =============================================================================

STABILITY_THRESHOLDS = {
    'critical_params': {
        # Entry/exit thresholds - most important
        'max_cv': 0.20,
        'target_cv': 0.15,
    },
    'secondary_params': {
        # Stops/targets/holding - less critical
        'max_cv': 0.30,
        'target_cv': 0.25,
    },
}

CRITICAL_PARAMS = ['ibs_entry_high', 'ibs_exit_low']
SECONDARY_PARAMS = ['stop_atr_mult', 'target_atr_mult', 'max_holding_bars']


# =============================================================================
# STABILITY CALCULATIONS
# =============================================================================

def calculate_parameter_stability(param_values: List[float]) -> Dict:
    """
    Calculate coefficient of variation for a parameter.

    Args:
        param_values: List of parameter values across windows

    Returns:
        dict with 'cv', 'mean', 'std', 'stability_rating'
    """
    values = np.array(param_values)
    mean_val = np.mean(values)
    std_val = np.std(values)

    if mean_val == 0:
        cv = None
    else:
        cv = std_val / mean_val

    # Rate stability
    if cv is None:
        rating = 'invalid'
    elif cv < 0.10:
        rating = 'very_stable'
    elif cv < 0.20:
        rating = 'stable'
    elif cv < 0.30:
        rating = 'moderate'
    else:
        rating = 'unstable'

    return {
        'cv': float(cv) if cv is not None else None,
        'mean': float(mean_val),
        'std': float(std_val),
        'stability_rating': rating
    }


def calculate_robust_parameter_range(param_values: List[float]) -> Dict:
    """
    Calculate robust parameter range: median ± 1 std dev

    Returns:
        dict with 'median', 'std', 'lower_bound', 'upper_bound', 'recommended'
    """
    values = np.array(param_values)
    median = np.median(values)
    std = np.std(values)

    lower_bound = median - std
    upper_bound = median + std

    return {
        'median': float(median),
        'std': float(std),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'recommended': float(median),  # Use median for final parameters
    }


def evaluate_parameter_stability(param_name: str, cv: float) -> str:
    """
    Evaluate if parameter stability meets criteria.

    Returns: 'pass', 'warning', or 'fail'
    """
    if cv is None:
        return 'fail'

    if param_name in CRITICAL_PARAMS:
        if cv < STABILITY_THRESHOLDS['critical_params']['target_cv']:
            return 'pass'
        elif cv < STABILITY_THRESHOLDS['critical_params']['max_cv']:
            return 'warning'
        else:
            return 'fail'
    else:  # Secondary params
        if cv < STABILITY_THRESHOLDS['secondary_params']['target_cv']:
            return 'pass'
        elif cv < STABILITY_THRESHOLDS['secondary_params']['max_cv']:
            return 'warning'
        else:
            return 'fail'


def evaluate_stability_success(stability_results: Dict) -> Dict:
    """
    Evaluate if parameter stability meets success criteria.

    Returns: dict with 'approved' (bool), 'status', 'failures', 'warnings'
    """
    failures = []
    warnings = []

    for param_name, param_stability in stability_results.items():
        cv = param_stability['cv']
        if cv is None:
            continue

        evaluation = evaluate_parameter_stability(param_name, cv)

        if evaluation == 'fail':
            failures.append({
                'param': param_name,
                'cv': cv,
                'type': 'critical' if param_name in CRITICAL_PARAMS else 'secondary'
            })
        elif evaluation == 'warning':
            warnings.append({
                'param': param_name,
                'cv': cv,
            })

    # Determine approval
    critical_failures = sum(1 for f in failures if f['type'] == 'critical')

    if critical_failures == 0:
        if len(failures) == 0:
            status = 'APPROVED'
            approved = True
        else:
            status = 'APPROVED_WITH_WARNINGS'
            approved = True
    else:
        status = 'REJECTED'
        approved = False

    return {
        'approved': approved,
        'status': status,
        'failures': failures,
        'warnings': warnings,
        'critical_failures': critical_failures,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze parameter stability from walk-forward optimization'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Symbol to analyze (e.g., ES)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory (default: optimization_results/{SYMBOL})'
    )

    args = parser.parse_args()

    # Setup paths
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = Path('optimization_results') / args.symbol

    windows_dir = results_root / 'windows'
    analysis_dir = results_root / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"PARAMETER STABILITY ANALYSIS")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Results: {results_root}")
    logger.info(f"{'='*80}\n")

    # Load optimal parameters from all windows
    all_params = []
    window_ids = []

    for window_id in range(1, 6):
        params_file = windows_dir / f"window_{window_id}_optimal_params.json"

        if not params_file.exists():
            logger.warning(f"Missing window {window_id} parameters: {params_file}")
            continue

        with open(params_file, 'r') as f:
            params = json.load(f)
            all_params.append(params)
            window_ids.append(window_id)

    if len(all_params) < 3:
        logger.error(f"Insufficient windows for stability analysis (found {len(all_params)}, need 3+)")
        return

    logger.info(f"Loaded parameters from {len(all_params)} windows: {window_ids}")

    # Parameters to analyze (only the optimized ones)
    param_names = [
        'ibs_entry_high',
        'ibs_exit_low',
        'stop_atr_mult',
        'target_atr_mult',
        'max_holding_bars',
    ]

    # Calculate stability for each parameter
    logger.info(f"\nParameter Stability Analysis:")
    logger.info(f"{'='*80}")

    stability_results = {}

    for param_name in param_names:
        param_values = [p[param_name] for p in all_params]

        stability = calculate_parameter_stability(param_values)
        stability_results[param_name] = stability

        param_type = "CRITICAL" if param_name in CRITICAL_PARAMS else "Secondary"
        evaluation = evaluate_parameter_stability(param_name, stability['cv'])

        logger.info(f"\n{param_name} ({param_type}):")
        logger.info(f"  Values across windows: {param_values}")
        logger.info(f"  Mean: {stability['mean']:.3f}")
        logger.info(f"  Std Dev: {stability['std']:.3f}")
        logger.info(f"  CV: {stability['cv']:.3f}" if stability['cv'] is not None else "  CV: N/A")
        logger.info(f"  Rating: {stability['stability_rating'].upper()}")
        logger.info(f"  Evaluation: {evaluation.upper()}")

    # Calculate robust parameter ranges
    logger.info(f"\n{'='*80}")
    logger.info(f"Robust Parameter Ranges (median ± 1 std):")
    logger.info(f"{'='*80}")

    robust_ranges = {}

    for param_name in param_names:
        param_values = [p[param_name] for p in all_params]
        robust_range = calculate_robust_parameter_range(param_values)
        robust_ranges[param_name] = robust_range

        logger.info(f"\n{param_name}:")
        logger.info(f"  Median: {robust_range['median']:.3f}")
        logger.info(f"  Range: [{robust_range['lower_bound']:.3f}, {robust_range['upper_bound']:.3f}]")
        logger.info(f"  RECOMMENDED: {robust_range['recommended']:.3f}")

    # Evaluate overall stability success
    logger.info(f"\n{'='*80}")
    logger.info(f"STABILITY EVALUATION")
    logger.info(f"{'='*80}")

    stability_eval = evaluate_stability_success(stability_results)

    logger.info(f"\nOverall Status: {stability_eval['status']}")
    logger.info(f"Approved: {stability_eval['approved']}")

    if stability_eval['failures']:
        logger.info(f"\nFailed Parameters ({len(stability_eval['failures'])}):")
        for f in stability_eval['failures']:
            logger.info(f"  - {f['param']} (CV={f['cv']:.3f}, Type={f['type']})")

    if stability_eval['warnings']:
        logger.info(f"\nWarning Parameters ({len(stability_eval['warnings'])}):")
        for w in stability_eval['warnings']:
            logger.info(f"  - {w['param']} (CV={w['cv']:.3f})")

    # Save results
    stability_file = analysis_dir / 'parameter_stability_analysis.json'
    with open(stability_file, 'w') as f:
        json.dump({
            'symbol': args.symbol,
            'windows_analyzed': window_ids,
            'stability_results': stability_results,
            'stability_evaluation': stability_eval,
        }, f, indent=2)

    logger.info(f"\nSaved stability analysis to: {stability_file}")

    ranges_file = analysis_dir / 'robust_parameter_ranges.json'
    with open(ranges_file, 'w') as f:
        json.dump(robust_ranges, f, indent=2)

    logger.info(f"Saved robust ranges to: {ranges_file}")

    logger.info(f"\n{'='*80}")
    logger.info(f"STABILITY ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Next step:")
    logger.info(f"  python research/validate_strategy_heldout.py --symbol {args.symbol}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
