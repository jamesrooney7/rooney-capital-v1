#!/usr/bin/env python3
"""
Held-Out Validation for Optimized Strategy Parameters.

This script validates the optimized parameters on truly unseen data (2021-2024)
that was never used during optimization. This is the final test to ensure
the strategy generalizes to future periods.

Calculates:
- Validation Efficiency (VE = Heldout Sharpe / WF Mean Sharpe)
- Metric consistency (win rate, drawdown)
- Trade volume sufficiency

Usage:
    python research/validate_strategy_heldout.py --symbol ES
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research.utils.vectorized_backtest import load_data, run_backtest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELD-OUT PERIOD
# =============================================================================

HELDOUT_PERIOD = {
    'start': '2021-01-01',
    'end': '2024-12-31',
}

HELDOUT_THRESHOLDS = {
    'reject': 1000,              # Below this: Can't validate
    'marginal': 2000,            # Below this: High uncertainty
    'target': 2800,              # Expected baseline (4/14 × 10k)
    'good': 3500,                # Good validation volume
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def evaluate_heldout_volume(num_trades: int) -> str:
    """
    Evaluate held-out trade volume.

    Returns: 'reject', 'marginal', 'ok', or 'good'
    """
    if num_trades < HELDOUT_THRESHOLDS['reject']:
        return 'reject'
    elif num_trades < HELDOUT_THRESHOLDS['marginal']:
        return 'marginal'
    elif num_trades < HELDOUT_THRESHOLDS['target']:
        return 'ok'
    else:
        return 'good'


def calculate_validation_efficiency(
    heldout_sharpe: float,
    walkforward_mean_sharpe: float
) -> float:
    """
    Calculate Validation Efficiency.

    VE = Heldout Sharpe / Walk-Forward Mean Sharpe

    Similar to WFE but for held-out validation.
    """
    if walkforward_mean_sharpe <= 0:
        return None

    ve = heldout_sharpe / walkforward_mean_sharpe
    return ve


def evaluate_heldout_success(
    heldout_results: Dict,
    walkforward_mean_metrics: Dict
) -> Dict:
    """
    Evaluate if held-out validation meets success criteria.

    Returns: dict with 'approved' (bool), 'status', 'criteria', etc.
    """
    # Calculate validation efficiency
    ve = calculate_validation_efficiency(
        heldout_results['sharpe_ratio'],
        walkforward_mean_metrics['sharpe_ratio']
    )

    # Calculate win rate divergence
    win_rate_diff = abs(
        heldout_results['win_rate'] - walkforward_mean_metrics['win_rate']
    )

    # Criteria checks
    criteria = {
        'validation_efficiency': {
            'value': ve,
            'threshold': 0.5,
            'pass': ve >= 0.5 if ve is not None else False,
            'weight': 'critical',
        },
        'num_trades': {
            'value': heldout_results['num_trades'],
            'threshold': 1500,
            'pass': heldout_results['num_trades'] >= 1500,
            'weight': 'critical',
        },
        'win_rate_consistency': {
            'value': win_rate_diff,
            'threshold': 0.05,  # Within 5 percentage points
            'pass': win_rate_diff <= 0.05,
            'weight': 'important',
        },
        'max_drawdown': {
            'value': heldout_results['max_drawdown_pct'],
            'threshold': 0.25,  # Less than 25%
            'pass': heldout_results['max_drawdown_pct'] < 0.25,
            'weight': 'important',
        },
    }

    # Count critical failures
    critical_failures = sum(
        1 for c in criteria.values()
        if c['weight'] == 'critical' and not c['pass']
    )

    # Determine approval
    if critical_failures == 0:
        if all(c['pass'] for c in criteria.values()):
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
        'validation_efficiency': ve,
        'criteria': criteria,
        'critical_failures': critical_failures,
    }


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate optimized parameters on held-out data'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Symbol to validate (e.g., ES)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/resampled',
        help='Data directory (default: data/resampled)'
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

    analysis_dir = results_root / 'analysis'
    heldout_dir = results_root / 'heldout'
    heldout_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"HELD-OUT VALIDATION")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {HELDOUT_PERIOD['start']} to {HELDOUT_PERIOD['end']}")
    logger.info(f"{'='*80}\n")

    # Load robust parameters (median from walk-forward)
    robust_ranges_file = analysis_dir / 'robust_parameter_ranges.json'
    if not robust_ranges_file.exists():
        logger.error(f"Robust parameter ranges not found: {robust_ranges_file}")
        logger.error(f"Run analyze_strategy_stability.py first!")
        return

    with open(robust_ranges_file, 'r') as f:
        robust_ranges = json.load(f)

    # Extract recommended parameters
    final_params = {k: v['recommended'] for k, v in robust_ranges.items()}

    # Add fixed parameters
    final_params['ibs_entry_low'] = 0.0
    final_params['ibs_exit_high'] = 1.0
    final_params['atr_period'] = 14
    final_params['auto_close_hour'] = 15

    logger.info(f"Final Parameters (median from walk-forward):")
    for k, v in final_params.items():
        logger.info(f"  {k}: {v}")

    # Load held-out data
    logger.info(f"\nLoading held-out data...")
    heldout_data = load_data(
        args.symbol,
        data_dir=args.data_dir,
        start_date=HELDOUT_PERIOD['start'],
        end_date=HELDOUT_PERIOD['end']
    )

    # Run backtest on held-out period
    logger.info(f"\nRunning backtest on held-out data...")
    heldout_metrics = run_backtest(heldout_data, final_params, symbol=args.symbol)

    logger.info(f"\nHeld-Out Results:")
    logger.info(f"{'='*80}")
    logger.info(f"Trades: {heldout_metrics['num_trades']:,}")
    logger.info(f"Win Rate: {heldout_metrics['win_rate']:.2%}")
    logger.info(f"Sharpe: {heldout_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Avg P&L: {heldout_metrics['avg_pnl_per_trade']:.2f}")
    logger.info(f"Profit Factor: {heldout_metrics['profit_factor']:.2f}" if heldout_metrics['profit_factor'] else "Profit Factor: N/A")
    logger.info(f"Max Drawdown: {heldout_metrics['max_drawdown_pct']:.2%}")
    logger.info(f"Avg Duration: {heldout_metrics['avg_duration_bars']:.1f} hours")

    # Check held-out volume
    volume_rating = evaluate_heldout_volume(heldout_metrics['num_trades'])
    logger.info(f"\nVolume Rating: {volume_rating.upper()}")

    # Load walk-forward aggregate metrics
    aggregate_file = analysis_dir / 'aggregate_walkforward_metrics.json'
    if not aggregate_file.exists():
        # Try parent directory
        aggregate_file = results_root / 'analysis' / 'aggregate_walkforward_metrics.json'

    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)

    # Calculate mean walk-forward metrics
    windows = aggregate_data['windows']

    wf_mean_metrics = {
        'num_trades': np.mean([w['test_metrics']['num_trades'] for w in windows]),
        'win_rate': np.mean([w['test_metrics']['win_rate'] for w in windows]),
        'sharpe_ratio': np.mean([w['test_metrics']['sharpe_ratio'] for w in windows]),
        'avg_pnl_per_trade': np.mean([w['test_metrics']['avg_pnl_per_trade'] for w in windows]),
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"Comparison to Walk-Forward Mean:")
    logger.info(f"{'='*80}")
    logger.info(f"Metric              WF Mean     Held-Out    Delta")
    logger.info(f"{'─'*80}")
    logger.info(f"Trades              {wf_mean_metrics['num_trades']:>7.0f}     {heldout_metrics['num_trades']:>8,}    {heldout_metrics['num_trades'] - wf_mean_metrics['num_trades']:>+8.0f}")
    logger.info(f"Win Rate            {wf_mean_metrics['win_rate']:>7.2%}     {heldout_metrics['win_rate']:>8.2%}    {heldout_metrics['win_rate'] - wf_mean_metrics['win_rate']:>+8.2%}")
    logger.info(f"Sharpe              {wf_mean_metrics['sharpe_ratio']:>7.3f}     {heldout_metrics['sharpe_ratio']:>8.3f}    {heldout_metrics['sharpe_ratio'] - wf_mean_metrics['sharpe_ratio']:>+8.3f}")
    logger.info(f"Avg P&L             {wf_mean_metrics['avg_pnl_per_trade']:>7.2f}     {heldout_metrics['avg_pnl_per_trade']:>8.2f}    {heldout_metrics['avg_pnl_per_trade'] - wf_mean_metrics['avg_pnl_per_trade']:>+8.2f}")

    # Evaluate held-out success
    heldout_eval = evaluate_heldout_success(heldout_metrics, wf_mean_metrics)

    logger.info(f"\n{'='*80}")
    logger.info(f"HELD-OUT EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Status: {heldout_eval['status']}")
    logger.info(f"Approved: {heldout_eval['approved']}")
    if heldout_eval['validation_efficiency'] is not None:
        logger.info(f"Validation Efficiency: {heldout_eval['validation_efficiency']:.3f}")

    logger.info(f"\nCriteria:")
    for criterion_name, criterion in heldout_eval['criteria'].items():
        status_icon = "✓" if criterion['pass'] else "✗"
        value = criterion['value']
        if value is None:
            value_str = "N/A"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        logger.info(f"  {status_icon} {criterion_name}: {value_str} (threshold: {criterion['threshold']})")

    # Save results
    heldout_metrics_file = heldout_dir / 'heldout_validation_metrics.json'
    with open(heldout_metrics_file, 'w') as f:
        json.dump(heldout_metrics, f, indent=2)

    logger.info(f"\nSaved held-out metrics to: {heldout_metrics_file}")

    heldout_eval_file = heldout_dir / 'heldout_evaluation.json'
    with open(heldout_eval_file, 'w') as f:
        json.dump({
            'symbol': args.symbol,
            'heldout_period': HELDOUT_PERIOD,
            'final_params': final_params,
            'heldout_metrics': heldout_metrics,
            'walkforward_mean_metrics': wf_mean_metrics,
            'evaluation': heldout_eval,
        }, f, indent=2)

    logger.info(f"Saved held-out evaluation to: {heldout_eval_file}")

    logger.info(f"\n{'='*80}")
    logger.info(f"HELD-OUT VALIDATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Next step:")
    logger.info(f"  python research/finalize_strategy_decision.py --symbol {args.symbol}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
