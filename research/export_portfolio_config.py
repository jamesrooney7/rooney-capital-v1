#!/usr/bin/env python3
"""Export portfolio optimization results to live trading config format.

Converts results/optimal_portfolio.json (optimizer output) to
config/portfolio_optimization.json (live system input).

Usage:
    python research/export_portfolio_config.py
    python research/export_portfolio_config.py --input results/optimal_portfolio.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys


def export_portfolio_config(
    input_json: Path,
    output_json: Path,
    daily_stop_loss: float = None
) -> None:
    """Convert optimization results to live config format.

    Args:
        input_json: Path to optimization results JSON
        output_json: Path to output live config JSON
        daily_stop_loss: Override daily stop loss (uses 2500 default if None)
    """
    # Check input exists
    if not input_json.exists():
        print(f"❌ Error: Input file not found: {input_json}")
        print(f"\nRun portfolio optimization first:")
        print(f"  python research/portfolio_simulator_three_way.py \\")
        print(f"    --greedy-selection \\")
        print(f"    --portfolio-train-start 2019-01-01 \\")
        print(f"    --portfolio-train-end 2021-12-31 \\")
        print(f"    --output results/optimal_portfolio.json")
        sys.exit(1)

    # Load optimization results
    print(f"Loading optimization results from: {input_json}")
    with open(input_json) as f:
        opt_results = json.load(f)

    # Extract required fields
    symbols = opt_results.get('symbols', [])
    max_positions = opt_results.get('max_positions')

    # Use provided daily stop loss or default to 2500
    if daily_stop_loss is None:
        daily_stop_loss = 2500.0

    if not symbols or max_positions is None:
        print(f"❌ Error: Invalid optimization results format")
        print(f"   Expected fields: 'symbols', 'max_positions'")
        sys.exit(1)

    # Convert to live trading format
    live_config = {
        "optimization_metadata": {
            "generated_date": datetime.now().isoformat(),
            "source_file": str(input_json),
            "portfolio_train_period": opt_results.get('portfolio_train_period', {}),
        },
        "portfolio_constraints": {
            "symbols": symbols,
            "max_positions": max_positions,
            "daily_stop_loss": daily_stop_loss
        }
    }

    # Add test performance if available
    if 'test_period' in opt_results:
        test_metrics = opt_results['test_period'].get('metrics', {})
        live_config['optimization_metadata']['test_period'] = opt_results['test_period']
        live_config['expected_performance'] = {
            "sharpe_ratio": test_metrics.get('sharpe', 0),
            "sortino_ratio": test_metrics.get('sortino', 0),
            "cagr": test_metrics.get('cagr', 0),
            "total_return_dollars": test_metrics.get('total_return', 0),
            "max_drawdown_dollars": test_metrics.get('max_drawdown', 0),
            "max_drawdown_pct": test_metrics.get('max_drawdown_pct', 0),
            "profit_factor": test_metrics.get('profit_factor', 0),
            "win_rate": test_metrics.get('win_rate', 0),
            "avg_positions": test_metrics.get('avg_positions', 0),
            "daily_stops_hit": test_metrics.get('daily_stops_hit', 0)
        }

    # Create output directory if needed
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Save to live config location
    print(f"Exporting to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(live_config, f, indent=2)

    # Display configuration
    print(f"\n{'='*70}")
    print("✅ Portfolio Configuration Exported Successfully!")
    print(f"{'='*70}")
    print(f"\nSymbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"Max Positions: {max_positions}")
    print(f"Daily Stop Loss: ${daily_stop_loss:,.0f}")

    # Show test period metrics if available
    if 'test_period' in opt_results:
        test_metrics = opt_results['test_period'].get('metrics', {})
        test_start = opt_results['test_period'].get('start', 'Unknown')
        test_end = opt_results['test_period'].get('end', 'Unknown')

        if test_metrics:
            print(f"\n{'='*70}")
            print(f"Test Period Performance ({test_start} to {test_end}):")
            print(f"{'='*70}")
            print(f"Sharpe:        {test_metrics.get('sharpe', 0):.3f}")
            print(f"Sortino:       {test_metrics.get('sortino', 0):.3f}")
            print(f"CAGR:          {test_metrics.get('cagr', 0)*100:.2f}%")
            print(f"Max Drawdown:  ${test_metrics.get('max_drawdown', 0):,.0f} ({test_metrics.get('max_drawdown_pct', 0)*100:.2f}%)")
            print(f"Profit Factor: {test_metrics.get('profit_factor', 0):.2f}")
            print(f"Win Rate:      {test_metrics.get('win_rate', 0)*100:.2f}%")
            print(f"Total Return:  ${test_metrics.get('total_return', 0):,.0f}")

    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print("1. Review the configuration above")
    print("2. Verify all optimized symbols have models in src/models/")
    print("3. Start live worker with this config")
    print("4. Monitor first few days closely")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export portfolio optimization results to live trading config"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('results/optimal_portfolio.json'),
        help='Input optimization results JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/portfolio_optimization.json'),
        help='Output live config JSON file'
    )
    parser.add_argument(
        '--daily-stop-loss',
        type=float,
        default=2500.0,
        help='Daily stop loss in USD (default: 2500)'
    )

    args = parser.parse_args()

    export_portfolio_config(
        input_json=args.input,
        output_json=args.output,
        daily_stop_loss=args.daily_stop_loss
    )


if __name__ == '__main__':
    main()
