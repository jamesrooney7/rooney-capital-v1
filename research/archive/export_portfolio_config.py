#!/usr/bin/env python3
"""Export greedy optimizer results to portfolio configuration JSON.

Converts the CSV output from portfolio_optimizer_greedy.py into a
JSON configuration file for use in live trading.

Usage:
    python research/export_portfolio_config.py \
        --results results/greedy_optimization_results.csv \
        --output config/portfolio_optimization.json
"""

import argparse
import json
import ast
from pathlib import Path
from datetime import datetime
import pandas as pd


def export_portfolio_config(
    results_csv: Path,
    output_json: Path,
    select_row: int = 0
) -> None:
    """Export optimization results to JSON config.

    Args:
        results_csv: Path to greedy_optimization_results.csv
        output_json: Path to output JSON file
        select_row: Which row to export (0 = best/first row)
    """
    # Load results
    df = pd.read_csv(results_csv)

    if len(df) == 0:
        raise ValueError(f"No results found in {results_csv}")

    if select_row >= len(df):
        raise ValueError(f"Row {select_row} not found (only {len(df)} rows)")

    # Get selected configuration
    row = df.iloc[select_row]

    # Parse symbols list (stored as string representation of list)
    symbols_str = row['symbols']
    try:
        symbols = ast.literal_eval(symbols_str)
    except:
        # Fallback: try json.loads
        symbols = json.loads(symbols_str)

    # Create configuration
    config = {
        "optimization_metadata": {
            "generated_date": datetime.now().isoformat(),
            "optimization_period": "2023-2024",
            "source_file": str(results_csv),
            "selected_row": select_row,
            "total_configurations": len(df)
        },
        "portfolio_constraints": {
            "max_positions": int(row['max_positions']),
            "daily_stop_loss": 2500.0,
            "symbols": symbols,
            "n_symbols": int(row['n_symbols'])
        },
        "expected_performance": {
            "sharpe_ratio": float(row['sharpe']),
            "cagr": float(row['cagr']),
            "total_return_dollars": float(row['total_return']),
            "max_drawdown_dollars": float(row['max_dd']),
            "breach_events": int(row['breach_events']),
            "breach_periods": int(row['breach_periods']),
            "pct_time_in_breach": float(row['pct_time_in_breach']),
            "daily_stops_hit": int(row['daily_stops']),
            "profit_factor": float(row['profit_factor']),
            "avg_positions": float(row['avg_positions'])
        },
        "notes": [
            "This configuration was generated from greedy portfolio optimization",
            "Optimization period: 2023-2024 (1.99 years, 6,588 trades)",
            f"Selected configuration: max_positions={int(row['max_positions'])}, {int(row['n_symbols'])} symbols",
            "WARNING: Results are optimized on the same data they were evaluated on (look-ahead bias)",
            "RECOMMENDATION: Paper trade for 2-4 weeks before going live with real money"
        ]
    }

    # Create output directory if needed
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_json, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Portfolio configuration exported to: {output_json}")
    print(f"\nConfiguration Summary:")
    print(f"  Max Positions: {config['portfolio_constraints']['max_positions']}")
    print(f"  Symbols ({config['portfolio_constraints']['n_symbols']}): {', '.join(config['portfolio_constraints']['symbols'])}")
    print(f"  Expected Sharpe: {config['expected_performance']['sharpe_ratio']:.3f}")
    print(f"  Expected CAGR: {config['expected_performance']['cagr']*100:.2f}%")
    print(f"  Expected MaxDD: ${config['expected_performance']['max_drawdown_dollars']:,.0f}")
    print(f"  DD Breach Events: {config['expected_performance']['breach_events']}")


def main():
    parser = argparse.ArgumentParser(
        description='Export portfolio optimization results to JSON config'
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to greedy_optimization_results.csv'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/portfolio_optimization.json'),
        help='Output JSON file path'
    )
    parser.add_argument(
        '--row',
        type=int,
        default=0,
        help='Which row to export (0 = best, default: 0)'
    )

    args = parser.parse_args()

    export_portfolio_config(
        results_csv=args.results,
        output_json=args.output,
        select_row=args.row
    )


if __name__ == '__main__':
    main()
