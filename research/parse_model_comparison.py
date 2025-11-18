#!/usr/bin/env python3
"""
Parse model training logs and extract Phase 2 and Phase 3 metrics for comparison.

This script parses log files from train_rf_three_way_split.py runs and extracts:
- Phase 2 (Threshold Period 2019-2020): Sharpe, Sortino, PF, Trades, PnL
- Phase 3 (Test Period 2021-2024): Sharpe, Sortino, PF, Trades, PnL, CAGR, Max DD

Usage:
    python research/parse_model_comparison.py \
        --log-dir outputs/model_comparison/logs \
        --output-dir outputs/model_comparison
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_phase2_metrics(log_content: str) -> Optional[Dict]:
    """
    Parse Phase 2 (Threshold Period) metrics from log output.

    Looks for section:
        Phase 2 Complete - Fixed Threshold Validation:
        Threshold: 0.50 (FIXED - not optimized)
        Trades: 123
        Sharpe: 1.234
        Profit Factor: 1.567
        Win Rate: 55.5%
    """
    # Find Phase 2 section - look for the entire block
    # Skip the header and first separator line, capture until next separator or Phase 3
    phase2_pattern = r"Phase 2 Complete[^\n]*\n={50,}\n(.*?)(?:={50,}|Phase 3|$)"
    phase2_match = re.search(phase2_pattern, log_content, re.DOTALL)

    if not phase2_match:
        return None

    phase2_section = phase2_match.group(1)

    # Extract metrics
    metrics = {}

    # Threshold
    threshold_match = re.search(r"Threshold:\s*([\d.]+)", phase2_section)
    metrics['threshold'] = float(threshold_match.group(1)) if threshold_match else 0.0

    # Trades
    trades_match = re.search(r"Trades:\s*(\d+)", phase2_section)
    metrics['trades'] = int(trades_match.group(1)) if trades_match else 0

    # Sharpe
    sharpe_match = re.search(r"Sharpe:\s*([-\d.]+)", phase2_section)
    metrics['sharpe'] = float(sharpe_match.group(1)) if sharpe_match else 0.0

    # Profit Factor
    pf_match = re.search(r"Profit Factor:\s*([-\d.]+)", phase2_section)
    metrics['profit_factor'] = float(pf_match.group(1)) if pf_match else 0.0

    # Win Rate
    wr_match = re.search(r"Win Rate:\s*([-\d.]+)%", phase2_section)
    metrics['win_rate'] = float(wr_match.group(1)) if wr_match else 0.0

    # Calculate PnL estimate (not directly in Phase 2 output, set to 0)
    # Phase 2 doesn't print PnL, only Phase 3 does
    metrics['pnl_usd'] = 0.0

    # Add sortino (not in Phase 2 output)
    metrics['sortino'] = 0.0

    return metrics


def parse_phase3_metrics(log_content: str) -> Optional[Dict]:
    """
    Parse Phase 3 (Test Period) metrics from log output.

    Looks for section:
        Phase 3 Complete - TRUE OUT-OF-SAMPLE PERFORMANCE:
        Test Period: 2021-01-04 to 2024-11-15
        Trades: 123
        Sharpe Ratio: 1.234
        Sortino Ratio: 1.567
        Profit Factor: 1.890
        Win Rate: 55.5%
        Total PnL: $12,345.67
        CAGR: 15.5%
        Max Drawdown: 10.5%
    """
    # Find Phase 3 section - look for the entire block
    # Skip the header and first separator line, capture until next separator or end
    phase3_pattern = r"Phase 3 Complete[^\n]*\n={50,}\n(.*?)(?:={50,}|$)"
    phase3_match = re.search(phase3_pattern, log_content, re.DOTALL)

    if not phase3_match:
        return None

    phase3_section = phase3_match.group(1)

    # Extract metrics
    metrics = {}

    # Test Period
    period_match = re.search(r"Test Period:\s*([\d-]+)\s+to\s+([\d-]+)", phase3_section)
    if period_match:
        metrics['start_date'] = period_match.group(1)
        metrics['end_date'] = period_match.group(2)

    # Trades
    trades_match = re.search(r"Trades:\s*(\d+)", phase3_section)
    metrics['trades'] = int(trades_match.group(1)) if trades_match else 0

    # Sharpe Ratio
    sharpe_match = re.search(r"Sharpe Ratio:\s*([-\d.]+)", phase3_section)
    metrics['sharpe'] = float(sharpe_match.group(1)) if sharpe_match else 0.0

    # Sortino Ratio
    sortino_match = re.search(r"Sortino Ratio:\s*([-\d.]+)", phase3_section)
    metrics['sortino'] = float(sortino_match.group(1)) if sortino_match else 0.0

    # Profit Factor
    pf_match = re.search(r"Profit Factor:\s*([-\d.]+)", phase3_section)
    metrics['profit_factor'] = float(pf_match.group(1)) if pf_match else 0.0

    # Win Rate
    wr_match = re.search(r"Win Rate:\s*([-\d.]+)%", phase3_section)
    metrics['win_rate'] = float(wr_match.group(1)) if wr_match else 0.0

    # Total PnL (remove $ and commas)
    pnl_match = re.search(r"Total PnL:\s*\$?([-\d,]+\.?\d*)", phase3_section)
    if pnl_match:
        pnl_str = pnl_match.group(1).replace(',', '')
        metrics['pnl_usd'] = float(pnl_str)
    else:
        metrics['pnl_usd'] = 0.0

    # CAGR
    cagr_match = re.search(r"CAGR:\s*([-\d.]+)%", phase3_section)
    metrics['cagr'] = float(cagr_match.group(1)) if cagr_match else 0.0

    # Max Drawdown
    dd_match = re.search(r"Max Drawdown:\s*([-\d.]+)%", phase3_section)
    metrics['max_drawdown'] = float(dd_match.group(1)) if dd_match else 0.0

    return metrics


def parse_log_file(log_path: Path) -> Dict:
    """
    Parse a model training log file and extract both Phase 2 and Phase 3 metrics.
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        phase2 = parse_phase2_metrics(content)
        phase3 = parse_phase3_metrics(content)

        return {
            'log_file': log_path.name,
            'phase2': phase2,
            'phase3': phase3,
        }
    except Exception as e:
        print(f"Error parsing {log_path}: {e}", file=sys.stderr)
        return {
            'log_file': log_path.name,
            'phase2': None,
            'phase3': None,
            'error': str(e),
        }


def print_phase2_comparison(results: Dict[str, Dict]):
    """Print Phase 2 (Threshold Period) comparison table."""
    print("\n" + "="*80)
    print("PHASE 2 RESULTS (Threshold Period 2019-2020)")
    print("="*80)
    print(f"{'Model':<18} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'Trades':>7} {'PnL':>15}")
    print("-" * 80)

    # Sort by Sharpe ratio descending
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['phase2']['sharpe'] if x[1]['phase2'] else -999,
        reverse=True
    )

    for model_name, data in sorted_models:
        if not data['phase2']:
            print(f"{model_name:<18} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'N/A':>7} {'N/A':>15}")
            continue

        p2 = data['phase2']
        print(
            f"{model_name:<18} "
            f"{p2['sharpe']:>7.3f} "
            f"{p2['sortino']:>8.3f} "
            f"{p2['profit_factor']:>6.2f} "
            f"{p2['trades']:>7} "
            f"$ {p2['pnl_usd']:>13,.2f}"
        )

    print("-" * 80)

    # Find winner
    if sorted_models and sorted_models[0][1]['phase2']:
        winner_name, winner_data = sorted_models[0]
        p2 = winner_data['phase2']
        print(f"\n🏆 WINNER: {winner_name}")
        print(f"   Phase 2 Sharpe: {p2['sharpe']:.3f}")
        print(f"   Phase 2 PnL: ${p2['pnl_usd']:,.2f}")


def print_phase3_results(model_name: str, phase3: Dict):
    """Print Phase 3 (Test Period) results for winning model."""
    print("\n" + "="*80)
    print(f"PHASE 3 RESULTS (Test Period 2021-2024) - {model_name}")
    print("="*80)

    if not phase3:
        print("No Phase 3 results found.")
        return

    print(f"Sharpe Ratio:    {phase3['sharpe']:.3f}")
    print(f"Sortino Ratio:   {phase3['sortino']:.3f}")
    print(f"Profit Factor:   {phase3['profit_factor']:.3f}")
    print(f"Win Rate:        {phase3['win_rate']:.1f}%")
    print(f"Total PnL:       ${phase3['pnl_usd']:,.2f}")
    print(f"CAGR:            {phase3['cagr']:.1f}%")
    print(f"Max Drawdown:    {phase3['max_drawdown']:.1f}%")
    print(f"Trades:          {phase3['trades']}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Parse model training logs and compare Phase 2/3 results'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Directory containing model training log files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/model_comparison',
        help='Output directory for results JSON'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='ES',
        help='Trading symbol (e.g., ES, NQ)'
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}", file=sys.stderr)
        return 1

    # Find all log files (MODEL_*.log or similar)
    log_files = sorted(log_dir.glob("MODEL_*.log"))

    if not log_files:
        print(f"Error: No log files found in {log_dir}", file=sys.stderr)
        print("Expected files like: MODEL_A.log, MODEL_B.log, etc.", file=sys.stderr)
        return 1

    print(f"Found {len(log_files)} model log files")

    # Parse all logs
    results = {}
    for log_path in log_files:
        # Extract model name from filename (e.g., MODEL_A from MODEL_A.log)
        model_name = log_path.stem

        print(f"Parsing {model_name}...", end=' ')
        data = parse_log_file(log_path)

        if data['phase2'] or data['phase3']:
            print("✓")
            results[model_name] = data
        else:
            print("✗ (no metrics found)")

    if not results:
        print("\nError: No valid results parsed from any log file", file=sys.stderr)
        return 1

    # Print Phase 2 comparison
    print_phase2_comparison(results)

    # Find winner and print Phase 3 results
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['phase2']['sharpe'] if x[1]['phase2'] else -999,
        reverse=True
    )

    if sorted_models and sorted_models[0][1]['phase3']:
        winner_name, winner_data = sorted_models[0]
        print_phase3_results(winner_name, winner_data['phase3'])

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{sorted_models[0][0].lower()}_{args.symbol}_best.json"

    # Save winner's full data
    winner_name, winner_data = sorted_models[0]
    output_data = {
        'model': winner_name,
        'symbol': args.symbol,
        'phase2': winner_data['phase2'],
        'phase3': winner_data['phase3'],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Model selection complete!")
    print(f"   Best model JSON: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
