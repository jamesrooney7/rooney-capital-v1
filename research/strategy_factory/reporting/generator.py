"""
Report generation for Strategy Factory results.

Generates markdown reports with:
- Summary statistics
- Top strategies
- Filter breakdown
- Performance charts (text-based)
"""

from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from datetime import datetime


def generate_phase1_report(
    run_id: str,
    results: List[Dict[str, Any]],
    filter_stats: Dict[str, int],
    output_path: Path
) -> str:
    """
    Generate Phase 1 markdown report.

    Args:
        run_id: Execution run ID
        results: List of result dictionaries (winners)
        filter_stats: Dictionary of filter statistics
        output_path: Path to save report

    Returns:
        Path to generated report
    """
    report_lines = []

    # Header
    report_lines.append("# Strategy Factory - Phase 1 Results\n")
    report_lines.append(f"**Run ID**: `{run_id}`\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("\n---\n\n")

    # Executive Summary
    report_lines.append("## Executive Summary\n\n")
    report_lines.append(f"- **Total Backtests**: {filter_stats.get('total', 0):,}\n")
    report_lines.append(f"- **Strategies Passed All Filters**: {len(results)}\n")
    report_lines.append(f"- **Success Rate**: {len(results)/filter_stats.get('total', 1)*100:.1f}%\n")
    report_lines.append("\n")

    # Filter Funnel
    report_lines.append("## Filter Funnel\n\n")
    report_lines.append("| Filter | Passed | Pass Rate |\n")
    report_lines.append("|--------|--------|----------|\n")

    total = filter_stats.get('total', 1)
    for filter_name in ['gate1', 'walkforward', 'regime', 'stability', 'statistical']:
        passed = filter_stats.get(filter_name, 0)
        rate = (passed / total) * 100
        report_lines.append(f"| {filter_name.title()} | {passed:,} | {rate:.1f}% |\n")

    report_lines.append("\n")

    # Top Strategies
    if results:
        report_lines.append("## Winning Strategies\n\n")

        # Sort by Sharpe
        sorted_results = sorted(results, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)

        report_lines.append("| Rank | Strategy | ID | Sharpe | Trades | Win Rate | PF | Max DD |\n")
        report_lines.append("|------|----------|-----|--------|--------|----------|----|---------|\n")

        for i, result in enumerate(sorted_results, 1):
            report_lines.append(
                f"| {i} | {result['strategy_name']} | #{result['strategy_id']} | "
                f"{result['sharpe_ratio']:.3f} | {result['total_trades']:,} | "
                f"{result['win_rate']*100:.1f}% | {result['profit_factor']:.2f} | "
                f"{result['max_drawdown_pct']*100:.1f}% |\n"
            )

        report_lines.append("\n")

        # Detailed breakdown
        report_lines.append("## Strategy Details\n\n")

        for i, result in enumerate(sorted_results, 1):
            report_lines.append(f"### {i}. {result['strategy_name']} (#{result['strategy_id']})\n\n")
            report_lines.append(f"**Parameters**: `{result.get('params', 'N/A')}`\n\n")

            report_lines.append("**Performance Metrics**:\n")
            report_lines.append(f"- Sharpe Ratio: {result['sharpe_ratio']:.3f}\n")
            report_lines.append(f"- Total Trades: {result['total_trades']:,}\n")
            report_lines.append(f"- Win Rate: {result['win_rate']*100:.1f}%\n")
            report_lines.append(f"- Profit Factor: {result['profit_factor']:.2f}\n")
            report_lines.append(f"- Max Drawdown: {result['max_drawdown_pct']*100:.1f}%\n")
            report_lines.append(f"- Avg Bars Held: {result['avg_bars_held']:.1f}\n")
            report_lines.append("\n")

            report_lines.append("**P&L Breakdown**:\n")
            report_lines.append(f"- Total P&L: ${result['total_pnl']:,.2f}\n")
            report_lines.append(f"- Avg Win: ${result['avg_win']:.2f}\n")
            report_lines.append(f"- Avg Loss: ${result['avg_loss']:.2f}\n")
            report_lines.append(f"- Largest Win: ${result['largest_win']:.2f}\n")
            report_lines.append(f"- Largest Loss: ${result['largest_loss']:.2f}\n")
            report_lines.append("\n")

    else:
        report_lines.append("## ⚠️ No Winning Strategies\n\n")
        report_lines.append("No strategies passed all filters.\n\n")
        report_lines.append("**Recommendations**:\n")
        report_lines.append("1. Review filter thresholds - may be too strict\n")
        report_lines.append("2. Test different parameter ranges\n")
        report_lines.append("3. Try different symbols or timeframes\n")
        report_lines.append("4. Check data quality\n\n")

    # Next Steps
    report_lines.append("---\n\n")
    report_lines.append("## Next Steps\n\n")

    if results:
        report_lines.append("### Phase 2: Multi-Symbol Validation\n\n")
        report_lines.append("Run the winning strategies on multiple symbols:\n\n")
        report_lines.append("```bash\n")
        report_lines.append(f"python -m research.strategy_factory.main phase2 \\\n")
        report_lines.append(f"    --run-id {run_id} \\\n")
        report_lines.append(f"    --symbols ES NQ YM RTY GC SI \\\n")
        report_lines.append(f"    --workers 16\n")
        report_lines.append("```\n\n")

        report_lines.append("### Phase 3: ML Integration\n\n")
        report_lines.append("Extract features and train ML models:\n\n")
        report_lines.append("```bash\n")
        report_lines.append("# For each winning strategy\n")
        report_lines.append("python research/extract_training_data.py \\\n")
        report_lines.append("    --strategy <strategy_name> \\\n")
        report_lines.append("    --symbol ES \\\n")
        report_lines.append("    --start 2010-01-01 \\\n")
        report_lines.append("    --end 2024-12-31\n\n")
        report_lines.append("python research/train_rf_cpcv_bo.py \\\n")
        report_lines.append("    --symbol ES \\\n")
        report_lines.append("    --strategy <strategy_name>\n")
        report_lines.append("```\n\n")
    else:
        report_lines.append("Re-run Phase 1 with adjusted parameters or different data.\n\n")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(report_lines)

    return str(output_path)


def create_ascii_bar_chart(data: Dict[str, float], title: str, width: int = 40) -> str:
    """
    Create ASCII bar chart.

    Args:
        data: Dictionary of label -> value
        title: Chart title
        width: Maximum bar width

    Returns:
        ASCII art string
    """
    if not data:
        return f"{title}\n(No data)\n"

    lines = [f"\n{title}\n", "=" * 50 + "\n"]

    max_value = max(data.values())
    max_label_len = max(len(str(k)) for k in data.keys())

    for label, value in data.items():
        bar_length = int((value / max_value) * width) if max_value > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{label:>{max_label_len}}: {bar} {value:.2f}\n")

    return "".join(lines)


if __name__ == "__main__":
    """Test report generation."""
    from pathlib import Path

    # Sample data
    results = [
        {
            'strategy_name': 'RSI2_MeanReversion',
            'strategy_id': 21,
            'sharpe_ratio': 0.45,
            'total_trades': 12453,
            'win_rate': 0.58,
            'profit_factor': 1.32,
            'max_drawdown_pct': 0.18,
            'avg_bars_held': 4.2,
            'total_pnl': 25430.50,
            'avg_win': 145.30,
            'avg_loss': -98.20,
            'largest_win': 890.50,
            'largest_loss': -450.30,
            'params': "{'rsi_length': 2, 'rsi_oversold': 10}"
        }
    ]

    filter_stats = {
        'total': 235,
        'gate1': 150,
        'walkforward': 95,
        'regime': 60,
        'stability': 40,
        'statistical': 8
    }

    output_path = Path("test_report.md")
    generate_phase1_report("test_run_123", results, filter_stats, output_path)
    print(f"Test report generated: {output_path}")
