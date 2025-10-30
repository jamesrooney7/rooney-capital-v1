"""Performance evaluation metrics for trading strategies.

This package provides advanced statistical metrics for evaluating trading strategies,
with corrections for multiple testing bias and overfitting.

Main components:
- calculate_deflated_sharpe_ratio: DSR calculation with multiple testing correction
- get_required_sharpe: Minimum Sharpe for statistical significance
- validate_sharpe_significance: Automated significance testing
- print_dsr_analysis: Comprehensive DSR report
"""

from .performance_metrics import (
    calculate_deflated_sharpe_ratio,
    get_required_sharpe,
    print_dsr_analysis,
    validate_sharpe_significance,
)

__all__ = [
    "calculate_deflated_sharpe_ratio",
    "get_required_sharpe",
    "validate_sharpe_significance",
    "print_dsr_analysis",
]
