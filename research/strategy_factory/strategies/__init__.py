"""
Strategy implementations for Strategy Factory.

Each strategy inherits from BaseStrategy and implements:
- entry_logic(): Boolean series indicating entry signals
- exit_logic(): Boolean indicating when to exit
- calculate_indicators(): Compute required indicators
- param_grid: Dictionary of parameters to optimize
"""

from .base import BaseStrategy

__all__ = ['BaseStrategy']
