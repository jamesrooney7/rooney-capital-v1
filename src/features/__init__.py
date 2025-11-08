"""
Feature engineering library for multi-alpha architecture.

This module contains shared indicator calculations and feature utilities
used across all strategies. By centralizing feature engineering:

1. Consistency: All strategies use the same indicator calculations
2. DRY: Don't duplicate complex logic across strategies
3. Testing: Easier to test indicator logic in isolation
4. Maintenance: Update once, affects all strategies

Key modules:
- indicators: Technical indicator calculations (IBS, RSI, ATR, etc.)
- filter_state: Filter value tracking and management
- safe_div: Safe division with zero-handling
"""

# Import key functions for easy access
from .indicators import normalize_column_name
from .filter_state import FilterColumn
from .safe_div import safe_div, monkey_patch_division

__all__ = [
    'normalize_column_name',
    'FilterColumn',
    'safe_div',
    'monkey_patch_division',
]
