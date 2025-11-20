"""
Engine components for Strategy Factory.

Core modules:
- data_loader: Load OHLCV data from local files
- backtester: Event-driven backtesting with proper position management
- optimizer: Parameter grid generation and parallel execution
- filters: Statistical filters for strategy selection
- statistics: Monte Carlo, walk-forward, regime analysis
"""

from .data_loader import load_data, load_multiple_symbols, get_available_symbols
from .backtester import Backtester, BacktestResults, Trade

__all__ = [
    'load_data',
    'load_multiple_symbols',
    'get_available_symbols',
    'Backtester',
    'BacktestResults',
    'Trade'
]
