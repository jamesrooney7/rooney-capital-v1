"""
Strategy implementations for Strategy Factory.

Each strategy inherits from BaseStrategy and implements:
- entry_logic(): Boolean series indicating entry signals
- exit_logic(): Boolean indicating when to exit
- calculate_indicators(): Compute required indicators
- param_grid: Dictionary of parameters to optimize

Available Tier 1 Strategies:
1. BollingerBands (#1) - Mean Reversion - 16 combos
2. RSI2MeanReversion (#21) - Mean Reversion - 36 combos
3. RSI2SMAFilter (#36) - Mean Reversion - 81 combos
4. Double7s (#37) - Mean Reversion - 27 combos
5. MACross (#17) - Trend Following - 16 combos
6. VWAPReversion (#24) - Mean Reversion - 4 combos
7. GapFill (#23) - Mean Reversion - 7 combos
8. OpeningRangeBreakout (#25) - Breakout - 9 combos
9. MACDStrategy (#19) - Momentum - 27 combos
10. PriceChannelBreakout (#15) - Breakout - 12 combos

Total: 235 parameter combinations
"""

from .base import BaseStrategy
from .rsi2_mean_reversion import RSI2MeanReversion
from .bollinger_bands import BollingerBands
from .ma_cross import MACross
from .vwap_reversion import VWAPReversion
from .rsi2_sma_filter import RSI2SMAFilter
from .double_7s import Double7s
from .macd_strategy import MACDStrategy
from .price_channel_breakout import PriceChannelBreakout
from .gap_fill import GapFill
from .opening_range_breakout import OpeningRangeBreakout

__all__ = [
    'BaseStrategy',
    'RSI2MeanReversion',
    'BollingerBands',
    'MACross',
    'VWAPReversion',
    'RSI2SMAFilter',
    'Double7s',
    'MACDStrategy',
    'PriceChannelBreakout',
    'GapFill',
    'OpeningRangeBreakout'
]

# Strategy registry for easy lookup by ID
STRATEGY_REGISTRY = {
    1: BollingerBands,
    15: PriceChannelBreakout,
    17: MACross,
    19: MACDStrategy,
    21: RSI2MeanReversion,
    23: GapFill,
    24: VWAPReversion,
    25: OpeningRangeBreakout,
    36: RSI2SMAFilter,
    37: Double7s
}
