"""
Strategy implementations for Strategy Factory.

Each strategy inherits from BaseStrategy and implements:
- entry_logic(): Boolean series indicating entry signals
- exit_logic(): Boolean indicating when to exit
- calculate_indicators(): Compute required indicators
- param_grid: Dictionary of parameters to optimize

Tier 1 (Original 10):
1-10: BollingerBands, RSI2, MACD, etc. - 235 combos

Batch 1 (New 10):
2, 4, 7, 11, 12, 13, 16, 18, 22, 26 - ~400 combos

Total Strategies: 20
Total Combinations: ~635
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
# Batch 1
from .keltner_channel_breakout import KeltnerChannelBreakout
from .support_resistance_bounce import SupportResistanceBounce
from .moving_average_envelope import MovingAverageEnvelope
from .stochastic_rsi import StochasticRSI
from .williams_percent_r import WilliamsPercentR
from .cci_strategy import CCIStrategy
from .turtle_trading import TurtleTrading
from .ema_ribbon import EMARibbon
from .rsi_divergence import RSIDivergence
from .inside_bar_breakout import InsideBarBreakout

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
    'OpeningRangeBreakout',
    'KeltnerChannelBreakout',
    'SupportResistanceBounce',
    'MovingAverageEnvelope',
    'StochasticRSI',
    'WilliamsPercentR',
    'CCIStrategy',
    'TurtleTrading',
    'EMARibbon',
    'RSIDivergence',
    'InsideBarBreakout'
]

# Strategy registry for easy lookup by ID
STRATEGY_REGISTRY = {
    1: BollingerBands,
    2: KeltnerChannelBreakout,
    4: SupportResistanceBounce,
    7: MovingAverageEnvelope,
    11: StochasticRSI,
    12: WilliamsPercentR,
    13: CCIStrategy,
    15: PriceChannelBreakout,
    16: TurtleTrading,
    17: MACross,
    18: EMARibbon,
    19: MACDStrategy,
    21: RSI2MeanReversion,
    22: RSIDivergence,
    23: GapFill,
    24: VWAPReversion,
    25: OpeningRangeBreakout,
    26: InsideBarBreakout,
    36: RSI2SMAFilter,
    37: Double7s
}
