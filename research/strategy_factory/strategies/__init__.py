"""
Strategy implementations for Strategy Factory.

Each strategy inherits from BaseStrategy and implements:
- entry_logic(): Boolean series indicating entry signals
- exit_logic(): Boolean indicating when to exit
- calculate_indicators(): Compute required indicators
- param_grid: Dictionary of parameters to optimize

Tier 1 (Original 10):
1-10: BollingerBands, RSI2, MACD, etc. - 235 combos

Batch 1 (10 strategies):
2, 4, 7, 11, 12, 13, 16, 18, 22, 26 - ~400 combos

Batch 2 (10 strategies):
3, 5, 6, 8, 9, 10, 14, 20, 27, 28 - ~550 combos

Batch 3 (9 strategies):
29, 30, 31, 32, 33, 34, 35, 38, 39 - ~250 combos

Total Strategies: 39/39 (100% COMPLETE!)
Total Combinations: ~1,435
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
# Batch 2
from .three_bar_reversal import ThreeBarReversal
from .engulfing_pattern import EngulfingPattern
from .hammer_shooting_star import HammerShootingStar
from .parabolic_sar import ParabolicSAR
from .adx_trend_strength import ADXTrendStrength
from .ichimoku_cloud import IchimokuCloud
from .doji_reversal import DojiReversal
from .fibonacci_retracement import FibonacciRetracement
from .mean_reversion_bands import MeanReversionBands
from .atr_trailing_stop import ATRTrailingStop
# Batch 3
from .volume_breakout import VolumeBreakout
from .momentum_fade import MomentumFade
from .pivot_point_reversal import PivotPointReversal
from .roc_strategy import ROCStrategy
from .aroon_indicator import AroonIndicator
from .money_flow_index import MoneyFlowIndex
from .chaikin_oscillator import ChaikinOscillator
from .overnight_gap_strategy import OvernightGapStrategy
from .time_of_day_reversal import TimeOfDayReversal

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
    'InsideBarBreakout',
    'ThreeBarReversal',
    'EngulfingPattern',
    'HammerShootingStar',
    'ParabolicSAR',
    'ADXTrendStrength',
    'IchimokuCloud',
    'DojiReversal',
    'FibonacciRetracement',
    'MeanReversionBands',
    'ATRTrailingStop',
    'VolumeBreakout',
    'MomentumFade',
    'PivotPointReversal',
    'ROCStrategy',
    'AroonIndicator',
    'MoneyFlowIndex',
    'ChaikinOscillator',
    'OvernightGapStrategy',
    'TimeOfDayReversal'
]

# Strategy registry for easy lookup by ID
STRATEGY_REGISTRY = {
    1: BollingerBands,
    2: KeltnerChannelBreakout,
    3: ThreeBarReversal,
    4: SupportResistanceBounce,
    5: EngulfingPattern,
    6: HammerShootingStar,
    7: MovingAverageEnvelope,
    8: ParabolicSAR,
    9: ADXTrendStrength,
    10: IchimokuCloud,
    11: StochasticRSI,
    12: WilliamsPercentR,
    13: CCIStrategy,
    14: DojiReversal,
    15: PriceChannelBreakout,
    16: TurtleTrading,
    17: MACross,
    18: EMARibbon,
    19: MACDStrategy,
    20: FibonacciRetracement,
    21: RSI2MeanReversion,
    22: RSIDivergence,
    23: GapFill,
    24: VWAPReversion,
    25: OpeningRangeBreakout,
    26: InsideBarBreakout,
    27: MeanReversionBands,
    28: ATRTrailingStop,
    29: VolumeBreakout,
    30: MomentumFade,
    31: PivotPointReversal,
    32: ROCStrategy,
    33: AroonIndicator,
    34: MoneyFlowIndex,
    35: ChaikinOscillator,
    36: RSI2SMAFilter,
    37: Double7s,
    38: OvernightGapStrategy,
    39: TimeOfDayReversal
}
