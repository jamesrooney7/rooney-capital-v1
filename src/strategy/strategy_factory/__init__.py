"""
Strategy Factory - Backtrader Implementations

Production-ready strategies ported from Strategy Factory for ML meta-labeling.

All strategies inherit from IbsStrategy to get:
- collect_filter_values() for 50+ ML features
- ML meta-labeling integration
- ATR-based risk management
- Time-based stops
- End-of-day exits

Phase A (Simple Strategies - COMPLETE):
- RSI2MeanReversionBT: Classic Larry Connors mean reversion
- BuyOn5BarLowBT: N-bar low breakout reversal
- ThreeBarLowBT: 3-bar low with SMA exit
- Double7sBT: 7-period low/high system
- MACrossBT: Moving average crossover trend following

Phase B (Mean Reversion & Calendar - COMPLETE):
- GapDownReversalBT: Gap down reversal pattern
- TurnOfMonthBT: Calendar-based monthly cycle
- BBIBSReversalBT: Bollinger Band + IBS double confirmation
- IBSStrategyBT: Classic IBS mean reversion
- IBSExtremeBT: Extreme IBS levels
- FourBarMomentumReversalBT: N-bar momentum exhaustion
- ConsecutiveBearishCandleBT: Consecutive down closes
- ConsecutiveBarsEMABT: Consecutive bars below MA
- AvgHLRangeIBSBT: Volatility + IBS combo
- ThreeDownThreeUpBT: Symmetric entry/exit pattern
- ConsecutiveCloseLowBT: Aggressive selling pressure detection
- ATRBuyDipBT: ATR-based dynamic dip buying

Phase C (Indicator-based Strategies - COMPLETE):
- BollingerBandsBT: Classic BB mean reversion
- KeltnerChannelBreakoutBT: Volatility breakout strategy
- MovingAverageEnvelopeBT: MA envelope mean reversion
- StochasticRSIBT: Stochastic RSI momentum
- WilliamsPercentRBT: Williams %R oscillator
- CCIStrategyBT: Commodity Channel Index
- PriceChannelBreakoutBT: Donchian Channel breakout
- EMARibbonBT: Multi-EMA trend following
- MACDStrategyBT: MACD crossover system
- RSIDivergenceBT: RSI divergence trading
- RSI2SMAFilterBT: Larry Connors RSI2 + 200 SMA
- TenBarHighBreakoutBT: Breakout with IBS filter

Phase D (Pattern Recognition & Advanced Indicators - COMPLETE):
- ThreeBarReversalBT: 3-bar reversal pattern
- SupportResistanceBounceBT: S/R level bounce strategy
- EngulfingPatternBT: Bullish engulfing candlestick
- HammerShootingStarBT: Hammer pattern recognition
- ParabolicSARBT: Parabolic SAR trend following
- ADXTrendStrengthBT: ADX trend strength system
- IchimokuCloudBT: Ichimoku Cloud strategy
- DojiReversalBT: Doji candlestick reversal
- TurtleTradingBT: Turtle Trading breakout system
- FibonacciRetracementBT: Fibonacci retracement levels
- GapFillBT: Gap fill mean reversion
- VWAPReversionBT: VWAP deviation strategy

Phase E (Remaining 13 strategies - IN PROGRESS):
- Coming soon...

Usage:
    from strategy.strategy_factory.rsi2_mean_reversion_bt import RSI2MeanReversionBT

    # Use in backtest
    cerebro.addstrategy(RSI2MeanReversionBT,
                       rsi_length=2,
                       rsi_oversold=10,
                       rsi_overbought=65)

    # Or with ML filter
    cerebro.addstrategy(RSI2MeanReversionBT,
                       enable_ml_filter=True,
                       ml_model_path='models/ES_21_model.pkl')
"""

# Phase A imports
from .rsi2_mean_reversion_bt import RSI2MeanReversionBT
from .buy_on_5_bar_low_bt import BuyOn5BarLowBT
from .three_bar_low_bt import ThreeBarLowBT
from .double_7s_bt import Double7sBT
from .ma_cross_bt import MACrossBT

# Phase B imports
from .gap_down_reversal_bt import GapDownReversalBT
from .turn_of_month_bt import TurnOfMonthBT
from .bb_ibs_reversal_bt import BBIBSReversalBT
from .ibs_strategy_bt import IBSStrategyBT
from .ibs_extreme_bt import IBSExtremeBT
from .four_bar_momentum_reversal_bt import FourBarMomentumReversalBT
from .consecutive_bearish_candle_bt import ConsecutiveBearishCandleBT
from .consecutive_bars_ema_bt import ConsecutiveBarsEMABT
from .avg_hl_range_ibs_bt import AvgHLRangeIBSBT
from .three_down_three_up_bt import ThreeDownThreeUpBT
from .consecutive_close_low_bt import ConsecutiveCloseLowBT
from .atr_buy_dip_bt import ATRBuyDipBT

# Phase C imports
from .bollinger_bands_bt import BollingerBandsBT
from .keltner_channel_breakout_bt import KeltnerChannelBreakoutBT
from .moving_average_envelope_bt import MovingAverageEnvelopeBT
from .stochastic_rsi_bt import StochasticRSIBT
from .williams_percent_r_bt import WilliamsPercentRBT
from .cci_strategy_bt import CCIStrategyBT
from .price_channel_breakout_bt import PriceChannelBreakoutBT
from .ema_ribbon_bt import EMARibbonBT
from .macd_strategy_bt import MACDStrategyBT
from .rsi_divergence_bt import RSIDivergenceBT
from .rsi2_sma_filter_bt import RSI2SMAFilterBT
from .ten_bar_high_breakout_bt import TenBarHighBreakoutBT

# Phase D imports
from .three_bar_reversal_bt import ThreeBarReversalBT
from .support_resistance_bounce_bt import SupportResistanceBounceBT
from .engulfing_pattern_bt import EngulfingPatternBT
from .hammer_shooting_star_bt import HammerShootingStarBT
from .parabolic_sar_bt import ParabolicSARBT
from .adx_trend_strength_bt import ADXTrendStrengthBT
from .ichimoku_cloud_bt import IchimokuCloudBT
from .doji_reversal_bt import DojiReversalBT
from .turtle_trading_bt import TurtleTradingBT
from .fibonacci_retracement_bt import FibonacciRetracementBT
from .gap_fill_bt import GapFillBT
from .vwap_reversion_bt import VWAPReversionBT

__all__ = [
    # Phase A
    'RSI2MeanReversionBT',
    'BuyOn5BarLowBT',
    'ThreeBarLowBT',
    'Double7sBT',
    'MACrossBT',
    # Phase B
    'GapDownReversalBT',
    'TurnOfMonthBT',
    'BBIBSReversalBT',
    'IBSStrategyBT',
    'IBSExtremeBT',
    'FourBarMomentumReversalBT',
    'ConsecutiveBearishCandleBT',
    'ConsecutiveBarsEMABT',
    'AvgHLRangeIBSBT',
    'ThreeDownThreeUpBT',
    'ConsecutiveCloseLowBT',
    'ATRBuyDipBT',
    # Phase C
    'BollingerBandsBT',
    'KeltnerChannelBreakoutBT',
    'MovingAverageEnvelopeBT',
    'StochasticRSIBT',
    'WilliamsPercentRBT',
    'CCIStrategyBT',
    'PriceChannelBreakoutBT',
    'EMARibbonBT',
    'MACDStrategyBT',
    'RSIDivergenceBT',
    'RSI2SMAFilterBT',
    'TenBarHighBreakoutBT',
    # Phase D
    'ThreeBarReversalBT',
    'SupportResistanceBounceBT',
    'EngulfingPatternBT',
    'HammerShootingStarBT',
    'ParabolicSARBT',
    'ADXTrendStrengthBT',
    'IchimokuCloudBT',
    'DojiReversalBT',
    'TurtleTradingBT',
    'FibonacciRetracementBT',
    'GapFillBT',
    'VWAPReversionBT',
]

# Strategy ID mapping (from Strategy Factory)
STRATEGY_ID_MAP = {
    # Phase A
    17: MACrossBT,             # MA Cross
    21: RSI2MeanReversionBT,   # RSI2 Mean Reversion
    37: Double7sBT,            # Double 7s
    40: BuyOn5BarLowBT,        # Buy on 5 Bar Low
    41: ThreeBarLowBT,         # Three Bar Low
    # Phase B
    42: GapDownReversalBT,     # Gap Down Reversal
    43: TurnOfMonthBT,         # Turn of Month
    44: BBIBSReversalBT,       # BB + IBS Reversal
    45: IBSStrategyBT,         # IBS Strategy
    46: FourBarMomentumReversalBT,  # Four Bar Momentum Reversal
    47: ConsecutiveBearishCandleBT,  # Consecutive Bearish Candle
    48: ConsecutiveBarsEMABT,  # Consecutive Bars EMA
    49: AvgHLRangeIBSBT,       # Avg HL Range + IBS
    50: ThreeDownThreeUpBT,    # Three Down Three Up
    51: ConsecutiveCloseLowBT,  # Consecutive Close Low
    52: ATRBuyDipBT,           # ATR Buy Dip
    53: IBSExtremeBT,          # IBS Extreme
    # Phase C
    1: BollingerBandsBT,       # Bollinger Bands
    2: KeltnerChannelBreakoutBT,  # Keltner Channel Breakout
    7: MovingAverageEnvelopeBT,  # Moving Average Envelope
    11: StochasticRSIBT,       # Stochastic RSI
    12: WilliamsPercentRBT,    # Williams %R
    13: CCIStrategyBT,         # CCI Strategy
    15: PriceChannelBreakoutBT,  # Price Channel Breakout
    18: EMARibbonBT,           # EMA Ribbon
    19: MACDStrategyBT,        # MACD Strategy
    22: RSIDivergenceBT,       # RSI Divergence
    36: RSI2SMAFilterBT,       # RSI2 + SMA Filter
    54: TenBarHighBreakoutBT,  # 10 Bar High Breakout
    # Phase D
    3: ThreeBarReversalBT,     # Three Bar Reversal
    4: SupportResistanceBounceBT,  # Support/Resistance Bounce
    5: EngulfingPatternBT,     # Engulfing Pattern
    6: HammerShootingStarBT,   # Hammer/Shooting Star
    8: ParabolicSARBT,         # Parabolic SAR
    9: ADXTrendStrengthBT,     # ADX Trend Strength
    10: IchimokuCloudBT,       # Ichimoku Cloud
    14: DojiReversalBT,        # Doji Reversal
    16: TurtleTradingBT,       # Turtle Trading
    20: FibonacciRetracementBT,  # Fibonacci Retracement
    23: GapFillBT,             # Gap Fill
    24: VWAPReversionBT,       # VWAP Reversion
}


def get_strategy_by_id(strategy_id: int):
    """
    Get strategy class by Strategy Factory ID.

    Args:
        strategy_id: Strategy ID from Strategy Factory

    Returns:
        Strategy class

    Raises:
        KeyError: If strategy ID not yet ported
    """
    if strategy_id not in STRATEGY_ID_MAP:
        raise KeyError(
            f"Strategy ID {strategy_id} not yet ported to Backtrader. "
            f"Available IDs: {list(STRATEGY_ID_MAP.keys())}"
        )

    return STRATEGY_ID_MAP[strategy_id]
