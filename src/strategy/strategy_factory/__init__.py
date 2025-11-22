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

Phase B-D (Remaining 49 strategies - IN PROGRESS):
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

from .rsi2_mean_reversion_bt import RSI2MeanReversionBT
from .buy_on_5_bar_low_bt import BuyOn5BarLowBT
from .three_bar_low_bt import ThreeBarLowBT
from .double_7s_bt import Double7sBT
from .ma_cross_bt import MACrossBT

__all__ = [
    'RSI2MeanReversionBT',
    'BuyOn5BarLowBT',
    'ThreeBarLowBT',
    'Double7sBT',
    'MACrossBT',
]

# Strategy ID mapping (from Strategy Factory)
STRATEGY_ID_MAP = {
    21: RSI2MeanReversionBT,   # RSI2 Mean Reversion
    40: BuyOn5BarLowBT,        # Buy on 5 Bar Low
    41: ThreeBarLowBT,         # Three Bar Low
    37: Double7sBT,            # Double 7s
    17: MACrossBT,             # MA Cross
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
