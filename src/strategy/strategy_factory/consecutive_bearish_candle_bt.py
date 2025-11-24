#!/usr/bin/env python3
"""
Consecutive Bearish Candle Strategy - Backtrader Implementation

Entry: N consecutive bars where close < previous close (momentum exhaustion)
Exit: Close > high[1] (reversal confirmation)

Strategy ID: 47
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ConsecutiveBearishCandleBT(IbsStrategy):
    """Consecutive Bearish Candle with ML Meta-Labeling."""

    params = (
        ('consecutive_lookback', 3),  # Number of consecutive down closes needed

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        self.consecutive_down = 0

    def entry_conditions_met(self):
        if len(self.data) < 5:
            return False

        # Check if current close < previous close
        if self.data.close[0] < self.data.close[-1]:
            self.consecutive_down += 1
        else:
            self.consecutive_down = 0

        # Entry when consecutive down >= threshold
        return self.consecutive_down >= self.params.consecutive_lookback

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1]
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
