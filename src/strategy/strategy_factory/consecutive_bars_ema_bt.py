#!/usr/bin/env python3
"""
Consecutive Bars Below EMA Strategy - Backtrader Implementation

Entry: Close below MA for N consecutive bars (buy the dip)
Exit: Close > high[1] (reversal confirmation)

Strategy ID: 48
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ConsecutiveBarsEMABT(IbsStrategy):
    """Consecutive Bars Below EMA with ML Meta-Labeling."""

    params = (
        ('consecutive_bars_threshold', 3),  # Consecutive bars below MA
        ('ma_type', 'SMA'),                 # 'SMA' or 'EMA'
        ('ma_length', 10),                  # MA period

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Moving average
        if self.params.ma_type == 'EMA':
            self.ma = bt.indicators.EMA(
                self.data.close,
                period=self.params.ma_length
            )
        else:  # SMA
            self.ma = bt.indicators.SMA(
                self.data.close,
                period=self.params.ma_length
            )

        self.consecutive_below = 0

    def entry_conditions_met(self):
        if len(self.data) < self.params.ma_length + 5:
            return False

        # Check if close < MA
        if self.data.close[0] < self.ma[0]:
            self.consecutive_below += 1
        else:
            self.consecutive_below = 0

        # Entry when consecutive below >= threshold
        return self.consecutive_below >= self.params.consecutive_bars_threshold

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1]
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
