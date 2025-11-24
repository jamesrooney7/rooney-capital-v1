#!/usr/bin/env python3
"""
MACD Crossover Strategy - Backtrader Implementation

Entry: MACD line crosses above signal line
Exit: MACD line crosses below signal line

Strategy ID: 19
Archetype: Momentum
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class MACDStrategyBT(IbsStrategy):
    """MACD Crossover with ML Meta-Labeling."""

    params = (
        ('macd_fast', 12),              # Fast EMA period
        ('macd_slow', 26),              # Slow EMA period
        ('macd_signal', 9),             # Signal line period

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.macd_slow + self.params.macd_signal + 10:
            return False

        # Entry: MACD line crosses above signal line
        macd_above = self.macd.macd[0] > self.macd.signal[0]

        if len(self.data) > 1:
            was_below = self.macd.macd[-1] <= self.macd.signal[-1]
        else:
            was_below = False

        return macd_above and was_below

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when MACD crosses below signal line
        return self.macd.macd[0] < self.macd.signal[0]
