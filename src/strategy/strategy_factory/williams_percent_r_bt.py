#!/usr/bin/env python3
"""
Williams %R Strategy - Backtrader Implementation

Entry: %R crosses above oversold (-80)
Exit: %R crosses below overbought (-20)

Strategy ID: 12
Archetype: Momentum
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class WilliamsPercentRBT(IbsStrategy):
    """Williams %R with ML Meta-Labeling."""

    params = (
        ('wr_length', 14),              # Lookback period
        ('wr_oversold', -80),           # Oversold level
        ('wr_overbought', -20),         # Overbought level

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Williams %R
        self.williams_r = bt.indicators.WilliamsR(
            self.data,
            period=self.params.wr_length
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.wr_length + 10:
            return False

        # Entry: %R crosses above oversold
        above_oversold = self.williams_r[0] > self.params.wr_oversold

        if len(self.data) > 1:
            was_below = self.williams_r[-1] <= self.params.wr_oversold
        else:
            was_below = False

        return above_oversold and was_below

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when %R crosses below overbought
        if len(self.data) > 1:
            below_overbought = self.williams_r[0] < self.params.wr_overbought
            was_above = self.williams_r[-1] >= self.params.wr_overbought
            return below_overbought and was_above

        return False
