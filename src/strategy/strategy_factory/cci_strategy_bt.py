#!/usr/bin/env python3
"""
CCI (Commodity Channel Index) Strategy - Backtrader Implementation

Entry: CCI crosses above oversold (-100)
Exit: CCI crosses below overbought (+100)

Strategy ID: 13
Archetype: Momentum
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class CCIStrategyBT(IbsStrategy):
    """CCI with ML Meta-Labeling."""

    params = (
        ('cci_length', 20),             # CCI period
        ('cci_oversold', -100),         # Oversold level
        ('cci_overbought', 100),        # Overbought level

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # CCI
        self.cci = bt.indicators.CCI(
            self.data,
            period=self.params.cci_length
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.cci_length + 10:
            return False

        # Entry: CCI crosses above oversold
        above_oversold = self.cci[0] > self.params.cci_oversold

        if len(self.data) > 1:
            was_below = self.cci[-1] <= self.params.cci_oversold
        else:
            was_below = False

        return above_oversold and was_below

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when CCI crosses below overbought
        if len(self.data) > 1:
            below_overbought = self.cci[0] < self.params.cci_overbought
            was_above = self.cci[-1] >= self.params.cci_overbought
            return below_overbought and was_above

        return False
