#!/usr/bin/env python3
"""
Bollinger Band IBS Reversal Strategy - Backtrader Implementation

Entry: Price < BB lower AND IBS < threshold (double oversold)
Exit: Price > BB middle or IBS > exit threshold

Strategy ID: 44
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class BBIBSReversalBT(IbsStrategy):
    """BB + IBS Reversal with ML Meta-Labeling."""

    params = (
        ('bb_period', 20),
        ('bb_std', 2.0),
        ('ibs_threshold', 0.3),

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        self.bbands = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_std
        )

        range_hl = self.data.high - self.data.low
        self.ibs = (self.data.close - self.data.low) / range_hl

    def entry_conditions_met(self):
        if len(self.data) < self.params.bb_period + 5:
            return False

        # Double confirmation: below BB lower AND low IBS
        entry = (self.data.close[0] < self.bbands.lines.bot[0] and
                self.ibs[0] < self.params.ibs_threshold)

        return entry

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit at BB middle or high IBS
        return self.data.close[0] > self.bbands.lines.mid[0]
