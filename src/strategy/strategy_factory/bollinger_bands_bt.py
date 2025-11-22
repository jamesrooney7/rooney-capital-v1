#!/usr/bin/env python3
"""
Bollinger Bands Mean Reversion Strategy - Backtrader Implementation

Entry: Close < Lower Bollinger Band (oversold)
Exit: Close > Middle Band (return to mean)

Strategy ID: 1
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class BollingerBandsBT(IbsStrategy):
    """Bollinger Bands with ML Meta-Labeling."""

    params = (
        ('bb_length', 20),              # Bollinger Band period
        ('bb_stddev', 2.0),             # Standard deviations

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Bollinger Bands
        self.bbands = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_length,
            devfactor=self.params.bb_stddev
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.bb_length + 10:
            return False

        # Entry: Close below lower Bollinger Band
        return self.data.close[0] < self.bbands.lines.bot[0]

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when price returns to middle band
        return self.data.close[0] > self.bbands.lines.mid[0]
