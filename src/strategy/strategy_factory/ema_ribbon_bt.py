#!/usr/bin/env python3
"""
EMA Ribbon Strategy - Backtrader Implementation

Entry: Fast EMA crosses above slow EMA with ribbon aligned
Exit: Fast EMA crosses below slow EMA

Strategy ID: 18
Archetype: Trend Following
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class EMARibbonBT(IbsStrategy):
    """EMA Ribbon with ML Meta-Labeling."""

    params = (
        ('ribbon_fast', 13),            # Fast EMA period
        ('ribbon_slow', 55),            # Slow EMA period

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Calculate EMAs
        self.ema_fast = bt.indicators.EMA(
            self.data.close,
            period=self.params.ribbon_fast
        )

        self.ema_slow = bt.indicators.EMA(
            self.data.close,
            period=self.params.ribbon_slow
        )

        # Middle EMAs for ribbon alignment
        mid1 = int((self.params.ribbon_fast + self.params.ribbon_slow) / 3)
        mid2 = int((self.params.ribbon_fast + self.params.ribbon_slow) * 2 / 3)

        self.ema_mid1 = bt.indicators.EMA(self.data.close, period=mid1)
        self.ema_mid2 = bt.indicators.EMA(self.data.close, period=mid2)

    def entry_conditions_met(self):
        if len(self.data) < self.params.ribbon_slow + 20:
            return False

        # Fast crosses above slow
        cross_above = self.ema_fast[0] > self.ema_slow[0]

        if len(self.data) > 1:
            was_below = self.ema_fast[-1] <= self.ema_slow[-1]
        else:
            was_below = False

        # Ribbon alignment: EMAs in ascending order (bullish)
        ribbon_aligned = (
            (self.ema_fast[0] > self.ema_mid1[0]) and
            (self.ema_mid1[0] > self.ema_mid2[0]) and
            (self.ema_mid2[0] > self.ema_slow[0])
        )

        return cross_above and was_below and ribbon_aligned

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when fast crosses below slow
        return self.ema_fast[0] < self.ema_slow[0]
