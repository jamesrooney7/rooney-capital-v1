#!/usr/bin/env python3
"""
Keltner Channel Breakout Strategy - Backtrader Implementation

Entry: Price breaks above upper Keltner Channel
Exit: Price touches middle band (EMA)

Keltner Channels = EMA ± (ATR × multiplier)

Strategy ID: 2
Archetype: Breakout
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class KeltnerChannelBreakoutBT(IbsStrategy):
    """Keltner Channel Breakout with ML Meta-Labeling."""

    params = (
        ('kc_length', 20),              # EMA period
        ('kc_atr_length', 14),          # ATR period
        ('kc_multiplier', 2.0),         # ATR multiplier

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Middle band = EMA
        self.kc_middle = bt.indicators.EMA(
            self.data.close,
            period=self.params.kc_length
        )

        # ATR for channel width
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.params.kc_atr_length
        )

        # Upper and lower bands
        self.kc_upper = self.kc_middle + (self.atr * self.params.kc_multiplier)
        self.kc_lower = self.kc_middle - (self.atr * self.params.kc_multiplier)

        self.prev_close = None

    def entry_conditions_met(self):
        if len(self.data) < max(self.params.kc_length, self.params.kc_atr_length) + 10:
            return False

        # Entry: Close breaks above upper channel
        # Current bar: Close > upper
        # Previous bar: Close <= upper
        breakout = self.data.close[0] > self.kc_upper[0]

        if len(self.data) > 1:
            was_below = self.data.close[-1] <= self.kc_upper[-1]
        else:
            was_below = False

        return breakout and was_below

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when price touches or crosses below middle band
        return self.data.close[0] <= self.kc_middle[0]
