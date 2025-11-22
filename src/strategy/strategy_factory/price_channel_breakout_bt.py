#!/usr/bin/env python3
"""
Price Channel Breakout (Donchian Channel) Strategy - Backtrader Implementation

Entry: Price breaks above highest high of last N bars
Exit: Price breaks below lowest low of last N bars

Strategy ID: 15
Archetype: Breakout
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class PriceChannelBreakoutBT(IbsStrategy):
    """Price Channel Breakout with ML Meta-Labeling."""

    params = (
        ('channel_length', 20),         # Lookback period
        ('channel_breakout_pct', 0.0),  # Breakout buffer percentage

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Highest high and lowest low (exclude current bar)
        self.channel_high = bt.indicators.Highest(
            self.data.high(-1),  # Previous bar's high
            period=self.params.channel_length
        )

        self.channel_low = bt.indicators.Lowest(
            self.data.low(-1),  # Previous bar's low
            period=self.params.channel_length
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.channel_length + 5:
            return False

        # Apply breakout percentage buffer
        breakout_pct = self.params.channel_breakout_pct / 100.0
        channel_range = self.channel_high[0] - self.channel_low[0]
        channel_high_adj = self.channel_high[0] + (channel_range * breakout_pct)

        # Entry: Close > channel_high_adj
        return self.data.close[0] > channel_high_adj

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Apply breakout percentage buffer
        breakout_pct = self.params.channel_breakout_pct / 100.0
        channel_range = self.channel_high[0] - self.channel_low[0]
        channel_low_adj = self.channel_low[0] - (channel_range * breakout_pct)

        # Exit when price breaks below channel low
        return self.data.close[0] < channel_low_adj
