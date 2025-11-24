#!/usr/bin/env python3
"""
10 Bar High Breakout with Low IBS Strategy - Backtrader Implementation

Entry: High > highest high of previous N bars AND IBS <= threshold
Exit: Close > high[1]

Strategy ID: 54
Archetype: Breakout
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class TenBarHighBreakoutBT(IbsStrategy):
    """10 Bar High Breakout with Low IBS and ML Meta-Labeling."""

    params = (
        ('lookback_period', 10),        # Bars for highest high
        ('ibs_threshold', 0.15),        # IBS threshold
        ('ema_period', 200),            # EMA filter period
        ('use_ema_filter', False),      # Enable EMA filter

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Highest high of previous N bars (exclude current bar)
        self.highest_high = bt.indicators.Highest(
            self.data.high(-1),  # Previous bar's high
            period=self.params.lookback_period
        )

        # IBS
        range_hl = self.data.high - self.data.low
        self.ibs = (self.data.close - self.data.low) / range_hl

        # EMA filter
        self.ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_period
        )

    def entry_conditions_met(self):
        if len(self.data) < max(self.params.ema_period, self.params.lookback_period) + 10:
            return False

        # Condition 1: High breaks above highest high (breakout)
        breakout = self.data.high[0] > self.highest_high[0]

        # Condition 2: Low IBS (pullback/consolidation)
        low_ibs = self.ibs[0] <= self.params.ibs_threshold

        # Entry: Both conditions met
        entry = breakout and low_ibs

        # Optional EMA filter: only enter if price above EMA
        if self.params.use_ema_filter:
            entry = entry and (self.data.close[0] > self.ema[0])

        return entry

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1] (continuation)
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
