#!/usr/bin/env python3
"""
Average High-Low Range + IBS Reversal Strategy - Backtrader Implementation

Entry: Close below (high - multiplier×avg_HL_range) for N bars AND IBS < threshold
Exit: Close > high[1] (reversal confirmation)

Strategy ID: 49
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class AvgHLRangeIBSBT(IbsStrategy):
    """Average HL Range + IBS with ML Meta-Labeling."""

    params = (
        ('length', 20),                    # Period for average HL range
        ('range_multiplier', 2.5),         # Multiplier for threshold
        ('bars_below_threshold', 2),       # Consecutive bars below threshold
        ('ibs_buy_threshold', 0.2),        # IBS threshold

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Average high-low range
        hl_range = self.data.high - self.data.low
        self.avg_hl_range = bt.indicators.SMA(hl_range, period=self.params.length)

        # IBS calculation
        range_hl = self.data.high - self.data.low
        self.ibs = (self.data.close - self.data.low) / range_hl

        self.consecutive_below = 0

    def entry_conditions_met(self):
        if len(self.data) < self.params.length + 5:
            return False

        # Lower threshold: high - multiplier × avg_HL_range
        lower_threshold = self.data.high[0] - (self.params.range_multiplier * self.avg_hl_range[0])

        # Check if close below threshold
        if self.data.close[0] < lower_threshold:
            self.consecutive_below += 1
        else:
            self.consecutive_below = 0

        # Entry conditions:
        # 1. Consecutive bars below threshold >= threshold
        # 2. IBS < threshold
        below_condition = self.consecutive_below >= self.params.bars_below_threshold
        ibs_condition = self.ibs[0] < self.params.ibs_buy_threshold

        return below_condition and ibs_condition

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1]
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
