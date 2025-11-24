#!/usr/bin/env python3
"""
IBS Extreme Strategy - Backtrader Implementation

Entry: IBS < extreme_low (e.g., 0.1)
Exit: IBS > exit_threshold or time/ATR stops

Strategy ID: 53
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class IBSExtremeBT(IbsStrategy):
    """IBS Extreme with ML Meta-Labeling."""

    params = (
        ('ibs_entry_threshold', 0.1),   # Very oversold
        ('ibs_exit_threshold', 0.5),    # Middle of range

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        range_hl = self.data.high - self.data.low
        self.ibs = (self.data.close - self.data.low) / range_hl

    def entry_conditions_met(self):
        if len(self.data) < 5:
            return False
        return self.ibs[0] < self.params.ibs_entry_threshold

    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.ibs[0] > self.params.ibs_exit_threshold
