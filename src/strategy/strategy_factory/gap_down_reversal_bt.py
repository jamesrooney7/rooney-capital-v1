#!/usr/bin/env python3
"""
Gap Down Reversal Strategy - Backtrader Implementation

Entry: Gap down (open < prev close) AND close > open (bullish reversal)
Exit: Price > prev high (strength confirmed)

Strategy ID: 42
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class GapDownReversalBT(IbsStrategy):
    """Gap Down Reversal with ML Meta-Labeling."""

    params = (
        ('min_gap_pct', 0.5),  # Minimum gap size in %

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 15),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

    def entry_conditions_met(self):
        if len(self.data) < 5:
            return False

        # Gap down: open < prev close
        prev_close = self.data.close[-1]
        curr_open = self.data.open[0]
        curr_close = self.data.close[0]

        gap_pct = ((prev_close - curr_open) / prev_close) * 100

        # Entry: gap down + bullish close
        entry = (gap_pct >= self.params.min_gap_pct and
                curr_close > curr_open)

        return entry

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when price > previous bar's high
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
