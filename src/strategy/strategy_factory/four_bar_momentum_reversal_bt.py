#!/usr/bin/env python3
"""
Four Bar Momentum Reversal Strategy - Backtrader Implementation

Entry: N consecutive bars where close < close[N bars ago] (sustained bearish momentum)
Exit: Close > high[1] (reversal confirmation)

Strategy ID: 46
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class FourBarMomentumReversalBT(IbsStrategy):
    """Four Bar Momentum Reversal with ML Meta-Labeling."""

    params = (
        ('lookback', 4),          # N bars ago for reference close
        ('buy_threshold', 4),     # Consecutive bars threshold

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Track consecutive bars below reference
        self.consecutive_count = 0
        self.ref_close_values = []

    def entry_conditions_met(self):
        if len(self.data) < self.params.lookback + 5:
            return False

        # Reference close (N bars ago)
        ref_close = self.data.close[-self.params.lookback]
        curr_close = self.data.close[0]

        # Check if current close < reference close
        if curr_close < ref_close:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        # Entry when consecutive count >= threshold
        return self.consecutive_count >= self.params.buy_threshold

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1]
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
