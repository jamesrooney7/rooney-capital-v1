#!/usr/bin/env python3
"""
Three Bar Reversal Strategy - Backtrader Implementation

Entry: 3 consecutive down bars + reversal bar closes up
Exit: N consecutive up bars (momentum exhausted)

Strategy ID: 3
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ThreeBarReversalBT(IbsStrategy):
    """Three Bar Reversal with ML Meta-Labeling."""

    params = (
        ('tbr_reversal_pct', 0.2),      # % from low for reversal
        ('tbr_exit_bars', 3),           # Consecutive up bars to exit

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        self.down_streak = 0
        self.up_streak = 0

    def entry_conditions_met(self):
        if len(self.data) < 5:
            return False

        # Track streaks
        if self.data.close[0] < self.data.close[-1]:
            self.down_streak += 1
            self.up_streak = 0
        elif self.data.close[0] > self.data.close[-1]:
            self.up_streak += 1
            prev_down = self.down_streak
            self.down_streak = 0
            
            # Entry: had 3+ down bars, current bar is up
            if prev_down >= 3:
                # Reversal strength check
                reversal_strength = ((self.data.close[0] - self.data.low[0]) / self.data.low[0]) * 100
                return reversal_strength >= self.params.tbr_reversal_pct
        else:
            self.down_streak = 0
            self.up_streak = 0

        return False

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit after N consecutive up bars
        return self.up_streak >= self.params.tbr_exit_bars
