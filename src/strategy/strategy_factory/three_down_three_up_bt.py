#!/usr/bin/env python3
"""
Three Down Three Up Strategy - Backtrader Implementation

Entry: N consecutive down closes (close < previous close)
Exit: N consecutive up closes (close > previous close)

Strategy ID: 50
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ThreeDownThreeUpBT(IbsStrategy):
    """Three Down Three Up with ML Meta-Labeling."""

    params = (
        ('consecutive_down_entry', 3),   # Consecutive down closes for entry
        ('consecutive_up_exit', 3),      # Consecutive up closes for exit
        ('ema_period', 200),             # EMA period for filter
        ('use_ema_filter', False),       # Enable EMA filter

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # EMA for optional filter
        self.ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_period
        )

        self.consecutive_down = 0
        self.consecutive_up = 0

    def entry_conditions_met(self):
        if len(self.data) < max(self.params.ema_period, 5) + 5:
            return False

        # Check if current close < previous close (down)
        if self.data.close[0] < self.data.close[-1]:
            self.consecutive_down += 1
            self.consecutive_up = 0
        # Check if current close > previous close (up)
        elif self.data.close[0] > self.data.close[-1]:
            self.consecutive_up += 1
            self.consecutive_down = 0
        else:
            # Equal closes - reset both
            self.consecutive_down = 0
            self.consecutive_up = 0

        # Entry: consecutive down >= threshold
        entry = self.consecutive_down >= self.params.consecutive_down_entry

        # Optional EMA filter: only enter if price above EMA
        if self.params.use_ema_filter:
            entry = entry and (self.data.close[0] > self.ema[0])

        return entry

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Track consecutive up closes while in position
        # Note: consecutive_up is already tracked in entry_conditions_met

        # Exit after N consecutive up closes
        return self.consecutive_up >= self.params.consecutive_up_exit
