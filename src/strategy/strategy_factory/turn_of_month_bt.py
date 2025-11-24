#!/usr/bin/env python3
"""
Turn of Month Strategy - Backtrader Implementation

Entry: Last N days of month or first M days of month
Exit: After holding period

Strategy ID: 43
Archetype: Calendar/Seasonal
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TurnOfMonthBT(IbsStrategy):
    """Turn of Month with ML Meta-Labeling."""

    params = (
        ('last_days_of_month', 2),   # Last 2 days of month
        ('first_days_of_month', 3),  # First 3 days of month
        ('hold_days', 5),            # Holding period

        ('stop_loss_atr', 1.5),
        ('take_profit_atr', 3.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        self.entry_bar = None

    def entry_conditions_met(self):
        if len(self.data) < 30:  # Need data to identify month boundaries
            return False

        # Get current date
        dt = self.data.datetime.datetime(0)
        day = dt.day

        # Check if last N days of month
        # Simplified: assume ~30 days/month, adjust for actual implementation
        import calendar
        last_day = calendar.monthrange(dt.year, dt.month)[1]

        is_end_of_month = (last_day - day) < self.params.last_days_of_month
        is_start_of_month = day <= self.params.first_days_of_month

        entry = is_end_of_month or is_start_of_month

        if entry:
            self.entry_bar = len(self)

        return entry

    def exit_conditions_met(self):
        if not self.position or self.entry_bar is None:
            return False

        # Exit after hold_days
        bars_held = len(self) - self.entry_bar
        return bars_held >= self.params.hold_days
