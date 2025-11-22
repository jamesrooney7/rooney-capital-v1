#!/usr/bin/env python3
"""
Consecutive Close < Low[1] Strategy - Backtrader Implementation

Entry: N consecutive bars where close < low[1] (aggressive selling)
Exit: Close > high[1] (reversal confirmation)

Strategy ID: 51
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ConsecutiveCloseLowBT(IbsStrategy):
    """Consecutive Close < Low[1] with ML Meta-Labeling."""

    params = (
        ('threshold', 3),               # Consecutive bars threshold
        ('ema_period', 200),            # EMA period for filter
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

        # EMA for optional filter
        self.ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_period
        )

        self.consecutive_close_low = 0

    def entry_conditions_met(self):
        if len(self.data) < max(self.params.ema_period, 5) + 5:
            return False

        # Check if close < low[1] (aggressive down bar)
        if len(self.data) > 1:
            if self.data.close[0] < self.data.low[-1]:
                self.consecutive_close_low += 1
            else:
                self.consecutive_close_low = 0
        else:
            self.consecutive_close_low = 0

        # Entry: consecutive close < low[1] >= threshold
        entry = self.consecutive_close_low >= self.params.threshold

        # Optional EMA filter: only enter if price above EMA
        if self.params.use_ema_filter:
            entry = entry and (self.data.close[0] > self.ema[0])

        return entry

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when close > high[1]
        if len(self.data) > 1:
            return self.data.close[0] > self.data.high[-1]

        return False
