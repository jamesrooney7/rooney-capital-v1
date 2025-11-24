#!/usr/bin/env python3
"""
ATR Buy the Dip Strategy - Backtrader Implementation

Entry: Close < smoothed lower trigger (based on ATR)
Exit: Close > high[1] (reversal confirmation)

Lower trigger = SMA(close - ATR×multiplier, smoothing_period)

Strategy ID: 52
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ATRBuyDipBT(IbsStrategy):
    """ATR Buy the Dip with ML Meta-Labeling."""

    params = (
        ('atr_period', 20),              # ATR period
        ('atr_multiplier', 1.0),         # ATR multiplier for threshold
        ('smoothing_period', 10),        # Smoothing period for trigger
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

        # ATR
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.params.atr_period
        )

        # EMA for optional filter
        self.ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_period
        )

        # Raw lower trigger: close - ATR×multiplier
        # This will be smoothed in entry_conditions_met
        self.raw_trigger_values = []

    def entry_conditions_met(self):
        if len(self.data) < max(self.params.ema_period, self.params.atr_period) + self.params.smoothing_period + 5:
            return False

        # Calculate raw trigger: close - ATR×multiplier
        raw_trigger = self.data.close[0] - (self.atr[0] * self.params.atr_multiplier)

        # Store for smoothing calculation
        self.raw_trigger_values.append(raw_trigger)

        # Keep only smoothing_period values
        if len(self.raw_trigger_values) > self.params.smoothing_period:
            self.raw_trigger_values.pop(0)

        # Need enough values for smoothing
        if len(self.raw_trigger_values) < self.params.smoothing_period:
            return False

        # Smoothed lower trigger (SMA of raw trigger)
        lower_trigger = sum(self.raw_trigger_values) / len(self.raw_trigger_values)

        # Entry: Close < lower trigger (oversold)
        entry = self.data.close[0] < lower_trigger

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
