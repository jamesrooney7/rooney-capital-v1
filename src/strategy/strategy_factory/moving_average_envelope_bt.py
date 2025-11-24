#!/usr/bin/env python3
"""
Moving Average Envelope Strategy - Backtrader Implementation

Entry: Price closes below lower envelope (MA - X%)
Exit: Price returns to MA centerline

Strategy ID: 7
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class MovingAverageEnvelopeBT(IbsStrategy):
    """Moving Average Envelope with ML Meta-Labeling."""

    params = (
        ('mae_length', 50),             # MA period
        ('mae_envelope_pct', 2.0),      # Envelope percentage
        ('mae_type', 'sma'),            # 'sma' or 'ema'

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # Calculate middle band (MA)
        if self.params.mae_type == 'ema':
            self.mae_middle = bt.indicators.EMA(
                self.data.close,
                period=self.params.mae_length
            )
        else:  # SMA
            self.mae_middle = bt.indicators.SMA(
                self.data.close,
                period=self.params.mae_length
            )

        # Calculate envelopes
        envelope_pct = self.params.mae_envelope_pct / 100.0
        self.mae_upper = self.mae_middle * (1 + envelope_pct)
        self.mae_lower = self.mae_middle * (1 - envelope_pct)

    def entry_conditions_met(self):
        if len(self.data) < self.params.mae_length + 10:
            return False

        # Entry: Close below lower envelope
        return self.data.close[0] < self.mae_lower[0]

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when price returns to middle band
        return self.data.close[0] >= self.mae_middle[0]
