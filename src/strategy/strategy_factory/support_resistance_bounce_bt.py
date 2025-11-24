#!/usr/bin/env python3
"""
Support/Resistance Bounce Strategy - Backtrader Implementation

Entry: Price bounces off support level
Exit: Price reaches midpoint between support/resistance

Strategy ID: 4
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class SupportResistanceBounceBT(IbsStrategy):
    """Support/Resistance Bounce with ML Meta-Labeling."""

    params = (
        ('sr_lookback', 40),
        ('sr_touch_pct', 0.2),
        ('sr_bounce_pct', 0.2),
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),
        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        self.support = bt.indicators.Lowest(self.data.low, period=self.params.sr_lookback)
        self.resistance = bt.indicators.Highest(self.data.high, period=self.params.sr_lookback)
        self.sr_midpoint = (self.support + self.resistance) / 2
        self.touched_support = False

    def entry_conditions_met(self):
        if len(self.data) < self.params.sr_lookback + 10:
            return False
        
        touch_pct = self.params.sr_touch_pct / 100
        bounce_pct = self.params.sr_bounce_pct / 100
        
        touch_threshold = self.support[0] * (1 + touch_pct)
        bounce_level = self.support[0] * (1 + bounce_pct)
        
        if self.data.low[0] <= touch_threshold:
            self.touched_support = True
        
        if self.touched_support and self.data.close[0] > bounce_level:
            self.touched_support = False
            return True
        
        return False

    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.data.close[0] >= self.sr_midpoint[0]
