#!/usr/bin/env python3
"""Aroon Indicator - Backtrader Implementation
Entry: Aroon Up crosses above Aroon Down
Exit: Aroon Down crosses above or Aroon Up weakens
Strategy ID: 33"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class AroonIndicatorBT(IbsStrategy):
    params = (
        ('aroon_period', 25), ('aroon_threshold', 70),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.aroon = bt.indicators.AroonIndicator(self.data, period=self.params.aroon_period)
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.aroon_period + 10:
            return False
        
        strong_up = self.aroon.aroonup[0] >= self.params.aroon_threshold
        
        if len(self.data) > 1:
            up_cross = (self.aroon.aroonup[0] > self.aroon.aroondown[0] and 
                       self.aroon.aroonup[-1] <= self.aroon.aroondown[-1])
        else:
            up_cross = False
        
        return up_cross and strong_up
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.aroon.aroondown[0] > self.aroon.aroonup[0]:
            return True
        if self.aroon.aroonup[0] < self.params.aroon_threshold:
            return True
        return False
