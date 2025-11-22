#!/usr/bin/env python3
"""Gap Fill - Backtrader Implementation
Entry: Gap down > threshold
Exit: Gap partially filled
Strategy ID: 23"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class GapFillBT(IbsStrategy):
    params = (
        ('gap_threshold', 1.0), ('gap_fill_target', 0.5),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.gap_fill_price = None
    
    def entry_conditions_met(self):
        if len(self.data) < 2:
            return False
        
        prev_close = self.data.close[-1]
        gap_pct = ((self.data.open[0] - prev_close) / prev_close) * 100
        
        if gap_pct < -self.params.gap_threshold:
            gap_size = self.data.open[0] - prev_close
            self.gap_fill_price = prev_close + (self.params.gap_fill_target * gap_size)
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position or not self.gap_fill_price:
            return False
        return self.data.close[0] >= self.gap_fill_price
