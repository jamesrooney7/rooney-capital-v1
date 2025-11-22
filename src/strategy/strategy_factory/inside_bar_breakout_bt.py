#!/usr/bin/env python3
"""Inside Bar Breakout - Backtrader Implementation
Entry: Breakout above inside bar high
Exit: Breakdown below inside bar low
Strategy ID: 26"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class InsideBarBreakoutBT(IbsStrategy):
    params = (
        ('ib_min_inside_bars', 1), ('ib_breakout_pct', 0.0),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.inside_bar_count = 0
        self.ib_high = None
        self.ib_low = None
    
    def entry_conditions_met(self):
        if len(self.data) < 10:
            return False
        
        # Inside bar: High < prev High AND Low > prev Low
        is_inside = (self.data.high[0] < self.data.high[-1] and 
                    self.data.low[0] > self.data.low[-1])
        
        if is_inside:
            self.inside_bar_count += 1
            self.ib_high = self.data.high[-1]
            self.ib_low = self.data.low[-1]
        else:
            prev_count = self.inside_bar_count
            self.inside_bar_count = 0
            
            # Entry: breakout after inside bars
            if prev_count >= self.params.ib_min_inside_bars and self.ib_high:
                breakout_level = self.ib_high * (1 + self.params.ib_breakout_pct / 100)
                if self.data.close[0] > breakout_level:
                    return True
        
        return False
    
    def exit_conditions_met(self):
        if not self.position or not self.ib_low:
            return False
        return self.data.close[0] < self.ib_low
