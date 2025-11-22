#!/usr/bin/env python3
"""Mean Reversion Bands - Backtrader Implementation
Entry: Price below lower band
Exit: Price returns to mean
Strategy ID: 27"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class MeanReversionBandsBT(IbsStrategy):
    params = (
        ('mrb_length', 30), ('mrb_std', 2.0), ('mrb_exit_pct', 0.75),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.middle = bt.indicators.SMA(self.data.close, period=self.params.mrb_length)
        self.std = bt.indicators.StdDev(self.data.close, period=self.params.mrb_length)
        self.upper_band = self.middle + (self.params.mrb_std * self.std)
        self.lower_band = self.middle - (self.params.mrb_std * self.std)
        self.entry_middle = None
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.mrb_length + 10:
            return False
        
        below_band = self.data.close[0] < self.lower_band[0]
        if len(self.data) > 1:
            was_above = self.data.close[-1] >= self.lower_band[-1]
        else:
            was_above = False
        
        if below_band and was_above:
            self.entry_middle = self.middle[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position or not self.entry_middle:
            return False
        
        # Calculate target based on % return to mean
        entry_price = self.position.price
        entry_dist = self.entry_middle - entry_price
        target_price = entry_price + (entry_dist * self.params.mrb_exit_pct)
        
        return self.data.close[0] >= target_price
