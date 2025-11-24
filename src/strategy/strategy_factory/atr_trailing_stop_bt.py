#!/usr/bin/env python3
"""ATR Trailing Stop - Backtrader Implementation
Entry: Breakout above N-period high
Exit: ATR trailing stop hit
Strategy ID: 28"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class ATRTrailingStopBT(IbsStrategy):
    params = (
        ('atr_period', 14), ('atr_mult', 3.0), ('atr_entry_length', 20),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.entry_high = bt.indicators.Highest(self.data.high, period=self.params.atr_entry_length)
        self.ma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.highest_since_entry = None
    
    def entry_conditions_met(self):
        if len(self.data) < max(self.params.atr_period, self.params.atr_entry_length, 50) + 10:
            return False
        
        breakout = self.data.close[0] > self.entry_high[-1]
        above_ma = self.data.close[0] > self.ma_50[0]
        
        if len(self.data) > 2:
            was_below = self.data.close[-1] <= self.entry_high[-2]
        else:
            was_below = False
        
        if breakout and above_ma and was_below:
            self.highest_since_entry = self.data.high[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        
        # Update highest high
        if self.data.high[0] > self.highest_since_entry:
            self.highest_since_entry = self.data.high[0]
        
        # Trailing stop
        stop_level = self.highest_since_entry - (self.atr[0] * self.params.atr_mult)
        return self.data.close[0] < stop_level
