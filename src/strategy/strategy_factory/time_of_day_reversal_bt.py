#!/usr/bin/env python3
"""Time of Day Reversal - Backtrader Implementation
Entry: Oversold at specific time window
Exit: Target time or overbought
Strategy ID: 39"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class TimeOfDayReversalBT(IbsStrategy):
    params = (
        ('tod_entry_hour', 11), ('tod_rsi_threshold', 30), ('tod_exit_hour', 15),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
    
    def entry_conditions_met(self):
        # Simplified - would need datetime handling for hour detection
        # This is a placeholder
        if len(self.data) < 50:
            return False
        
        oversold = self.rsi[0] < self.params.tod_rsi_threshold
        
        if len(self.data) > 1:
            was_not_oversold = self.rsi[-1] >= self.params.tod_rsi_threshold
        else:
            was_not_oversold = False
        
        return oversold and was_not_oversold
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.rsi[0] >= 70
