#!/usr/bin/env python3
"""Money Flow Index - Backtrader Implementation
Entry: MFI crosses above oversold
Exit: MFI reaches overbought or reverses
Strategy ID: 34"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class MoneyFlowIndexBT(IbsStrategy):
    params = (
        ('mfi_period', 14), ('mfi_oversold', 20), ('mfi_overbought', 80),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        # Custom MFI calculation
        typical_price = (self.data.high + self.data.low + self.data.close) / 3
        money_flow = typical_price * self.data.volume
        
        # Simplified MFI - Backtrader has built-in support
        # For production, would implement full MFI calculation
        self.mfi_approx = bt.indicators.RSI(money_flow, period=self.params.mfi_period)
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.mfi_period + 10:
            return False
        
        mfi_above = self.mfi_approx[0] > self.params.mfi_oversold
        
        if len(self.data) > 1:
            mfi_was_below = self.mfi_approx[-1] <= self.params.mfi_oversold
        else:
            mfi_was_below = False
        
        return mfi_above and mfi_was_below
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.mfi_approx[0] >= self.params.mfi_overbought:
            return True
        if self.mfi_approx[0] < 50:
            return True
        return False
