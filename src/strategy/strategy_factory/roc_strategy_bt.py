#!/usr/bin/env python3
"""ROC Strategy - Backtrader Implementation
Entry: ROC crosses above threshold
Exit: ROC fades or trend breaks
Strategy ID: 32"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class ROCStrategyBT(IbsStrategy):
    params = (
        ('roc_period', 20), ('roc_threshold', 4), ('roc_ma_length', 100),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.roc = bt.indicators.ROC(self.data.close, period=self.params.roc_period)
        self.ma_trend = bt.indicators.SMA(self.data.close, period=self.params.roc_ma_length)
    
    def entry_conditions_met(self):
        if len(self.data) < max(self.params.roc_period, self.params.roc_ma_length) + 10:
            return False
        
        roc_above = self.roc[0] > self.params.roc_threshold
        above_ma = self.data.close[0] > self.ma_trend[0]
        
        if len(self.data) > 1:
            roc_was_below = self.roc[-1] <= self.params.roc_threshold
        else:
            roc_was_below = False
        
        return roc_above and roc_was_below and above_ma
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.roc[0] < self.params.roc_threshold:
            return True
        if self.data.close[0] < self.ma_trend[0]:
            return True
        return False
