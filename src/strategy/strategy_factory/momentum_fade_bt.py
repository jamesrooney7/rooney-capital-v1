#!/usr/bin/env python3
"""Momentum Fade - Backtrader Implementation
Entry: Extreme negative momentum (ROC)
Exit: Momentum normalizes or mean reached
Strategy ID: 30"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class MomentumFadeBT(IbsStrategy):
    params = (
        ('mf_period', 10), ('mf_threshold', 7), ('mf_ma_length', 50),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.roc = bt.indicators.ROC(self.data.close, period=self.params.mf_period)
        self.mean = bt.indicators.SMA(self.data.close, period=self.params.mf_ma_length)
    
    def entry_conditions_met(self):
        if len(self.data) < max(self.params.mf_period, self.params.mf_ma_length) + 10:
            return False
        
        extreme_down = self.roc[0] < -self.params.mf_threshold
        below_mean = self.data.close[0] < self.mean[0]
        
        if len(self.data) > 1:
            was_not_extreme = self.roc[-1] >= -self.params.mf_threshold
        else:
            was_not_extreme = False
        
        return extreme_down and below_mean and was_not_extreme
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.roc[0] > -2:
            return True
        if self.data.close[0] >= self.mean[0]:
            return True
        return False
