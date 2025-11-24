#!/usr/bin/env python3
"""ADX Trend Strength - Backtrader Implementation
Entry: ADX > threshold AND +DI crosses above -DI
Exit: ADX < threshold OR +DI < -DI
Strategy ID: 9"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class ADXTrendStrengthBT(IbsStrategy):
    params = (
        ('adx_period', 14), ('adx_threshold', 25),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.adx = bt.indicators.AverageDirectionalMovementIndex(
            self.data, period=self.params.adx_period
        )
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.adx_period * 2 + 20:
            return False
        strong_trend = self.adx.adx[0] > self.params.adx_threshold
        if len(self.data) > 1:
            di_cross = (self.adx.plusDI[0] > self.adx.minusDI[0]) and (self.adx.plusDI[-1] <= self.adx.minusDI[-1])
        else:
            di_cross = False
        return strong_trend and di_cross
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.adx.adx[0] < self.params.adx_threshold:
            return True
        if self.adx.plusDI[0] < self.adx.minusDI[0]:
            return True
        return False
