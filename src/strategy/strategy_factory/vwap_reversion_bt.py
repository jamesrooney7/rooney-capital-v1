#!/usr/bin/env python3
"""VWAP Reversion - Backtrader Implementation
Entry: Price deviates N std from VWAP
Exit: Price returns to VWAP
Strategy ID: 24"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class VWAPReversionBT(IbsStrategy):
    params = (
        ('vwap_std_threshold', 2.0),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.vwap = bt.indicators.VWAP(self.data)
        self.vwap_diff = self.data.close - self.vwap
        self.vwap_std = bt.indicators.StdDev(self.vwap_diff, period=20)
    
    def entry_conditions_met(self):
        if len(self.data) < 20:
            return False
        lower_band = self.vwap[0] - (self.params.vwap_std_threshold * self.vwap_std[0])
        return self.data.close[0] < lower_band
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.data.close[0] >= self.vwap[0]
