#!/usr/bin/env python3
"""Ichimoku Cloud - Backtrader Implementation
Entry: Price above cloud AND Tenkan crosses above Kijun
Exit: Price falls into cloud OR Tenkan < Kijun
Strategy ID: 10"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class IchimokuCloudBT(IbsStrategy):
    params = (
        ('ich_tenkan', 9), ('ich_kijun', 26),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.ichimoku = bt.indicators.Ichimoku(
            self.data,
            tenkan=self.params.ich_tenkan,
            kijun=self.params.ich_kijun
        )
        self.cloud_top = bt.Max(self.ichimoku.senkou_span_a, self.ichimoku.senkou_span_b)
    
    def entry_conditions_met(self):
        if len(self.data) < 60:
            return False
        above_cloud = self.data.close[0] > self.cloud_top[0]
        if len(self.data) > 1:
            tk_cross = (self.ichimoku.tenkan_sen[0] > self.ichimoku.kijun_sen[0]) and                       (self.ichimoku.tenkan_sen[-1] <= self.ichimoku.kijun_sen[-1])
        else:
            tk_cross = False
        return above_cloud and tk_cross
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.data.close[0] <= self.cloud_top[0]:
            return True
        if self.ichimoku.tenkan_sen[0] < self.ichimoku.kijun_sen[0]:
            return True
        return False
