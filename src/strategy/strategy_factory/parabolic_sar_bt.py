#!/usr/bin/env python3
"""Parabolic SAR - Backtrader Implementation
Entry: SAR flips from above to below price
Exit: SAR flips back above price
Strategy ID: 8"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class ParabolicSARBT(IbsStrategy):
    params = (
        ('psar_af_start', 0.02), ('psar_af_max', 0.2),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.psar = bt.indicators.ParabolicSAR(
            self.data,
            af=self.params.psar_af_start,
            afmax=self.params.psar_af_max
        )
    
    def entry_conditions_met(self):
        if len(self.data) < 20:
            return False
        sar_now_below = self.psar[0] < self.data.close[0]
        if len(self.data) > 1:
            sar_was_above = self.psar[-1] >= self.data.close[-1]
        else:
            sar_was_above = False
        return sar_now_below and sar_was_above
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.psar[0] >= self.data.close[0]
