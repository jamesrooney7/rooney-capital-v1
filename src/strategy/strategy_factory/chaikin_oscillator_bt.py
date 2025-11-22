#!/usr/bin/env python3
"""Chaikin Oscillator - Backtrader Implementation
Entry: Oscillator crosses above zero
Exit: Oscillator crosses below zero
Strategy ID: 35"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class ChaikinOscillatorBT(IbsStrategy):
    params = (
        ('co_fast', 3), ('co_slow', 10),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        # Accumulation/Distribution Line
        clv = ((self.data.close - self.data.low) - (self.data.high - self.data.close))
        range_hl = self.data.high - self.data.low
        clv_norm = clv / range_hl
        mf_volume = clv_norm * self.data.volume
        
        # Chaikin Oscillator = EMA(fast) - EMA(slow) of A/D
        ema_fast = bt.indicators.EMA(mf_volume, period=self.params.co_fast)
        ema_slow = bt.indicators.EMA(mf_volume, period=self.params.co_slow)
        self.chaikin_osc = ema_fast - ema_slow
    
    def entry_conditions_met(self):
        if len(self.data) < max(self.params.co_fast, self.params.co_slow) + 50:
            return False
        
        osc_above = self.chaikin_osc[0] > 0
        
        if len(self.data) > 1:
            osc_was_below = self.chaikin_osc[-1] <= 0
        else:
            osc_was_below = False
        
        return osc_above and osc_was_below
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.chaikin_osc[0] < 0
