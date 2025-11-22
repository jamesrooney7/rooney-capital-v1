#!/usr/bin/env python3
"""Doji Reversal - Backtrader Implementation
Entry: Doji pattern (small body, long shadows) after downtrend
Exit: Close below doji low
Strategy ID: 14"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class DojiReversalBT(IbsStrategy):
    params = (
        ('doji_body_pct', 0.10), ('doji_require_trend', 2), ('doji_shadow_min', 0.5),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.avg_range = bt.indicators.SMA(self.data.high - self.data.low, period=20)
        self.down_streak = 0
        self.entry_low = None
    
    def entry_conditions_met(self):
        if len(self.data) < 20:
            return False
        
        body = abs(self.data.close[0] - self.data.open[0])
        range_hl = self.data.high[0] - self.data.low[0]
        total_shadow = range_hl - body
        
        if range_hl <= 0:
            return False
        
        body_pct = body / range_hl
        shadow_pct = total_shadow / range_hl
        
        if self.data.close[0] < self.data.close[-1]:
            self.down_streak += 1
        else:
            self.down_streak = 0
        
        small_body = body_pct <= self.params.doji_body_pct
        long_shadows = shadow_pct >= self.params.doji_shadow_min
        min_range = range_hl > self.avg_range[0] * 0.3
        had_downtrend = self.down_streak >= self.params.doji_require_trend if self.params.doji_require_trend > 0 else True
        
        if small_body and long_shadows and min_range and had_downtrend:
            self.entry_low = self.data.low[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position or not self.entry_low:
            return False
        return self.data.close[0] < self.entry_low
