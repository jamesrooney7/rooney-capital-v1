#!/usr/bin/env python3
"""Hammer/Shooting Star - Backtrader Implementation
Entry: Hammer pattern (long lower shadow, small body at top)
Exit: Close below hammer low (pattern failure)
Strategy ID: 6"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class HammerShootingStarBT(IbsStrategy):
    params = (
        ('hammer_shadow_ratio', 2.5), ('hammer_body_position', 0.7),
        ('hammer_require_trend', 2), ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0), ('max_bars_held', 20),
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
        lower_shadow = min(self.data.open[0], self.data.close[0]) - self.data.low[0]
        upper_shadow = self.data.high[0] - max(self.data.open[0], self.data.close[0])
        
        if range_hl <= 0 or body <= 0:
            return False
        
        shadow_ratio = lower_shadow / body if body > 0 else 0
        body_position = (min(self.data.open[0], self.data.close[0]) - self.data.low[0]) / range_hl if range_hl > 0 else 0
        
        # Track down streak
        if self.data.close[0] < self.data.close[-1]:
            self.down_streak += 1
        else:
            self.down_streak = 0
        
        long_shadow = shadow_ratio >= self.params.hammer_shadow_ratio
        body_at_top = body_position >= self.params.hammer_body_position
        small_upper = upper_shadow <= body
        min_range = range_hl > self.avg_range[0] * 0.5
        had_downtrend = self.down_streak >= self.params.hammer_require_trend if self.params.hammer_require_trend > 0 else True
        
        if long_shadow and body_at_top and small_upper and min_range and had_downtrend:
            self.entry_low = self.data.low[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position or not self.entry_low:
            return False
        return self.data.close[0] < self.entry_low
