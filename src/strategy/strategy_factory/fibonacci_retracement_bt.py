#!/usr/bin/env python3
"""Fibonacci Retracement - Backtrader Implementation
Entry: Price retraces to Fib level and bounces
Exit: Price reaches swing high OR breaks swing low
Strategy ID: 20"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class FibonacciRetracementBT(IbsStrategy):
    params = (
        ('fib_lookback', 30), ('fib_level', 0.500), ('fib_tolerance', 0.02),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.swing_high = bt.indicators.Highest(self.data.high, period=self.params.fib_lookback)
        self.swing_low = bt.indicators.Lowest(self.data.low, period=self.params.fib_lookback)
        self.avg_range = bt.indicators.SMA(self.swing_high - self.swing_low, period=50)
        self.entry_swing_high = None
        self.entry_swing_low = None
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.fib_lookback + 10:
            return False
        
        swing_range = self.swing_high[0] - self.swing_low[0]
        fib_price = self.swing_high[0] - self.params.fib_level * swing_range
        
        price_at_level = (self.data.low[0] <= fib_price * (1 + self.params.fib_tolerance) and
                         self.data.high[0] >= fib_price * (1 - self.params.fib_tolerance))
        valid_range = swing_range > self.avg_range[0] * 0.3
        bouncing = self.data.close[0] > fib_price
        
        if price_at_level and valid_range and bouncing:
            self.entry_swing_high = self.swing_high[0]
            self.entry_swing_low = self.swing_low[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.entry_swing_high and self.data.close[0] >= self.entry_swing_high * 0.98:
            return True
        if self.entry_swing_low and self.data.close[0] < self.entry_swing_low:
            return True
        return False
