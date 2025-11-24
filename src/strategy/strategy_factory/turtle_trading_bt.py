#!/usr/bin/env python3
"""Turtle Trading - Backtrader Implementation
Entry: Price breaks above N-day high
Exit: Price breaks below N-day low
Strategy ID: 16"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class TurtleTradingBT(IbsStrategy):
    params = (
        ('turtle_entry_length', 20), ('turtle_exit_length', 10),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.entry_high = bt.indicators.Highest(
            self.data.high,
            period=self.params.turtle_entry_length
        )
        self.exit_low = bt.indicators.Lowest(
            self.data.low,
            period=self.params.turtle_exit_length
        )
    
    def entry_conditions_met(self):
        if len(self.data) < self.params.turtle_entry_length + 10:
            return False
        breakout = self.data.close[0] > self.entry_high[-1]
        if len(self.data) > 2:
            was_below = self.data.close[-1] <= self.entry_high[-2]
        else:
            was_below = False
        return breakout and was_below
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        return self.data.close[0] < self.exit_low[0]
