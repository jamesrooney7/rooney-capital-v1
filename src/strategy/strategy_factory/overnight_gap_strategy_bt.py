#!/usr/bin/env python3
"""Overnight Gap Strategy - Backtrader Implementation
Entry: Gap down exceeds threshold
Exit: Gap fills
Strategy ID: 38"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class OvernightGapStrategyBT(IbsStrategy):
    params = (
        ('gap_threshold', 0.005), ('gap_max', 0.020), ('gap_entry_time', 1),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.prior_close = None
        self.gap_pct = None
    
    def entry_conditions_met(self):
        # Simplified - would need session detection for real implementation
        return False
    
    def exit_conditions_met(self):
        return False
