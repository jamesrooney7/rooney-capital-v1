#!/usr/bin/env python3
"""Volume Breakout - Backtrader Implementation
Entry: Price breakout with volume surge
Exit: Volume fades or price fails
Strategy ID: 29"""
import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class VolumeBreakoutBT(IbsStrategy):
    params = (
        ('vb_lookback', 20), ('vb_vol_mult', 2.0), ('vb_vol_period', 30),
        ('stop_loss_atr', 1.0), ('take_profit_atr', 2.0), ('max_bars_held', 20),
        ('enable_ml_filter', True), ('ml_model_path', None), ('ml_threshold', 0.60),
    )
    
    def __init__(self):
        super().__init__()
        self.breakout_high = bt.indicators.Highest(self.data.high, period=self.params.vb_lookback)
        self.avg_volume = bt.indicators.SMA(self.data.volume, period=self.params.vb_vol_period)
        self.vol_ratio = self.data.volume / self.avg_volume
        self.entry_low = None
    
    def entry_conditions_met(self):
        if len(self.data) < max(self.params.vb_lookback, self.params.vb_vol_period) + 10:
            return False
        
        price_breakout = self.data.close[0] > self.breakout_high[-1]
        volume_surge = self.vol_ratio[0] >= self.params.vb_vol_mult
        
        if len(self.data) > 2:
            was_below = self.data.close[-1] <= self.breakout_high[-2]
        else:
            was_below = False
        
        if price_breakout and volume_surge and was_below:
            self.entry_low = self.data.low[0]
            return True
        return False
    
    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.entry_low and self.data.close[0] < self.entry_low:
            return True
        if self.vol_ratio[0] < 0.8:
            return True
        return False
