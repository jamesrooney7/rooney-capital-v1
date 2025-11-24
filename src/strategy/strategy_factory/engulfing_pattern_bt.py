#!/usr/bin/env python3
"""
Engulfing Pattern Strategy - Backtrader Implementation

Entry: Bullish engulfing pattern after downtrend
Exit: Pattern failure (close below engulfing bar low)

Strategy ID: 5
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy

class EngulfingPatternBT(IbsStrategy):
    """Engulfing Pattern with ML Meta-Labeling."""

    params = (
        ('eng_min_body_pct', 0.5),
        ('eng_require_trend', 2),
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),
        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()
        self.down_streak = 0
        self.entry_low = None

    def entry_conditions_met(self):
        if len(self.data) < 5:
            return False
        
        # Track down streak
        if self.data.close[0] < self.data.close[-1]:
            self.down_streak += 1
        else:
            prev_down = self.down_streak
            self.down_streak = 0
            
            # Bullish engulfing check
            curr_body = abs(self.data.close[0] - self.data.open[0])
            curr_range = self.data.high[0] - self.data.low[0]
            prev_body = abs(self.data.close[-1] - self.data.open[-1])
            
            curr_up = self.data.close[0] > self.data.open[0]
            prev_down = self.data.close[-1] < self.data.open[-1]
            
            if curr_up and prev_down:
                # Engulfing condition
                engulfing = (self.data.open[0] <= self.data.close[-1] and 
                           self.data.close[0] >= self.data.open[-1])
                
                # Strong body
                body_pct = curr_body / curr_range if curr_range > 0 else 0
                strong_body = body_pct >= self.params.eng_min_body_pct
                
                # Trend requirement
                had_trend = prev_down >= self.params.eng_require_trend if self.params.eng_require_trend > 0 else True
                
                if engulfing and strong_body and had_trend:
                    self.entry_low = self.data.low[0]
                    return True
        
        return False

    def exit_conditions_met(self):
        if not self.position:
            return False
        if self.entry_low and self.data.close[0] < self.entry_low:
            return True
        return False
