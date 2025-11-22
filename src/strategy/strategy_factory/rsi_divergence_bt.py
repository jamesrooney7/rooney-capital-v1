#!/usr/bin/env python3
"""
RSI Divergence Strategy - Backtrader Implementation

Entry: Bullish divergence (price lower low + RSI higher low)
Exit: RSI reaches overbought or divergence negated

Strategy ID: 22
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class RSIDivergenceBT(IbsStrategy):
    """RSI Divergence with ML Meta-Labeling."""

    params = (
        ('rsi_div_length', 14),         # RSI period
        ('rsi_div_lookback', 10),       # Divergence lookback
        ('rsi_div_overbought', 70),     # Exit level

        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        super().__init__()

        # RSI
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_div_length
        )

        # Lowest values for divergence detection
        self.price_low = bt.indicators.Lowest(
            self.data.low,
            period=self.params.rsi_div_lookback
        )

        self.rsi_low = bt.indicators.Lowest(
            self.rsi,
            period=self.params.rsi_div_lookback
        )

        self.entry_price = None
        self.entry_low = None

    def entry_conditions_met(self):
        if len(self.data) < self.params.rsi_div_length + self.params.rsi_div_lookback + 20:
            return False

        if len(self.data) < 2:
            return False

        # Bullish divergence: price lower low + RSI higher low
        price_lower_low = self.data.low[0] < self.price_low[-1]
        rsi_higher_low = self.rsi[0] > self.rsi_low[-1]

        if price_lower_low and rsi_higher_low:
            self.entry_low = self.data.low[0]
            return True

        return False

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit 1: RSI reaches overbought
        if self.rsi[0] >= self.params.rsi_div_overbought:
            return True

        # Exit 2: Divergence negated (new lower low)
        if self.entry_low is not None:
            if self.data.low[0] < self.entry_low * 0.995:  # 0.5% lower
                return True

        return False
