#!/usr/bin/env python3
"""
RSI(2) + 200 SMA Filter Strategy - Backtrader Implementation

Larry Connors' classic strategy:
Entry: RSI(2) < 5 AND price > 200 SMA (trend filter)
Exit: RSI(2) > 70

Strategy ID: 36
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class RSI2SMAFilterBT(IbsStrategy):
    """RSI(2) + SMA Filter with ML Meta-Labeling."""

    params = (
        ('rsi_length', 2),              # RSI period
        ('rsi_oversold', 5),            # Oversold threshold
        ('rsi_overbought', 70),         # Overbought threshold
        ('sma_filter', 200),            # Trend filter SMA

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
            period=self.params.rsi_length
        )

        # SMA filter
        self.sma_filter = bt.indicators.SMA(
            self.data.close,
            period=self.params.sma_filter
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.sma_filter + 10:
            return False

        # Entry condition 1: RSI oversold
        rsi_condition = self.rsi[0] < self.params.rsi_oversold

        # Entry condition 2: Price above SMA (trend filter)
        trend_condition = self.data.close[0] > self.sma_filter[0]

        # Entry: Both conditions met
        return rsi_condition and trend_condition

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when RSI crosses above overbought
        return self.rsi[0] > self.params.rsi_overbought
