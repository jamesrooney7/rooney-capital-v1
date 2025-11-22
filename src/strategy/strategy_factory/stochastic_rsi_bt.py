#!/usr/bin/env python3
"""
Stochastic RSI Strategy - Backtrader Implementation

Entry: StochRSI crosses above oversold level
Exit: StochRSI crosses below overbought level

Strategy ID: 11
Archetype: Momentum
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class StochasticRSIBT(IbsStrategy):
    """Stochastic RSI with ML Meta-Labeling."""

    params = (
        ('stochrsi_length', 14),        # RSI period
        ('stochrsi_stoch_length', 14),  # Stochastic period
        ('stochrsi_oversold', 20),      # Oversold level
        ('stochrsi_overbought', 80),    # Overbought level

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
            period=self.params.stochrsi_length
        )

        # Stochastic of RSI
        self.stochrsi = bt.indicators.Stochastic(
            self.rsi,
            period=self.params.stochrsi_stoch_length,
            period_dfast=1,  # No smoothing
            period_dslow=1
        )

    def entry_conditions_met(self):
        if len(self.data) < self.params.stochrsi_length + self.params.stochrsi_stoch_length + 20:
            return False

        # Entry: StochRSI crosses above oversold
        above_oversold = self.stochrsi[0] > self.params.stochrsi_oversold

        if len(self.data) > 1:
            was_below = self.stochrsi[-1] <= self.params.stochrsi_oversold
        else:
            was_below = False

        return above_oversold and was_below

    def exit_conditions_met(self):
        if not self.position:
            return False

        # Exit when StochRSI crosses below overbought
        if len(self.data) > 1:
            below_overbought = self.stochrsi[0] < self.params.stochrsi_overbought
            was_above = self.stochrsi[-1] >= self.params.stochrsi_overbought
            return below_overbought and was_above

        return False
