#!/usr/bin/env python3
"""
Moving Average Crossover Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/ma_cross.py)

Classic trend following:
- Entry: Fast MA crosses above slow MA
- Exit: Fast MA crosses below slow MA

Strategy ID: 17
Archetype: Trend Following
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class MACrossBT(IbsStrategy):
    """Moving Average Crossover with ML Meta-Labeling."""

    params = (
        # Strategy-specific
        ('fast_period', 10),   # Fast MA period
        ('slow_period', 30),   # Slow MA period

        # Risk management
        ('stop_loss_atr', 1.5),
        ('take_profit_atr', 3.0),
        ('max_bars_held', 40),  # Trend followers hold longer

        # ML meta-labeling
        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        """Initialize indicators."""
        super().__init__()

        # Fast and slow moving averages
        self.fast_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.fast_period
        )

        self.slow_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.slow_period
        )

        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(
            self.fast_ma,
            self.slow_ma
        )

        logger.info(
            f"MACrossBT initialized: "
            f"fast={self.params.fast_period}, "
            f"slow={self.params.slow_period}"
        )

    def entry_conditions_met(self):
        """Entry: Fast MA crosses above slow MA."""
        if len(self.data) < self.params.slow_period + 5:
            return False

        # Crossover indicator: 1 = bullish cross, -1 = bearish cross, 0 = no cross
        entry = self.crossover[0] > 0

        if entry and self.params.verbose:
            logger.debug(
                f"Entry: FastMA={self.fast_ma[0]:.2f} crossed above "
                f"SlowMA={self.slow_ma[0]:.2f}"
            )

        return entry

    def exit_conditions_met(self):
        """Exit: Fast MA crosses below slow MA."""
        if not self.position:
            return False

        # Bearish crossover
        exit_signal = self.crossover[0] < 0

        if exit_signal and self.params.verbose:
            logger.debug(
                f"Exit: FastMA={self.fast_ma[0]:.2f} crossed below "
                f"SlowMA={self.slow_ma[0]:.2f}"
            )

        return exit_signal
