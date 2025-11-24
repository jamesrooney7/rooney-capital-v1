#!/usr/bin/env python3
"""
Three Bar Low Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/three_bar_low.py)

Entry: Close makes new 3-bar low
Exit: Close > SMA(close, 5) - price strength

Strategy ID: 41
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class ThreeBarLowBT(IbsStrategy):
    """Three Bar Low with ML Meta-Labeling."""

    params = (
        # Strategy-specific
        ('lookback', 3),  # Bars to check for new low
        ('exit_sma_period', 5),  # SMA period for exit

        # Risk management
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        # ML meta-labeling
        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),
    )

    def __init__(self):
        """Initialize indicators."""
        super().__init__()

        # Lowest low indicator
        self.lowest_low = bt.indicators.Lowest(
            self.data.low,
            period=self.params.lookback
        )

        # SMA for exit
        self.sma = bt.indicators.SMA(
            self.data.close,
            period=self.params.exit_sma_period
        )

        logger.info(
            f"ThreeBarLowBT initialized: lookback={self.params.lookback}"
        )

    def entry_conditions_met(self):
        """Entry: Close makes new N-bar low."""
        if len(self.data) < max(self.params.lookback, self.params.exit_sma_period) + 5:
            return False

        # Entry: current close is the lowest of past N bars
        entry = self.data.close[0] <= self.lowest_low[0]

        if entry and self.params.verbose:
            logger.debug(
                f"Entry: Close={self.data.close[0]:.2f} = "
                f"{self.params.lookback}-bar low"
            )

        return entry

    def exit_conditions_met(self):
        """Exit: Close > SMA (price showing strength)."""
        if not self.position:
            return False

        exit_signal = self.data.close[0] > self.sma[0]

        if exit_signal and self.params.verbose:
            logger.debug(
                f"Exit: Close={self.data.close[0]:.2f} > "
                f"SMA({self.params.exit_sma_period})={self.sma[0]:.2f}"
            )

        return exit_signal
