#!/usr/bin/env python3
"""
Buy on 5 Bar Low Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/buy_on_5_bar_low.py)

Simple mean reversion:
- Entry: Close < lowest low of previous N bars
- Exit: Close > previous bar's high (showing strength)

Strategy ID: 40
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class BuyOn5BarLowBT(IbsStrategy):
    """Buy on 5 Bar Low with ML Meta-Labeling."""

    params = (
        # Strategy-specific
        ('lookback', 5),  # Number of bars to look back for low

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

        # Lowest low of previous N bars
        self.lowest_low = bt.indicators.Lowest(
            self.data.low(-1),  # Exclude current bar
            period=self.params.lookback
        )

        logger.info(
            f"BuyOn5BarLowBT initialized: lookback={self.params.lookback}"
        )

    def entry_conditions_met(self):
        """
        Entry: Close < lowest low of previous N bars.

        This signals an oversold reversal condition.
        """
        if len(self.data) < self.params.lookback + 5:
            return False

        entry = self.data.close[0] < self.lowest_low[0]

        if entry and self.params.verbose:
            logger.debug(
                f"Entry: Close={self.data.close[0]:.2f} < "
                f"LowestLow({self.params.lookback})={self.lowest_low[0]:.2f}"
            )

        return entry

    def exit_conditions_met(self):
        """
        Exit: Close > previous bar's high (showing price strength).
        """
        if not self.position:
            return False

        # Exit when close > previous bar's high
        if len(self.data) > 1:
            prev_high = self.data.high[-1]
            exit_signal = self.data.close[0] > prev_high

            if exit_signal and self.params.verbose:
                logger.debug(
                    f"Exit: Close={self.data.close[0]:.2f} > "
                    f"PrevHigh={prev_high:.2f}"
                )

            return exit_signal

        return False
