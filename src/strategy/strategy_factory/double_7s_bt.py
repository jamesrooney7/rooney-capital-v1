#!/usr/bin/env python3
"""
Double 7s Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/double_7s.py)

Classic mean reversion:
- Entry: Close < 7-period low
- Exit: Close > 7-period high

Strategy ID: 37
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class Double7sBT(IbsStrategy):
    """Double 7s with ML Meta-Labeling."""

    params = (
        # Strategy-specific
        ('entry_period', 7),  # Period for entry low
        ('exit_period', 7),   # Period for exit high

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

        # Entry: N-period low
        self.entry_low = bt.indicators.Lowest(
            self.data.low,
            period=self.params.entry_period
        )

        # Exit: N-period high
        self.exit_high = bt.indicators.Highest(
            self.data.high,
            period=self.params.exit_period
        )

        logger.info(
            f"Double7sBT initialized: "
            f"entry_period={self.params.entry_period}, "
            f"exit_period={self.params.exit_period}"
        )

    def entry_conditions_met(self):
        """Entry: Close < N-period low."""
        if len(self.data) < max(self.params.entry_period, self.params.exit_period) + 5:
            return False

        entry = self.data.close[0] < self.entry_low[0]

        if entry and self.params.verbose:
            logger.debug(
                f"Entry: Close={self.data.close[0]:.2f} < "
                f"Low({self.params.entry_period})={self.entry_low[0]:.2f}"
            )

        return entry

    def exit_conditions_met(self):
        """Exit: Close > N-period high."""
        if not self.position:
            return False

        exit_signal = self.data.close[0] > self.exit_high[0]

        if exit_signal and self.params.verbose:
            logger.debug(
                f"Exit: Close={self.data.close[0]:.2f} > "
                f"High({self.params.exit_period})={self.exit_high[0]:.2f}"
            )

        return exit_signal
