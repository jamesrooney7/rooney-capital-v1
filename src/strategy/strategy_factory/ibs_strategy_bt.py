#!/usr/bin/env python3
"""
IBS (Internal Bar Strength) Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/ibs_strategy.py)

Classic mean reversion based on IBS indicator:
- IBS = (Close - Low) / (High - Low)
- Entry: IBS < threshold (oversold)
- Exit: IBS > threshold (overbought)

Strategy ID: 45
Archetype: Mean Reversion
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class IBSStrategyBT(IbsStrategy):
    """IBS Mean Reversion with ML Meta-Labeling."""

    params = (
        # Strategy-specific
        ('ibs_entry_threshold', 0.2),   # IBS < 0.2 = oversold
        ('ibs_exit_threshold', 0.8),    # IBS > 0.8 = overbought

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

        # IBS indicator: (Close - Low) / (High - Low)
        # Note: IbsStrategy already calculates this, but we expose it explicitly
        range_hl = self.data.high - self.data.low
        self.ibs = (self.data.close - self.data.low) / range_hl

        logger.info(
            f"IBSStrategyBT initialized: "
            f"entry_threshold={self.params.ibs_entry_threshold}, "
            f"exit_threshold={self.params.ibs_exit_threshold}"
        )

    def entry_conditions_met(self):
        """Entry: IBS < entry threshold (oversold)."""
        if len(self.data) < 5:
            return False

        entry = self.ibs[0] < self.params.ibs_entry_threshold

        if entry and self.params.verbose:
            logger.debug(
                f"Entry: IBS={self.ibs[0]:.3f} < "
                f"threshold={self.params.ibs_entry_threshold}"
            )

        return entry

    def exit_conditions_met(self):
        """Exit: IBS > exit threshold (overbought)."""
        if not self.position:
            return False

        exit_signal = self.ibs[0] > self.params.ibs_exit_threshold

        if exit_signal and self.params.verbose:
            logger.debug(
                f"Exit: IBS={self.ibs[0]:.3f} > "
                f"threshold={self.params.ibs_exit_threshold}"
            )

        return exit_signal
