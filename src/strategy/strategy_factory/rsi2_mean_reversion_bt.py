#!/usr/bin/env python3
"""
RSI2 Mean Reversion Strategy - Backtrader Implementation

Ported from Strategy Factory (research/strategy_factory/strategies/rsi2_mean_reversion.py)
for ML meta-labeling integration.

Classic Larry Connors mean reversion strategy:
- Entry: RSI(2) < oversold threshold
- Exit: RSI(2) > overbought threshold

Inherits from IbsStrategy to get:
- collect_filter_values() for 50+ ML features
- ML meta-labeling filter
- ATR-based stops and targets
- Time-based stops
"""

import backtrader as bt
from strategy.ibs_strategy import IbsStrategy
import logging

logger = logging.getLogger(__name__)


class RSI2MeanReversionBT(IbsStrategy):
    """
    RSI(2) Mean Reversion with ML Meta-Labeling.

    Strategy ID: 21 (from Strategy Factory)
    Archetype: Mean Reversion

    Phase 1 Optimized Parameters (example - will be per-instrument):
    - rsi_length: 2, 3, or 4
    - rsi_oversold: 5, 10, or 15
    - rsi_overbought: 60, 65, 70, or 75
    - stop_loss_atr: 0.5, 1.0, 1.5, or 2.0
    - take_profit_atr: 0.5, 1.0, 1.5, 2.0, or 3.0
    """

    params = (
        # Strategy-specific parameters (will be set from Phase 1 winners)
        ('rsi_length', 2),
        ('rsi_oversold', 10),
        ('rsi_overbought', 65),

        # Risk management (from Phase 1 optimization)
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_bars_held', 20),

        # ML meta-labeling
        ('enable_ml_filter', True),
        ('ml_model_path', None),
        ('ml_threshold', 0.60),  # Only take trades with >60% ML confidence
    )

    def __init__(self):
        """Initialize indicators."""
        super().__init__()

        # RSI indicator
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_length
        )

        logger.info(
            f"RSI2MeanReversionBT initialized: "
            f"rsi_length={self.params.rsi_length}, "
            f"oversold={self.params.rsi_oversold}, "
            f"overbought={self.params.rsi_overbought}"
        )

    def entry_conditions_met(self):
        """
        Check if RSI-based entry conditions are met.

        Entry Rule: RSI < oversold threshold

        Returns:
            bool: True if should enter long position
        """
        # Need enough bars for RSI calculation
        if len(self.data) < self.params.rsi_length + 5:
            return False

        # Entry condition: RSI crosses below oversold
        entry = self.rsi[0] < self.params.rsi_oversold

        if entry and self.params.verbose:
            logger.debug(
                f"Entry signal: RSI={self.rsi[0]:.2f} < "
                f"oversold={self.params.rsi_oversold}"
            )

        return entry

    def exit_conditions_met(self):
        """
        Check if RSI-based exit conditions are met.

        Exit Rule: RSI > overbought threshold

        Returns:
            bool: True if should exit position
        """
        if not self.position:
            return False

        # Exit condition: RSI crosses above overbought
        exit_signal = self.rsi[0] > self.params.rsi_overbought

        if exit_signal and self.params.verbose:
            logger.debug(
                f"Exit signal: RSI={self.rsi[0]:.2f} > "
                f"overbought={self.params.rsi_overbought}"
            )

        return exit_signal


if __name__ == '__main__':
    """Quick test of RSI2 strategy."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    print("RSI2 Mean Reversion Strategy")
    print("=" * 50)
    print()
    print("Parameters:")
    print(f"  RSI Length: {RSI2MeanReversionBT.params.rsi_length}")
    print(f"  RSI Oversold: {RSI2MeanReversionBT.params.rsi_oversold}")
    print(f"  RSI Overbought: {RSI2MeanReversionBT.params.rsi_overbought}")
    print(f"  Stop Loss ATR: {RSI2MeanReversionBT.params.stop_loss_atr}")
    print(f"  Take Profit ATR: {RSI2MeanReversionBT.params.take_profit_atr}")
    print()
    print("Inherits from IbsStrategy:")
    print("  ✓ collect_filter_values() - 50+ ML features")
    print("  ✓ ML meta-labeling filter")
    print("  ✓ ATR-based stops and targets")
    print("  ✓ Time-based max hold")
    print("  ✓ End-of-day exits")
    print()
    print("Ready for:")
    print("  1. Training data extraction")
    print("  2. ML meta-labeling optimization")
    print("  3. Live trading deployment")
