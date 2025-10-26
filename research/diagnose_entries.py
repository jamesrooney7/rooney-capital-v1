"""Diagnostic script to check IBS entry logic and trade generation.

Usage: python research/diagnose_entries.py --symbol ES --start 2010-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from pathlib import Path

import backtrader as bt
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research.utils.data_loader import setup_cerebro_with_data
from strategy.ibs_strategy import IbsStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DiagnosticStrategy(IbsStrategy):
    """Strategy that counts potential vs actual entries."""

    def __init__(self):
        super().__init__()
        self.total_bars = 0
        self.ibs_in_range_count = 0
        self.session_pass_count = 0
        self.entry_allowed_count = 0
        self.ml_passed_count = 0
        self.actual_entries = 0

    def next(self):
        self.total_bars += 1

        # Get IBS value
        ibs_val = None
        if len(self.hourly) > 1:
            high = self.hourly.high[0]
            low = self.hourly.low[0]
            close = self.hourly.close[0]
            if high > low:
                ibs_val = (close - low) / (high - low)

        # Check if in position
        in_position = bool(self.getposition(self.hourly))

        if not in_position and ibs_val is not None:
            # Count IBS in range
            if self.p.ibs_entry_low <= ibs_val <= self.p.ibs_entry_high:
                self.ibs_in_range_count += 1

                # Check session time
                dt = bt.num2date(self.hourly.datetime[0])
                if self.in_session(dt):
                    self.session_pass_count += 1

                    # Check entry_allowed
                    if self.entry_allowed(dt, ibs_val):
                        self.entry_allowed_count += 1

                        # Check ML filter
                        snapshot = self.collect_filter_values(intraday_ago=0)
                        ml_result = self._with_ml_score(snapshot)
                        if ml_result.get("ml_passed", False):
                            self.ml_passed_count += 1

        # Call parent to actually place orders
        super().next()

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self.actual_entries += 1
        super().notify_order(order)

    def stop(self):
        """Print diagnostic summary."""
        logger.info("\n" + "=" * 60)
        logger.info(f"DIAGNOSTIC SUMMARY FOR {self.p.symbol}")
        logger.info("=" * 60)
        logger.info(f"Total bars processed: {self.total_bars}")
        logger.info(f"IBS in range (0.0-0.2): {self.ibs_in_range_count} ({self.ibs_in_range_count/self.total_bars*100:.1f}%)")
        logger.info(f"Session time passed: {self.session_pass_count} ({self.session_pass_count/self.ibs_in_range_count*100:.1f}% of IBS range)")
        logger.info(f"entry_allowed() passed: {self.entry_allowed_count} ({self.entry_allowed_count/self.session_pass_count*100:.1f}% of session)")
        logger.info(f"ML filter passed: {self.ml_passed_count} ({self.ml_passed_count/self.entry_allowed_count*100:.1f}% of entry_allowed)")
        logger.info(f"Actual entries: {self.actual_entries}")
        logger.info("=" * 60)

        # Check for issues
        if self.ibs_in_range_count == 0:
            logger.error("❌ No bars with IBS in range 0.0-0.2! Check data quality or IBS calculation.")
        elif self.session_pass_count < self.ibs_in_range_count * 0.5:
            logger.warning("⚠️  Session time filter blocking >50% of IBS signals!")
        elif self.entry_allowed_count < self.session_pass_count * 0.5:
            logger.warning("⚠️  entry_allowed() blocking >50% of session-valid signals!")
        elif self.ml_passed_count < self.entry_allowed_count:
            logger.warning("⚠️  ML filter blocking some entries!")
        elif self.actual_entries < self.ml_passed_count:
            logger.warning("⚠️  Some ML-passed signals not converting to actual entries!")

        super().stop()


def main():
    parser = argparse.ArgumentParser(description="Diagnose IBS entry logic")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to diagnose")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="data/resampled", help="Data directory")

    args = parser.parse_args()

    logger.info(f"Diagnosing {args.symbol} from {args.start} to {args.end}")

    # Create Cerebro
    cerebro = bt.Cerebro(runonce=False)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=1.00)

    # Load data
    symbols_to_load = {args.symbol, 'TLT', 'NQ', '6A', '6B', '6C', '6S', 'CL', 'SI'}
    logger.info(f"Loading data for: {', '.join(sorted(symbols_to_load))}")

    setup_cerebro_with_data(
        cerebro,
        symbols=list(symbols_to_load),
        data_dir=args.data_dir,
        start_date=args.start,
        end_date=args.end
    )

    # Add diagnostic strategy
    cerebro.addstrategy(DiagnosticStrategy, symbol=args.symbol)

    # Run
    logger.info("Running diagnostic backtest...")
    results = cerebro.run()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
