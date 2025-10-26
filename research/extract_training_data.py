#!/usr/bin/env python3
"""
Extract training data from IbsStrategy backtests.

This script runs production IbsStrategy and captures:
1. All features from collect_filter_values() at entry time
2. Trade outcomes (exit time, PnL, return, binary)

Output: transformed_features.csv in the format expected by rf_cpcv_random_then_bo.py

This ensures 100% parity: training features = production features (same code!)

Usage:
    python research/extract_training_data.py --symbol ES --start 2010-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import backtrader as bt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.utils.data_loader import setup_cerebro_with_data
from strategy.ibs_strategy import IbsStrategy
from models.loader import load_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureLoggingStrategy(IbsStrategy):
    """
    Wrapper around IbsStrategy that logs features and trade outcomes.

    IMPORTANT: This strategy disables ML filtering to capture ALL base IBS signals.
    We need both winning and losing trades to train the ML model!
    """

    def __init__(self, *args, **kwargs):
        # Initialize feature log storage
        self.feature_log = []
        self.trade_entry_features = {}  # {order_id: features_dict}

        super().__init__(*args, **kwargs)

    def _with_ml_score(self, snapshot: dict | None) -> dict:
        """
        Override to bypass ML filtering during training data extraction.

        We want to capture ALL base IBS trades (both winners and losers)
        so the ML model can learn which filter combinations predict success.

        Sets ml_passed = True regardless of ml_score.
        """
        result = dict(snapshot) if snapshot else {}
        result["ml_score"] = None  # No ML scoring during extraction
        result["ml_passed"] = True  # Always allow entries for training data
        return result

    def notify_order(self, order):
        """Capture features when orders are placed."""

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                # Entry order completed - capture features
                try:
                    features = self.collect_filter_values(intraday_ago=0)
                    entry_time = bt.num2date(self.hourly.datetime[0])
                    entry_price = order.executed.price

                    # Store features with order ref for matching with exit
                    order_id = id(order)
                    self.trade_entry_features[order_id] = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'features': features.copy(),
                    }

                    logger.debug(f"Logged entry features for order {order_id} at {entry_time}")

                except Exception as e:
                    logger.error(f"Error capturing entry features: {e}", exc_info=True)

            elif order.issell():
                # Exit order completed - calculate outcomes
                try:
                    exit_time = bt.num2date(self.hourly.datetime[0])
                    exit_price = order.executed.price

                    # Find matching entry (most recent entry for this position)
                    # In IbsStrategy, we only have one position at a time
                    if self.trade_entry_features:
                        # Get the most recent entry
                        entry_id = max(self.trade_entry_features.keys())
                        entry_data = self.trade_entry_features.pop(entry_id)

                        entry_time = entry_data['entry_time']
                        entry_price = entry_data['entry_price']
                        features = entry_data['features']

                        # Calculate outcomes
                        price_return = (exit_price - entry_price) / entry_price

                        # Get contract specs for PnL calculation
                        from strategy.contract_specs import point_value
                        pv = point_value(self.p.symbol)
                        pnl_usd = (exit_price - entry_price) * pv

                        # Binary outcome
                        binary = 1 if pnl_usd > 0 else 0

                        # Create training record
                        record = {
                            'Date/Time': entry_time,
                            'Exit Date/Time': exit_time,
                            'Entry_Price': entry_price,
                            'Exit_Price': exit_price,
                            'y_return': price_return,
                            'y_binary': binary,
                            'y_pnl_usd': pnl_usd,
                        }

                        # Add all features
                        record.update(features)

                        self.feature_log.append(record)

                        logger.info(
                            f"Logged trade: {entry_time} → {exit_time} | "
                            f"PnL=${pnl_usd:.2f} | Return={price_return*100:.2f}% | "
                            f"Features={len(features)}"
                        )

                except Exception as e:
                    logger.error(f"Error logging exit: {e}", exc_info=True)

        # Call parent notify_order for normal processing
        super().notify_order(order)


def extract_training_data(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'data/resampled',
    output_path: str = None,
    use_ml: bool = False,  # Don't load ML model during extraction
):
    """
    Extract training data by running IbsStrategy backtest.

    Args:
        symbol: Symbol to extract (e.g., 'ES', 'NQ')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing resampled data
        output_path: Output CSV path (default: data/training/{symbol}_transformed_features.csv)
        use_ml: Whether to load ML model (default: False for training data extraction)

    Returns:
        DataFrame with training data
    """
    logger.info(f"Extracting training data for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Create Cerebro instance
    cerebro = bt.Cerebro(runonce=False)

    # Set initial cash
    cerebro.broker.setcash(100000.0)

    # Set commission
    from config import COMMISSION_PER_SIDE
    cerebro.broker.setcommission(commission=COMMISSION_PER_SIDE)

    # Load data for symbol + reference symbols
    # For now, load common reference symbols
    # TODO: Auto-detect based on strategy requirements
    symbols_to_load = {symbol, 'TLT', 'NQ', '6A', '6B', '6C', '6S', 'CL', 'SI'}

    logger.info(f"Loading data for symbols: {', '.join(sorted(symbols_to_load))}")
    setup_cerebro_with_data(
        cerebro,
        symbols=list(symbols_to_load),
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date
    )

    # Add feature-logging strategy
    strat_params = {'symbol': symbol}
    logger.info(f"Adding FeatureLoggingStrategy with params: {strat_params}")
    cerebro.addstrategy(FeatureLoggingStrategy, **strat_params)

    # Run backtest
    logger.info("Running backtest to extract features...")
    results = cerebro.run()
    strat = results[0]

    # Get feature log
    feature_log = strat.feature_log
    logger.info(f"Extracted {len(feature_log)} trades with features")

    if not feature_log:
        logger.warning("No trades captured! Check strategy parameters.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(feature_log)

    # Ensure Date column (for compatibility with training script)
    df['Date'] = pd.to_datetime(df['Date/Time']).dt.date

    # Save to CSV
    if output_path is None:
        output_dir = Path('data/training')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_transformed_features.csv"

    df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to: {output_path}")
    logger.info(f"Shape: {df.shape} (rows={len(df)}, columns={len(df.columns)})")
    logger.info(f"Features extracted: {[c for c in df.columns if c not in ['Date/Time', 'Exit Date/Time', 'Entry_Price', 'Exit_Price', 'y_return', 'y_binary', 'y_pnl_usd', 'Date']]}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Extract training data from IbsStrategy backtests')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to extract (e.g., ES)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')
    parser.add_argument('--output', type=str, help='Output CSV path')

    args = parser.parse_args()

    try:
        df = extract_training_data(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            output_path=args.output,
        )

        if df.empty:
            logger.error("No training data extracted!")
            return 1

        logger.info("✅ Training data extraction complete!")
        return 0

    except Exception as e:
        logger.error(f"Error extracting training data: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
