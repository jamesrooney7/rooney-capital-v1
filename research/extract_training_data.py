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

        # Populate comprehensive list of all possible filter parameter keys
        # This tells collect_filter_values() to calculate ALL features
        self.ml_feature_param_keys = self._get_all_filter_param_keys()

    def _get_all_filter_param_keys(self) -> set:
        """
        Return comprehensive set of all filter parameter keys.

        This ensures collect_filter_values() calculates ALL possible features,
        not just the ones explicitly enabled in the strategy config.
        """
        keys = {
            # Calendar filters
            'allowedDOW', 'allowedMon', 'enableDOM', 'domDay',
            'enableBegWeek', 'enableEvenOdd',

            # Price/return filters
            'enablePrevDayPct', 'prev_day_pct',
            'enablePrevBarPct', 'prev_bar_pct',

            # IBS filters
            'enableIBSEntry', 'enableIBSExit', 'ibs',
            'enableDailyIBS', 'daily_ibs',
            'enablePrevIBS', 'prev_ibs',
            'enablePrevIBSDaily', 'prev_daily_ibs',

            # Pair filters
            'enablePairIBS', 'pair_ibs',
            'enablePairZ', 'pair_z',

            # RSI filters
            'enableRSIEntry', 'rsi_entry',
            'enableRSIEntry2Len', 'rsi_entry2_len',
            'enableRSIEntry14Len', 'rsi_entry14_len',
            'enableRSIEntry2', 'rsi2_entry',
            'enableDailyRSI', 'daily_rsi',
            'enableDailyRSI2Len', 'daily_rsi2_len',
            'enableDailyRSI14Len', 'daily_rsi14_len',

            # Bollinger Bands
            'enableBBHigh', 'bb_high',
            'enableBBHighD', 'bb_high_d',

            # EMA filters
            'enableEMA8', 'ema8',
            'enableEMA20', 'ema20',

            # ATR filters
            'enableATRZ', 'atrz',
            'enableHourlyATRPercentile', 'hourly_atr_percentile',

            # Volume filters
            'enableVolZ', 'volz',

            # Momentum/distance filters
            'enableDistZ', 'distz',
            'enableMom3Z', 'mom3z',
            'enablePriceZ', 'pricez',

            # Daily ATR/volume
            'enableDATRZ', 'datrz',
            'enableDVolZ', 'dvolz',

            # Trend/ratio filters
            'enableTRATRRatio', 'tratr_ratio',

            # Supply zone
            'use_supply_zone', 'supply_zone',

            # N7 bar
            'enableN7Bar', 'n7_bar',

            # Inside bar
            'enableInsideBar', 'inside_bar',

            # Bear count
            'enableBearCount', 'bear_count',

            # Spiral ER
            'enableSER', 'ser',

            # TWRC
            'enableTWRC', 'twrc',

            # VIX regime
            'enableVIXReg', 'vix_reg',
        }

        # Add cross-asset z-score and return keys dynamically
        cross_symbols = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S', 'TLT']
        timeframes = ['Hour', 'Day']

        for symbol in cross_symbols:
            for tf in timeframes:
                # Z-score filters
                keys.add(f'enable{symbol}ZScore{tf}')
                keys.add(f'{symbol.lower()}_z_score_{tf.lower()}')

                # Daily return filters
                keys.add(f'{symbol.lower()}_daily_return')
                keys.add(f'{symbol.lower()}_hourly_return')

        return keys

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

    def entry_allowed(self, dt, ibs_val: float) -> bool:
        """
        Override to bypass ALL filters during training data extraction.

        We only check:
        1. Session time (trading hours)
        2. IBS entry range (0.0-0.2 for the base signal)

        All other filters (RSI, ATR, calendar, etc.) are DISABLED so they
        don't block entries. We still CALCULATE those filter values via
        collect_filter_values(), but we don't use them to filter trades.

        This ensures we capture both winning and losing base IBS trades
        so the ML model can learn which filter combinations predict success.
        """
        # Check session time only
        if not self.in_session(dt):
            return False

        # Check base IBS entry range only
        if not (self.p.ibs_entry_low <= ibs_val <= self.p.ibs_entry_high):
            return False

        # All other filters bypassed!
        return True

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

                        # Gross PnL (price movement only)
                        gross_pnl = (exit_price - entry_price) * pv

                        # Net PnL (after commissions and slippage)
                        # Commission: $1.00 per side = $2.00 total
                        # Slippage: Already included in executed prices from Backtrader
                        commission_total = 2.00  # $1.00 entry + $1.00 exit
                        pnl_usd = gross_pnl - commission_total

                        # Binary outcome (based on NET PnL after all costs)
                        binary = 1 if pnl_usd > 0 else 0

                        # Create training record
                        record = {
                            'Date/Time': entry_time,
                            'Exit Date/Time': exit_time,
                            'Entry_Price': entry_price,
                            'Exit_Price': exit_price,
                            'y_return': price_return,
                            'y_binary': binary,
                            'y_pnl_usd': pnl_usd,  # Net PnL after commissions
                            'y_pnl_gross': gross_pnl,  # Gross PnL before commissions
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

    # Set commission: $1.00 per side (user requirement)
    cerebro.broker.setcommission(commission=1.00)

    # Set slippage: 1 tick per side (user requirement)
    from strategy.contract_specs import CONTRACT_SPECS
    spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 0.25})
    tick_size = spec["tick_size"]

    # Use FixedSlippage slicer
    cerebro.broker.set_slippage_fixed(tick_size)

    # Enable cheat-on-close to execute at bar close price (not next bar open)
    # This matches live trading: signal at bar close → execute at bar close
    # NOTE: This must be set AFTER slippage for Backtrader compatibility
    cerebro.broker.set_coc(True)

    # Load data for symbol + reference symbols
    # Try to load common reference symbols, but skip gracefully if data is missing/incomplete
    # The strategy will return None for cross-asset features if feeds aren't available
    primary_symbol = symbol
    reference_symbols = ['TLT', 'VIX', 'ES', 'NQ', 'RTY', 'YM',
                         'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S']

    # Remove primary symbol from reference list to avoid duplicates
    reference_symbols = [s for s in reference_symbols if s != primary_symbol]

    symbols_to_load = [primary_symbol]  # Always load primary

    logger.info(f"Loading primary symbol: {primary_symbol}")

    # Try loading reference symbols
    from research.utils.data_loader import load_symbol_data
    for ref_symbol in reference_symbols:
        try:
            hourly_df, daily_df = load_symbol_data(
                ref_symbol,
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date
            )

            # Check if we have sufficient data (at least 1 year)
            if len(hourly_df) >= 252 * 6 and len(daily_df) >= 252:  # ~1 year of hourly/daily
                symbols_to_load.append(ref_symbol)
                logger.info(f"✓ Loaded reference symbol: {ref_symbol}")
            else:
                logger.warning(f"⚠️  Skipping {ref_symbol}: insufficient data ({len(hourly_df)} hourly, {len(daily_df)} daily)")

        except FileNotFoundError:
            logger.debug(f"⚠️  Skipping {ref_symbol}: data file not found")
        except Exception as e:
            logger.warning(f"⚠️  Skipping {ref_symbol}: {e}")

    logger.info(f"Total symbols loaded: {len(symbols_to_load)} (1 primary + {len(symbols_to_load)-1} reference)")

    setup_cerebro_with_data(
        cerebro,
        symbols=symbols_to_load,
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
