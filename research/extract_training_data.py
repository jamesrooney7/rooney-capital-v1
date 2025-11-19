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

# Configure logging - keep at INFO to ensure all loggers work properly
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
        # Get output path before calling super().__init__
        self.output_csv_path = kwargs.pop('output_csv_path', None)
        self.filter_year = kwargs.pop('filter_year', None)  # Optional: only keep trades from this year

        # Initialize trade tracking
        self.trade_entry_features = {}  # {order_id: features_dict}
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers_written = False
        self.trade_count = 0
        self.filtered_count = 0

        # Track column filtering stats
        self.filtered_columns = set()
        self.kept_columns = set()
        self.column_filter_logged = False

        super().__init__(*args, **kwargs)

        # Populate comprehensive list of all possible filter parameter keys
        # This tells collect_filter_values() to calculate ALL features
        self.ml_feature_param_keys = FeatureLoggingStrategy._get_all_filter_param_keys()

        # Pre-build all possible CSV fieldnames (core + features, filtered)
        self.csv_fieldnames = self._build_csv_fieldnames()

    @staticmethod
    def _get_all_filter_param_keys() -> set:
        """
        Return comprehensive set of all filter parameter keys.

        This ensures collect_filter_values() calculates ALL possible features,
        not just the ones explicitly enabled in the strategy config.
        """
        keys = {
            # Calendar filters
            'allowedDOW', 'allowedMon', 'domDay',  # Removed enableDOM duplicate
            'enableBegWeek', 'enableEvenOdd',

            # Price/return filters (friendly names only - enable* are duplicates)
            'prev_day_pct',  # Removed enablePrevDayPct duplicate
            'prev_bar_pct',  # Removed enablePrevBarPct duplicate

            # IBS filters (friendly names only - enable* are duplicates)
            'ibs',  # Removed enableIBSEntry/enableIBSExit duplicates
            'daily_ibs',  # Removed enableDailyIBS duplicate
            'enablePrevIBS', 'prev_ibs',
            'enablePrevIBSDaily', 'prev_daily_ibs',

            # Pair filters (friendly names only - enable* are duplicates)
            'pair_ibs',  # Removed enablePairIBS duplicate
            'pair_z',  # Removed enablePairZ duplicate

            # RSI filters
            # Note: Some RSI features only populate enable* version, not friendly name
            'enableRSIEntry',  # Only enable* has data
            'enableRSIEntry2Len',  # Only enable* has data
            'enableRSIEntry14Len',  # Only enable* has data
            'enableRSIEntry2',  # Only enable* has data
            'daily_rsi',  # Removed enableDailyRSI duplicate
            # enableDailyRSI2Len removed - daily_rsi2_len is empty
            'enableDailyRSI14Len',  # Only enable* has data

            # Bollinger Bands (only enable* has data - friendly names empty)
            'enableBBHigh',
            'enableBBHighD',

            # EMA filters (only enable* has data)
            'enableEMA8',
            'enableEMA20',

            # ATR filters (friendly names only - enable* are duplicates)
            'atrz',  # Removed enableATRZ duplicate
            'enableHourlyATRPercentile',  # Only enable* has data

            # Volume filters (friendly name only - enable* is duplicate)
            'volz',  # Removed enableVolZ duplicate

            # Momentum/distance/price z-score filters (friendly names only)
            'dist_z',  # Removed enableDistZ duplicate
            'enableMom3', 'mom3_z',   # 3-period momentum z-score
            'enableZScore', 'z_score', # Generic price z-score

            # Daily ATR/volume (friendly names only - enable* are duplicates)
            'datrz',  # Removed enableDATRZ duplicate
            'dvolz',  # Removed enableDVolZ duplicate

            # Trend/ratio filters
            # Note: TRATR feature not implemented - excluded
            # Note: supply_zone/use_supply_zone not implemented - excluded

            # N7 bar (only enable* has data)
            'enableN7Bar',

            # Inside bar (only enable* has data)
            'enableInsideBar',

            # Bear count (only enable* has data)
            'enableBearCount',

            # Spiral Efficiency Ratio (only enable* has data)
            'enableSpiralER',

            # TWRC (only enable* has data)
            'enableTWRC',

            # Note: VIX regime excluded - VIX data not loaded, features filtered anyway
        }

        # Add cross-asset z-score and return keys dynamically
        # Note: VIX excluded - we filter VIX features in output CSV and don't load VIX data
        cross_symbols = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S', 'TLT']
        timeframes = ['Hour', 'Day']

        for symbol in cross_symbols:
            for tf in timeframes:
                # Z-score feature keys (not enable params - those get filtered out)
                # Feature keys must match ibs_strategy.py _metadata_feature_key() format:
                # "Hour" -> "hourly", "Day" -> "daily"
                tf_label = 'hourly' if tf == 'Hour' else 'daily'
                keys.add(f'{symbol.lower()}_{tf_label}_z_score')
                keys.add(f'{symbol.lower()}_{tf_label}_z_pipeline')

                # Return feature keys (one per symbol, not per timeframe)
                # Note: *_return_pipeline removed - perfect duplicates of *_return
                if tf == 'Hour':  # Only add once to avoid duplicates
                    keys.add(f'{symbol.lower()}_daily_return')
                    keys.add(f'{symbol.lower()}_hourly_return')

        return keys

    def _should_keep_column(self, col_name: str) -> bool:
        """
        Determine if a column should be kept in the output CSV.

        Filters out:
        1. Title case columns (columns with spaces) - these are duplicates
        2. Cross-asset enable* parameters - these have friendly-named alternatives
        3. VIX features - excluded per user requirement

        Keeps:
        - Core columns (dates, prices, labels)
        - All feature values including enable* params that don't have friendly alternatives
          (e.g., enableEMA8, enableRSIEntry are kept because they don't have ema8/rsi_entry columns)

        Args:
            col_name: Column name to check

        Returns:
            True if column should be kept, False if it should be filtered
        """
        # Always keep core columns
        core_columns = {
            'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',
            'Entry_Price', 'Exit_Price', 'Symbol', 'Trade_ID',
            'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross'
        }
        if col_name in core_columns:
            return True

        # Filter out title case columns (have spaces)
        if ' ' in col_name:
            return False

        # Filter out VIX features (case-insensitive)
        if 'vix' in col_name.lower():
            return False

        # Filter out *_return_pipeline columns (perfect duplicates of *_return)
        # Keep z_pipeline columns as they may differ from z_score
        if col_name.endswith('_return_pipeline'):
            return False

        # Filter out enable* params that have friendly name alternatives with data
        # Cross-asset patterns: enableXXZScore(Hour|Day), enableXXReturn(Hour|Day)
        # Specific duplicates confirmed by correlation analysis (r=1.0)
        if col_name.lower().startswith('enable'):
            import re
            # Pattern: enableXXZScore(Hour|Day) or enableXXReturn(Hour|Day)
            if re.match(r'enable[A-Z0-9]+ZScore(Hour|Day)', col_name):
                return False  # Has corresponding _z_score column
            if re.match(r'enable[A-Z0-9]+Return(Hour|Day)', col_name):
                return False  # Has corresponding _return column

            # Filter specific enable* params that are perfect duplicates (r=1.0)
            duplicates_to_filter = {
                # Friendly names that have data (keep these instead of enable*)
                'enableDailyRSI',  # Keep daily_rsi
                'enableDailyRSI2Len',  # Keep daily_rsi2_len
                'enablePrevDayPct',  # Keep prev_day_pct
                'enablePrevBarPct',  # Keep prev_bar_pct
                # Perfect duplicates found by correlation analysis
                'enableATRZ',  # Keep atrz
                'enableVolZ',  # Keep volz
                'enableDATRZ',  # Keep datrz
                'enableDVolZ',  # Keep dvolz
                'enableDistZ',  # Keep dist_z
                'enableDOM',  # Keep domDay
                'enableDailyIBS',  # Keep daily_ibs
                'enablePairIBS',  # Keep pair_ibs
                'enablePairZ',  # Keep pair_z
                'enableIBSEntry',  # Keep ibs (IBS entry/exit/ibs are identical)
                'enableIBSExit',  # Keep ibs
            }
            if col_name in duplicates_to_filter:
                return False

        # Keep everything else
        return True

    def _build_csv_fieldnames(self) -> list:
        """
        Build complete list of CSV fieldnames upfront.

        This prevents errors when later trades have features not in the first trade.
        Includes core columns + all possible features (filtered).

        Returns:
            Sorted list of column names for CSV header
        """
        # Core columns that always appear
        core_columns = [
            'Date/Time', 'Exit Date/Time', 'Date', 'Exit_Date',
            'Entry_Price', 'Exit_Price', 'Symbol', 'Trade_ID',
            'y_return', 'y_binary', 'y_pnl_usd', 'y_pnl_gross'
        ]

        # Get all possible feature keys
        all_features = self._get_all_filter_param_keys()

        # Combine and filter
        all_columns = set(core_columns) | all_features
        kept_columns = [col for col in all_columns if self._should_keep_column(col)]

        # Sort for consistent column order
        # Put core columns first, then alphabetical
        core_first = [col for col in core_columns if col in kept_columns]
        features_sorted = sorted([col for col in kept_columns if col not in core_columns])

        return core_first + features_sorted

    def _filter_record_columns(self, record: dict) -> dict:
        """
        Filter record columns to remove unwanted features.

        Args:
            record: Dictionary of column name -> value

        Returns:
            Filtered dictionary with only kept columns
        """
        filtered_record = {}

        for col_name, value in record.items():
            if self._should_keep_column(col_name):
                filtered_record[col_name] = value
                self.kept_columns.add(col_name)
            else:
                self.filtered_columns.add(col_name)

        return filtered_record

    def _write_trade_to_csv(self, record: dict):
        """Write a single trade record to CSV immediately."""
        import csv
        from pathlib import Path

        if self.output_csv_path is None:
            logger.warning("No output CSV path set, trade not written to disk")
            return

        # Filter by year if specified (for 2-year chunk extraction with warmup)
        if self.filter_year is not None:
            entry_time = record.get('Date/Time')
            if entry_time is not None:
                trade_year = entry_time.year if hasattr(entry_time, 'year') else None
                if trade_year != self.filter_year:
                    self.filtered_count += 1
                    return  # Skip trades not in the target year (warmup data)

        try:
            # Filter columns to remove title case, enable*, and VIX features
            filtered_record = self._filter_record_columns(record)

            # Log column filtering on first trade
            if not self.column_filter_logged:
                original_count = len(record)
                filtered_count = len(filtered_record)
                removed_count = original_count - filtered_count
                logger.info(f"Column filtering: {original_count} total → {filtered_count} kept, {removed_count} removed")
                logger.info(f"  Removed: title case ({sum(1 for c in self.filtered_columns if ' ' in c)}), "
                           f"enable* ({sum(1 for c in self.filtered_columns if c.lower().startswith('enable'))}), "
                           f"VIX ({sum(1 for c in self.filtered_columns if 'vix' in c.lower())})")
                self.column_filter_logged = True

            # Open file on first trade
            if self.csv_file is None:
                output_path = Path(self.output_csv_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self.csv_file = open(output_path, 'w', newline='')
                # Use pre-built fieldnames to handle all possible columns
                self.csv_writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=self.csv_fieldnames,
                    extrasaction='ignore'  # Ignore any extra fields not in header
                )
                self.csv_writer.writeheader()
                self.csv_headers_written = True
                logger.info(f"Opened CSV file for incremental writing: {output_path}")
                logger.info(f"CSV header has {len(self.csv_fieldnames)} columns")
                if self.filter_year:
                    logger.info(f"Filtering trades to year {self.filter_year} only (warmup year discarded)")

            # Write the filtered trade
            self.csv_writer.writerow(filtered_record)
            self.csv_file.flush()  # Ensure it's written to disk
            self.trade_count += 1

        except Exception as e:
            logger.error(f"Error writing trade to CSV: {e}", exc_info=True)

    def stop(self):
        """Called by Backtrader when strategy finishes - close CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()
            logger.info(f"Closed CSV file after writing {self.trade_count} trades")
            logger.info(f"Final column summary: {len(self.kept_columns)} columns kept, {len(self.filtered_columns)} columns filtered")
            if self.filter_year and self.filtered_count > 0:
                logger.info(f"Filtered out {self.filtered_count} warmup trades (not in year {self.filter_year})")
        super().stop()

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

                        # Add Date column for compatibility
                        record['Date'] = entry_time.date()

                        # Write immediately to CSV (incremental writing)
                        self._write_trade_to_csv(record)

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
    filter_year: int = None,  # Optional: only keep trades from this year (discard warmup)
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
        filter_year: Optional year to filter trades (e.g., 2011). Only trades with entry in this year are kept.

    Returns:
        DataFrame with training data
    """
    logger.info(f"Extracting training data for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Create Cerebro instance
    cerebro = bt.Cerebro(runonce=False)

    # Set initial cash - use very high amount for data extraction
    # During feature extraction, we don't care about realistic capital constraints
    # We want to capture ALL base IBS trades regardless of account balance
    cerebro.broker.setcash(1_000_000_000.0)  # $1 billion - no margin issues!

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
    # Note: VIX removed since we filter out VIX features in output CSV
    primary_symbol = symbol
    reference_symbols = ['TLT', 'ES', 'NQ', 'RTY', 'YM',
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

    # Determine output path
    if output_path is None:
        output_dir = Path('data/training')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_transformed_features.csv"

    # Build ml_features list to trigger indicator creation
    # Even though we're not using ML during extraction, passing ml_features
    # tells the strategy which indicators to create
    ml_features = list(FeatureLoggingStrategy._get_all_filter_param_keys())
    logger.info(f"Requesting {len(ml_features)} features for calculation")

    # Build enable parameters dict - set all enable* params to True
    # This forces the strategy to create ALL indicators
    enable_params = {}
    for key in ml_features:
        if key.lower().startswith('enable'):
            enable_params[key] = True

    # Also enable key indicator parameters
    enable_params.update({
        'use_supply_zone': True,
        'use_val_filter': False,  # Keep disabled - not needed
    })

    logger.info(f"Enabling {len(enable_params)} indicator parameters")

    # Add feature-logging strategy with incremental CSV writing
    strat_params = {
        'symbol': symbol,
        'output_csv_path': str(output_path),
        'ml_features': ml_features,  # Trigger indicator creation
        **enable_params,  # Enable all indicators
    }
    if filter_year is not None:
        strat_params['filter_year'] = filter_year
        logger.info(f"Filtering output to trades from year {filter_year} only")
    logger.info(f"Adding FeatureLoggingStrategy with incremental CSV writing to: {output_path}")
    cerebro.addstrategy(FeatureLoggingStrategy, **strat_params)

    # Run backtest (trades written incrementally to CSV during execution)
    logger.info("Running backtest to extract features (writing incrementally to CSV)...")
    results = cerebro.run()
    strat = results[0]

    # Verify CSV was created and has content
    if not output_path.exists():
        logger.warning("No CSV file created! Check strategy parameters.")
        return pd.DataFrame()

    # Read the CSV to verify and return
    df = pd.read_csv(output_path)
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Shape: {df.shape} (rows={len(df):,}, columns={len(df.columns)})")

    # Show date range
    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        logger.info(f"Date range: {df['Date/Time'].min()} to {df['Date/Time'].max()}")

    # Show target distribution
    if 'y_binary' in df.columns:
        wins = (df['y_binary'] == 1).sum()
        losses = (df['y_binary'] == 0).sum()
        win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0
        logger.info(f"Target distribution: {wins:,} wins ({win_rate:.1f}%), {losses:,} losses")

    # Show missing value summary
    missing_counts = df.isna().sum()
    cols_with_missing = (missing_counts > 0).sum()
    if cols_with_missing > 0:
        logger.info(f"Missing values: {cols_with_missing} columns have some missing data")
        # Show worst offenders
        worst = missing_counts.nlargest(5)
        for col, count in worst.items():
            pct = (count / len(df)) * 100
            if pct > 0:
                logger.info(f"  {col}: {count:,} missing ({pct:.1f}%)")
    else:
        logger.info("Missing values: None - all columns fully populated!")

    logger.info("=" * 80)

    if len(df) == 0:
        logger.warning("CSV has no trades! Check strategy parameters.")
        return pd.DataFrame()

    return df


def main():
    parser = argparse.ArgumentParser(description='Extract training data from IbsStrategy backtests')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to extract (e.g., ES)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--filter-year', type=int, help='Only keep trades from this year (e.g., 2011)')

    args = parser.parse_args()

    try:
        df = extract_training_data(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            output_path=args.output,
            filter_year=args.filter_year,
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
