#!/usr/bin/env python3
"""
Generate ML-filtered backtest trades for all instruments.

This script:
1. Loads pre-trained Random Forest models from JSON metadata
2. Runs IbsStrategy with ML filtering enabled
3. Exports all trades with filter snapshots to CSV

Usage:
    python research/generate_ml_trades.py
    python research/generate_ml_trades.py --symbols ES NQ  # Specific symbols
    python research/generate_ml_trades.py --start 2023-01-01 --end 2023-12-31
    python research/generate_ml_trades.py --data-dir data/historical
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pandas as pd

# Add project root and src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.loader import load_model_bundle, strategy_kwargs_from_bundle
from strategy.ibs_strategy import IbsStrategy
from strategy.filter_column import FilterColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# All trading instruments (excludes TLT and VIX - they're filters only)
# Note: 6N excluded due to missing model file
ALL_INSTRUMENTS = [
    "ES", "NQ", "RTY", "YM",                    # Equity indices
    "GC", "SI", "HG", "CL", "NG", "PL",         # Commodities
    "6A", "6B", "6C", "6E", "6J", "6M", "6S",   # Currencies (6N excluded - no model)
]

# Reference symbols (used for filters but not traded)
REFERENCE_SYMBOLS = ["TLT", "VIX"]

# Date ranges
WARMUP_START = datetime(2022, 1, 1)
WARMUP_END = datetime(2022, 12, 31)
TEST_START = datetime(2023, 1, 1)
TEST_END = datetime(2024, 12, 31)


def load_data_feeds(
    symbol: str,
    data_dir: Path,
    fromdate: datetime = None,
    todate: datetime = None,
) -> tuple[bt.feeds.GenericCSVData | None, bt.feeds.GenericCSVData | None]:
    """Load hourly and daily OHLCV data for a symbol.

    Args:
        symbol: Instrument symbol (e.g., "ES", "6M")
        data_dir: Directory containing resampled CSV files
        fromdate: Start date for data loading
        todate: End date for data loading

    Returns:
        Tuple of (hourly_feed, daily_feed) or (None, None) if files not found
    """
    hourly_path = data_dir / f"{symbol}_hourly.csv"
    daily_path = data_dir / f"{symbol}_daily.csv"

    if not hourly_path.exists() or not daily_path.exists():
        logger.warning(f"Data files not found for {symbol}: {hourly_path} or {daily_path}")
        return None, None

    logger.debug(f"Loading {symbol}: {hourly_path} and {daily_path}")

    try:
        # Load hourly data
        hourly_data = bt.feeds.GenericCSVData(
            dataname=str(hourly_path),
            dtformat="%Y-%m-%d %H:%M:%S",
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            fromdate=fromdate or WARMUP_START,
            todate=todate or TEST_END,
        )

        # Load daily data
        daily_data = bt.feeds.GenericCSVData(
            dataname=str(daily_path),
            dtformat="%Y-%m-%d %H:%M:%S",
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            fromdate=fromdate or WARMUP_START,
            todate=todate or TEST_END,
        )

        return hourly_data, daily_data
    except Exception as e:
        logger.error(f"Failed to load {symbol}: {e}")
        return None, None


def create_filter_columns_from_features(features: list[str]) -> list[FilterColumn]:
    """Create FilterColumn objects from ML feature names.

    This ensures that collect_filter_values() knows which filters to compute.

    Args:
        features: List of feature names from JSON (e.g., ["prev_day_pct", "daily_ibs"])

    Returns:
        List of FilterColumn objects
    """
    columns = []

    for feature in features:
        # Map feature name to parameter name
        param_name = feature

        # Special cases: cross-instrument filters
        if "_z_score" in feature or "_z_pipeline" in feature:
            # e.g., "6m_daily_z_score" → "enable6MZScoreDay"
            parts = feature.split("_")
            if len(parts) >= 3:
                symbol = parts[0].upper()
                # Determine timeframe from feature name
                if "daily" in feature:
                    timeframe = "Day"
                elif "hourly" in feature:
                    timeframe = "Hour"
                else:
                    continue
                param_name = f"enable{symbol}ZScore{timeframe}"

        elif "_return" in feature and not feature.startswith("prev"):
            # e.g., "hg_daily_return" → "enableHGReturnDay"
            parts = feature.split("_")
            if len(parts) >= 3:
                symbol = parts[0].upper()
                # Determine timeframe
                if "daily" in feature:
                    timeframe = "Day"
                elif "hourly" in feature:
                    timeframe = "Hour"
                else:
                    continue
                param_name = f"enable{symbol}Return{timeframe}"

        # Create FilterColumn (column_key, output, output_form, parameter, label)
        columns.append(FilterColumn(
            column_key=feature,
            output="",
            output_form="",
            parameter=param_name,
            label=feature
        ))

    return columns


def get_required_symbols(features: list[str], primary_symbol: str) -> set[str]:
    """Determine which symbols are needed for ML features.

    Args:
        features: List of ML features
        primary_symbol: Primary symbol being traded

    Returns:
        Set of symbols needed (includes primary + cross-instrument references)
    """
    import re

    required = {primary_symbol}

    # Always include reference symbols
    required.update(REFERENCE_SYMBOLS)

    # Define all possible symbols to check for
    all_possible_symbols = set(ALL_INSTRUMENTS) | set(REFERENCE_SYMBOLS)

    # Scan features for cross-instrument references
    for feature in features:
        # Pattern 1: Snake case format like "6m_daily_z_score", "hg_hourly_return"
        if "_z_score" in feature or "_z_pipeline" in feature or "_return" in feature:
            parts = feature.split("_")
            if len(parts) >= 2:
                cross_symbol = parts[0].upper()
                if cross_symbol in all_possible_symbols:
                    required.add(cross_symbol)

        # Pattern 2: Enable format like "enable6MZScoreDay", "enablePLReturnHour"
        # Match: enable{SYMBOL}ZScore{TF} or enable{SYMBOL}Return{TF}
        if feature.startswith("enable"):
            # Try to extract symbol from enable format
            for symbol in all_possible_symbols:
                # Check for ZScore pattern
                if f"enable{symbol}ZScore" in feature:
                    required.add(symbol)
                    break
                # Check for Return pattern
                if f"enable{symbol}Return" in feature:
                    required.add(symbol)
                    break

    return required


def run_backtest_for_symbol(
    symbol: str,
    data_dir: Path,
    models_dir: Path,
    output_dir: Path,
    fromdate: datetime = None,
    todate: datetime = None,
) -> bool:
    """Run ML-filtered backtest for a single symbol.

    Args:
        symbol: Instrument symbol (e.g., "ES")
        data_dir: Directory containing CSV data
        models_dir: Directory containing JSON metadata and RF models
        output_dir: Directory to save output CSV
        fromdate: Start date (includes warmup)
        todate: End date

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol}")
    logger.info(f"{'='*60}")

    # 1. Load ML model bundle
    try:
        bundle = load_model_bundle(symbol, base_dir=models_dir)
        logger.info(f"✓ Loaded ML bundle for {symbol}")
        logger.info(f"  - Features: {len(bundle.features)}")
        logger.info(f"  - Threshold: {bundle.threshold}")
    except Exception as e:
        logger.error(f"✗ Failed to load ML bundle: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Create filter columns from features
    filter_columns = create_filter_columns_from_features(list(bundle.features))
    logger.info(f"  - Filter columns: {len(filter_columns)}")

    # 3. Load ALL available symbols (strategy may need them for internal indicators)
    # IbsStrategy has CROSS_Z_INSTRUMENTS and RETURN_INSTRUMENTS lists that it
    # always tries to initialize, regardless of ML features
    required_symbols = set(ALL_INSTRUMENTS) | set(REFERENCE_SYMBOLS)
    logger.info(f"  - Loading all available symbols: {len(required_symbols)}")
    logger.debug(f"    Symbols: {', '.join(sorted(required_symbols))}")

    # 4. Load all data feeds (both hourly and daily for each symbol)
    loaded_feeds = {}
    for feed_symbol in required_symbols:
        hourly_feed, daily_feed = load_data_feeds(
            feed_symbol,
            data_dir,
            fromdate=fromdate or WARMUP_START,
            todate=todate or TEST_END
        )

        if hourly_feed is None or daily_feed is None:
            # Primary feed is required, reference feeds are optional
            if feed_symbol == symbol:
                logger.error(f"✗ Primary data feeds not found: {feed_symbol}")
                return False
            elif feed_symbol in REFERENCE_SYMBOLS:
                logger.warning(f"  ⚠ Reference data feeds not found: {feed_symbol} (continuing anyway)")
                continue
            else:
                logger.warning(f"  ⚠ Cross-instrument feeds not found: {feed_symbol}")
                continue

        # Store both hourly and daily feeds
        loaded_feeds[f"{feed_symbol}_hour"] = hourly_feed
        loaded_feeds[f"{feed_symbol}_day"] = daily_feed
        logger.debug(f"    ✓ Loaded {feed_symbol} (hourly + daily)")

    if not loaded_feeds:
        logger.error(f"✗ No data feeds loaded for {symbol}")
        return False

    logger.info(f"  - Successfully loaded {len(loaded_feeds)} data feeds ({len(loaded_feeds)//2} symbols)")

    # 5. Setup Cerebro
    # Disable runonce mode to handle data feeds with different lengths
    cerebro = bt.Cerebro(runonce=False)
    cerebro.broker.set_cash(100000)  # Starting capital
    cerebro.broker.setcommission(commission=0.0)  # Commission handled in strategy

    # 6. Add all data feeds with proper naming convention
    # IbsStrategy expects feeds named: {SYMBOL}_hour and {SYMBOL}_day
    for feed_name, feed in loaded_feeds.items():
        cerebro.adddata(feed, name=feed_name)
        logger.debug(f"    Added feed: {feed_name}")

    # 7. Prepare strategy kwargs
    strategy_kwargs = strategy_kwargs_from_bundle(bundle)
    strategy_kwargs.update({
        "symbol": symbol,
        "filter_columns": filter_columns,
        "trade_start": TEST_START.date() if fromdate is None else fromdate.date(),  # Only trade after warmup
    })

    logger.info(f"  - ML threshold: {strategy_kwargs.get('ml_threshold')}")
    logger.info(f"  - Trade start: {strategy_kwargs.get('trade_start')}")

    # 8. Add strategy
    cerebro.addstrategy(IbsStrategy, **strategy_kwargs)

    # 9. Run backtest
    logger.info(f"Running backtest...")
    try:
        results = cerebro.run()
        strategy = results[0]
    except Exception as e:
        logger.error(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 10. Get trades
    trades = strategy.trade_report()
    logger.info(f"✓ Generated {len(trades)} trades")

    if not trades:
        logger.warning(f"⚠ No trades generated for {symbol}")
        # Create empty CSV with proper columns
        df = pd.DataFrame(columns=[
            'dt', 'instrument', 'signal', 'price', 'size',
            'ibs_value', 'sma200', 'tlt_sma20',
            'slippage_usd', 'commission_usd', 'pnl', 'ml_score'
        ])
    else:
        # 11. Convert to DataFrame
        df = pd.DataFrame(trades)

        # 12. Filter to test period only (exclude warmup trades) if using default dates
        if fromdate is None:
            df['dt'] = pd.to_datetime(df['dt'])
            df = df[df['dt'] >= TEST_START]
            logger.info(f"  - Trades after warmup filter: {len(df)}")

    # 13. Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_ml_trades_2023_2024.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved trades to: {output_path}")

    # 14. Print summary stats
    if len(df) > 0:
        winning_trades = df[df['pnl'] > 0] if 'pnl' in df.columns else pd.DataFrame()
        losing_trades = df[df['pnl'] < 0] if 'pnl' in df.columns else pd.DataFrame()

        logger.info(f"\n{symbol} Summary:")
        logger.info(f"  Total Trades: {len(df)}")
        if len(winning_trades) > 0 or len(losing_trades) > 0:
            logger.info(f"  Winners: {len(winning_trades)} ({len(winning_trades)/len(df)*100:.1f}%)")
            logger.info(f"  Losers: {len(losing_trades)} ({len(losing_trades)/len(df)*100:.1f}%)")
            logger.info(f"  Total PnL: ${df['pnl'].sum():.2f}")
            logger.info(f"  Avg PnL: ${df['pnl'].mean():.2f}")

        # Check ML scores
        if 'ml_score' in df.columns:
            valid_scores = df['ml_score'].dropna()
            if len(valid_scores) > 0:
                logger.info(f"  ML Score Range: {valid_scores.min():.3f} - {valid_scores.max():.3f}")
                logger.info(f"  ML Score Avg: {valid_scores.mean():.3f}")

                # Warning if all scores are at/near threshold
                threshold = strategy_kwargs.get('ml_threshold', 0.5)
                if valid_scores.mean() < threshold + 0.05:
                    logger.warning(f"  ⚠ Avg ML score very close to threshold!")
            else:
                logger.warning(f"  ⚠ All ML scores are NaN! (Check feature computation)")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ML-filtered backtest trades for all trading instruments"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=ALL_INSTRUMENTS,
        help=f"Symbols to process (default: all {len(ALL_INSTRUMENTS)} instruments)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/resampled"),
        help="Directory containing resampled CSV data files (default: data/resampled)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=SRC / "models",
        help="Directory containing JSON metadata and RF models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ml_backtest_trades",
        help="Directory to save output CSVs"
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Warmup start date (YYYY-MM-DD). Default: 2022-01-01"
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Test end date (YYYY-MM-DD). Default: 2024-12-31"
    )
    parser.add_argument(
        "--test-start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Test period start (trades before this are discarded). Default: 2023-01-01"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Update logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Update global date ranges if specified
    global WARMUP_START, TEST_START, TEST_END
    if args.start:
        WARMUP_START = args.start
    if args.test_start:
        TEST_START = args.test_start
    if args.end:
        TEST_END = args.end

    # Print configuration
    logger.info("="*60)
    logger.info("ML-Filtered Backtest Data Generator")
    logger.info("="*60)
    logger.info(f"Warmup Period: {WARMUP_START.date()} to {WARMUP_END.date()}")
    logger.info(f"Test Period: {TEST_START.date()} to {TEST_END.date()}")
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Data Dir: {args.data_dir}")
    logger.info(f"Models Dir: {args.models_dir}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info("="*60)

    # Process each symbol
    results = {}
    successful = 0
    failed = 0

    for symbol in args.symbols:
        success = run_backtest_for_symbol(
            symbol=symbol,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            fromdate=args.start,
            todate=args.end,
        )

        if success:
            results[symbol] = "✓ SUCCESS"
            successful += 1
        else:
            results[symbol] = "✗ FAILED"
            failed += 1

    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("Final Results:")
    logger.info(f"{'='*60}")
    for symbol, status in results.items():
        logger.info(f"  {symbol}: {status}")

    logger.info(f"\nSummary: {successful} succeeded, {failed} failed")

    if successful > 0:
        logger.info(f"✓ Output files saved to: {args.output_dir}")

    # Exit with error code if any failed
    if failed > 0:
        logger.error(f"\n✗ {failed} instrument(s) failed - check logs above")
        sys.exit(1)

    logger.info(f"\n✓ All {successful} backtests completed successfully!")


if __name__ == "__main__":
    main()
