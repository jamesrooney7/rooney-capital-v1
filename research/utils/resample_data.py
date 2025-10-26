#!/usr/bin/env python3
"""
Resample tick/minute data to hourly and daily bars for backtesting.

This script converts raw tick data to the two timeframes needed by IbsStrategy:
- Hourly bars (for intraday indicators, IBS, entries)
- Daily bars (for trend filters, daily IBS, SMA200)

Session Handling:
- 24-hour continuous futures trading
- Daily break: 5-6pm ET (session close to next session open)
- Daily bars: 6pm ET (one day) to 5pm ET (next day)

Usage:
    python research/utils/resample_data.py --symbol ES --input data/historical/ES_bt.csv
    python research/utils/resample_data.py --all  # Process all symbols
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick/minute data to hourly OHLCV bars.

    Args:
        df: DataFrame with datetime index and OHLCV columns

    Returns:
        DataFrame with hourly bars
    """
    hourly = pd.DataFrame({
        'Open': df['Open'].resample('1H', label='left', closed='left').first(),
        'High': df['High'].resample('1H', label='left', closed='left').max(),
        'Low': df['Low'].resample('1H', label='left', closed='left').min(),
        'Close': df['Close'].resample('1H', label='left', closed='left').last(),
        'volume': df['volume'].resample('1H', label='left', closed='left').sum(),
    })

    # Remove bars with no data
    hourly = hourly.dropna()

    return hourly


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to daily bars respecting futures sessions.

    Futures session: 6pm ET (one day) to 5pm ET (next day) with 1-hour break.
    Daily bars are anchored at 5pm ET (when session closes).

    Args:
        df: DataFrame with datetime index and OHLCV columns

    Returns:
        DataFrame with daily bars
    """
    # Shift timestamps by 6 hours so that 6pm ET becomes the start of a "day"
    # This makes the daily resampling align with futures sessions
    # Standard offset: ET is UTC-5 (or UTC-4 in DST)
    # We'll use a simple approach: resample on calendar days, then shift

    # For futures, a "day" ends at 5pm ET and starts at 6pm ET
    # We resample using business day frequency but with custom anchoring

    # Create a copy and shift index by 6 hours forward
    # This way 6pm becomes midnight (start of day)
    df_shifted = df.copy()
    df_shifted.index = df_shifted.index + pd.Timedelta(hours=6)

    daily = pd.DataFrame({
        'Open': df_shifted['Open'].resample('1D', label='left', closed='left').first(),
        'High': df_shifted['High'].resample('1D', label='left', closed='left').max(),
        'Low': df_shifted['Low'].resample('1D', label='left', closed='left').min(),
        'Close': df_shifted['Close'].resample('1D', label='left', closed='left').last(),
        'volume': df_shifted['volume'].resample('1D', label='left', closed='left').sum(),
    })

    # Shift timestamps back to original timezone
    daily.index = daily.index - pd.Timedelta(hours=6)

    # Remove bars with no data
    daily = daily.dropna()

    return daily


def resample_symbol(symbol: str, input_path: Path, output_dir: Path, start_date: str = None, end_date: str = None):
    """
    Resample a single symbol's tick data to hourly and daily bars.

    Args:
        symbol: Symbol name (e.g., 'ES', 'NQ')
        input_path: Path to input CSV file
        output_dir: Directory to save resampled files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    """
    logger.info(f"Processing {symbol}...")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Read tick data
    logger.info(f"Reading {input_path}...")
    df = pd.read_csv(
        input_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    logger.info(f"Loaded {len(df):,} tick/minute bars from {df.index[0]} to {df.index[-1]}")

    # Filter by date range if specified
    if start_date:
        df = df[df.index >= start_date]
        logger.info(f"Filtered to start_date >= {start_date}: {len(df):,} bars")

    if end_date:
        df = df[df.index <= end_date]
        logger.info(f"Filtered to end_date <= {end_date}: {len(df):,} bars")

    # Ensure columns are named correctly
    df.columns = ['Open', 'High', 'Low', 'Close', 'volume', 'openinterest']

    # Sort by datetime
    df = df.sort_index()

    # Resample to hourly
    logger.info("Resampling to hourly bars...")
    hourly = resample_to_hourly(df[['Open', 'High', 'Low', 'Close', 'volume']])
    logger.info(f"Created {len(hourly):,} hourly bars")

    # Resample to daily
    logger.info("Resampling to daily bars...")
    daily = resample_to_daily(df[['Open', 'High', 'Low', 'Close', 'volume']])
    logger.info(f"Created {len(daily):,} daily bars")

    # Save resampled data
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_path = output_dir / f"{symbol}_hourly.csv"
    daily_path = output_dir / f"{symbol}_daily.csv"

    logger.info(f"Saving hourly bars to {hourly_path}...")
    hourly.to_csv(hourly_path)

    logger.info(f"Saving daily bars to {daily_path}...")
    daily.to_csv(daily_path)

    logger.info(f"✅ {symbol} complete: {len(hourly):,} hourly, {len(daily):,} daily bars\n")


def main():
    parser = argparse.ArgumentParser(description='Resample tick data to hourly and daily bars')
    parser.add_argument('--symbol', type=str, help='Symbol to process (e.g., ES, NQ)')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output-dir', type=str, default='data/resampled', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Process all symbols in data/historical/')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        # Process all symbols in data/historical/
        historical_dir = Path('data/historical')
        if not historical_dir.exists():
            logger.error(f"Historical data directory not found: {historical_dir}")
            return

        for csv_file in sorted(historical_dir.glob('*_bt.csv')):
            symbol = csv_file.stem.replace('_bt', '')
            resample_symbol(symbol, csv_file, output_dir, args.start_date, args.end_date)

    elif args.symbol and args.input:
        input_path = Path(args.input)
        resample_symbol(args.symbol, input_path, output_dir, args.start_date, args.end_date)

    else:
        parser.print_help()
        logger.error("\nError: Must specify either --all or both --symbol and --input")


if __name__ == '__main__':
    main()
