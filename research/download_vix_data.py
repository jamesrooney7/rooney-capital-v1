#!/usr/bin/env python3
"""
Download VIX data from Yahoo Finance and resample to match the format expected by IbsStrategy.

VIX is a volatility index, not a tradeable instrument, so intraday data may be limited.
This script downloads daily VIX data and resamples it to hourly format by forward-filling.

Usage:
    python research/download_vix_data.py --start 2010-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_vix_data(start_date: str, end_date: str, output_dir: str = 'data/resampled'):
    """
    Download VIX data from Yahoo Finance and save in backtrader-compatible format.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for resampled data
    """
    logger.info(f"Downloading VIX data: {start_date} to {end_date}")

    # Try importing yfinance
    try:
        import yfinance as yf
    except ImportError:
        logger.error("❌ yfinance not installed. Install with: pip install yfinance")
        return False

    # Download VIX data (^VIX is the Yahoo Finance ticker)
    logger.info("Downloading from Yahoo Finance (^VIX)...")
    try:
        vix = yf.download(
            '^VIX',
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,  # Use adjusted prices
        )
    except Exception as e:
        logger.error(f"❌ Failed to download VIX data: {e}")
        return False

    if vix.empty:
        logger.error("❌ No VIX data returned from Yahoo Finance")
        return False

    logger.info(f"✅ Downloaded {len(vix)} daily bars")

    # Ensure we have the expected columns
    # After auto_adjust=True, columns are: Open, High, Low, Close, Volume
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in vix.columns:
            logger.error(f"❌ Missing column: {col}")
            logger.error(f"Available columns: {list(vix.columns)}")
            return False

    # Reset index to get Date as a column
    vix_daily = vix.reset_index()
    vix_daily.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # VIX is an index (not traded), so volume is not meaningful
    # Set volume to 0 or a placeholder
    vix_daily['volume'] = 0

    logger.info(f"Daily bars: {len(vix_daily)} from {vix_daily['datetime'].min()} to {vix_daily['datetime'].max()}")

    # Create hourly data by resampling daily data
    # VIX daily values will be forward-filled for all hourly bars in that day
    logger.info("Creating hourly data from daily data...")

    # For each daily bar, create hourly bars (9:30 AM to 4:00 PM ET = 7 hours)
    # We'll create bars at: 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00 (7 bars per day)
    hourly_records = []

    for idx, row in vix_daily.iterrows():
        date = pd.to_datetime(row['datetime']).date()

        # Create 7 hourly bars per day (market hours)
        for hour in [10, 11, 12, 13, 14, 15, 16]:
            hourly_dt = datetime.combine(date, datetime.min.time().replace(hour=hour))

            # All hourly bars within a day have the same OHLC as the daily bar
            hourly_records.append({
                'datetime': hourly_dt,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': 0,
            })

    vix_hourly = pd.DataFrame(hourly_records)
    logger.info(f"✅ Created {len(vix_hourly)} hourly bars from {len(vix_daily)} daily bars")

    # Save to CSV files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hourly_file = output_path / 'VIX_hourly.csv'
    daily_file = output_path / 'VIX_daily.csv'

    # Save hourly data
    vix_hourly.to_csv(hourly_file, index=False)
    logger.info(f"✅ Saved hourly data: {hourly_file} ({len(vix_hourly)} bars)")

    # Save daily data
    vix_daily.to_csv(daily_file, index=False)
    logger.info(f"✅ Saved daily data: {daily_file} ({len(vix_daily)} bars)")

    # Show sample data
    logger.info("\nSample hourly data (first 10 bars):")
    logger.info(vix_hourly.head(10).to_string(index=False))

    logger.info("\nSample daily data (first 5 bars):")
    logger.info(vix_daily.head(5).to_string(index=False))

    return True


def main():
    parser = argparse.ArgumentParser(description='Download VIX data from Yahoo Finance')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='data/resampled', help='Output directory')

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info("VIX DATA DOWNLOADER")
    logger.info(f"{'='*80}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Output: {args.output_dir}/")
    logger.info(f"{'='*80}\n")

    success = download_vix_data(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir
    )

    if success:
        logger.info(f"\n{'='*80}")
        logger.info("✅ VIX DATA DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nFiles created:")
        logger.info(f"  - {args.output_dir}/VIX_hourly.csv")
        logger.info(f"  - {args.output_dir}/VIX_daily.csv")
        logger.info(f"\nYou can now run backtests with VIX features enabled.")
        logger.info(f"{'='*80}\n")
        return 0
    else:
        logger.error(f"\n{'='*80}")
        logger.error("❌ VIX DATA DOWNLOAD FAILED")
        logger.error(f"{'='*80}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
