#!/usr/bin/env python3
"""
Chunked extraction script to prevent memory accumulation.

Extracts training data in multi-year chunks, then concatenates.
This keeps Backtrader's indicator arrays bounded and prevents OOM.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

# Use existing extraction function
from extract_training_data import extract_training_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_in_chunks(
    symbol: str,
    start_date: str,
    end_date: str,
    chunk_years: int = 5,
    data_dir: str = 'data/resampled',
    output_path: str = None
):
    """
    Extract training data in time chunks to prevent memory accumulation.

    Args:
        symbol: Trading symbol (e.g., 'ES')
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)
        chunk_years: Years per chunk (default: 5)
        data_dir: Directory containing resampled data
        output_path: Final output CSV path (optional)

    Returns:
        Combined DataFrame of all chunks
    """
    from datetime import datetime, timedelta

    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Generate chunk boundaries
    chunks = []
    current_start = start

    while current_start < end:
        current_end = datetime(current_start.year + chunk_years, 1, 1) - timedelta(days=1)
        if current_end > end:
            current_end = end

        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d')
        ))

        current_start = current_end + timedelta(days=1)

    logger.info(f"Extracting {symbol} in {len(chunks)} chunks of ~{chunk_years} years each")

    # Extract each chunk
    chunk_dfs = []
    temp_dir = Path('data/training/temp_chunks')
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)}: {chunk_start} to {chunk_end}")

        chunk_output = temp_dir / f"{symbol}_chunk_{i}.csv"

        try:
            # Extract this chunk
            df_chunk = extract_training_data(
                symbol=symbol,
                start_date=chunk_start,
                end_date=chunk_end,
                data_dir=data_dir,
                output_path=str(chunk_output)
            )

            if not df_chunk.empty:
                chunk_dfs.append(df_chunk)
                logger.info(f"✓ Chunk {i} complete: {len(df_chunk)} trades")
            else:
                logger.warning(f"⚠ Chunk {i} had no trades")

        except Exception as e:
            logger.error(f"✗ Chunk {i} failed: {e}", exc_info=True)
            # Continue with other chunks even if one fails

    if not chunk_dfs:
        logger.error("No chunks extracted successfully!")
        return pd.DataFrame()

    # Concatenate all chunks
    logger.info(f"Concatenating {len(chunk_dfs)} chunks...")
    df_combined = pd.concat(chunk_dfs, ignore_index=True)

    # Sort by entry time
    df_combined = df_combined.sort_values('Date/Time').reset_index(drop=True)

    logger.info(f"Combined total: {len(df_combined)} trades")

    # Save combined CSV
    if output_path is None:
        output_dir = Path('data/training')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_transformed_features.csv"

    df_combined.to_csv(output_path, index=False)
    logger.info(f"✅ Saved combined data to: {output_path}")

    # Clean up temp chunks
    for chunk_file in temp_dir.glob(f"{symbol}_chunk_*.csv"):
        chunk_file.unlink()
    logger.info(f"Cleaned up temporary chunk files")

    return df_combined


def main():
    parser = argparse.ArgumentParser(description='Extract training data in chunks to prevent memory issues')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to extract (e.g., ES)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--chunk-years', type=int, default=5, help='Years per chunk (default: 5)')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')
    parser.add_argument('--output', type=str, help='Output CSV path')

    args = parser.parse_args()

    try:
        df = extract_in_chunks(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            chunk_years=args.chunk_years,
            data_dir=args.data_dir,
            output_path=args.output
        )

        if df.empty:
            logger.error("No training data extracted!")
            return 1

        logger.info("✅ Chunked extraction complete!")
        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
