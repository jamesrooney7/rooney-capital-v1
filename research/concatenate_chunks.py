#!/usr/bin/env python3
"""
Concatenate 2-year chunk CSV files into final training data files.

This script merges all year-specific chunks (e.g., ES_2011.csv, ES_2012.csv, ...)
into a single final CSV file (e.g., ES_transformed_features.csv).

Usage:
    python3 research/concatenate_chunks.py
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNKS_DIR = Path('data/training_chunks')
OUTPUT_DIR = Path('data/training')
YEARS = list(range(2011, 2025))  # 2011 through 2024

# All symbols
SYMBOLS = [
    "ES", "NQ", "RTY", "YM",
    "GC", "SI", "HG", "CL", "NG", "PL",
    "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S"
]


def concatenate_symbol_chunks(symbol: str) -> bool:
    """
    Concatenate all year chunks for a symbol into final CSV.

    Args:
        symbol: Symbol to concatenate (e.g., 'ES')

    Returns:
        True if successful, False if failed
    """
    logger.info(f"Processing {symbol}...")

    # Find all chunk files for this symbol
    chunk_files = []
    missing_years = []

    for year in YEARS:
        chunk_path = CHUNKS_DIR / f"{symbol}_{year}.csv"
        if chunk_path.exists():
            chunk_files.append((year, chunk_path))
        else:
            missing_years.append(year)

    if not chunk_files:
        logger.error(f"  ✗ {symbol}: No chunk files found!")
        return False

    if missing_years:
        logger.warning(f"  ⚠️  {symbol}: Missing years: {missing_years}")

    # Read and concatenate chunks in chronological order
    logger.info(f"  Reading {len(chunk_files)} chunks...")
    chunks = []

    for year, chunk_path in sorted(chunk_files):
        try:
            df = pd.read_csv(chunk_path)
            logger.info(f"    {year}: {len(df)} trades")
            chunks.append(df)
        except Exception as e:
            logger.error(f"    ✗ Error reading {chunk_path}: {e}")
            return False

    # Concatenate all chunks
    logger.info(f"  Concatenating {len(chunks)} chunks...")
    final_df = pd.concat(chunks, ignore_index=True)

    # Sort by Date/Time to ensure chronological order
    if 'Date/Time' in final_df.columns:
        final_df['Date/Time'] = pd.to_datetime(final_df['Date/Time'])
        final_df = final_df.sort_values('Date/Time').reset_index(drop=True)

    # Save final CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{symbol}_transformed_features.csv"

    logger.info(f"  Writing final CSV: {output_path}")
    final_df.to_csv(output_path, index=False)

    logger.info(f"  ✓ {symbol}: {len(final_df)} total trades, {len(final_df.columns)} features")

    return True


def main():
    """Concatenate all symbols."""
    logger.info("========================================")
    logger.info("Concatenating 2-Year Chunks")
    logger.info("========================================")
    logger.info(f"Input directory: {CHUNKS_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Symbols to process: {len(SYMBOLS)}")
    logger.info(f"Years expected: {min(YEARS)}-{max(YEARS)} ({len(YEARS)} years)")
    logger.info("========================================")
    logger.info("")

    if not CHUNKS_DIR.exists():
        logger.error(f"Chunks directory not found: {CHUNKS_DIR}")
        logger.error("Run extract_2year_chunks.sh first!")
        return 1

    # Process each symbol
    success_count = 0
    failed_symbols = []

    for symbol in SYMBOLS:
        success = concatenate_symbol_chunks(symbol)
        if success:
            success_count += 1
        else:
            failed_symbols.append(symbol)
        logger.info("")

    # Summary
    logger.info("========================================")
    logger.info("Concatenation Complete!")
    logger.info("========================================")
    logger.info(f"Successfully processed: {success_count}/{len(SYMBOLS)} symbols")

    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols}")
        logger.warning("Check logs above for details")
    else:
        logger.info("✓ All symbols processed successfully!")

    logger.info("")
    logger.info(f"Final CSV files saved to: {OUTPUT_DIR}")

    return 0 if not failed_symbols else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
