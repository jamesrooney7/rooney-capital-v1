"""
Data loader for Strategy Factory.

Loads resampled OHLCV data from local CSV files.
Supports 15-min, 1H, and daily timeframes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Data directory (relative to project root)
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "resampled"


def load_data(
    symbol: str,
    timeframe: str = "15min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol.

    Args:
        symbol: Symbol code (e.g., 'ES', 'NQ', 'GC')
        timeframe: Timeframe ('15min', '1H', '1D')
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: datetime, Open, High, Low, Close, volume

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is empty or invalid
    """
    # Construct file path
    file_path = DATA_DIR / f"{symbol}_{timeframe}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Available files: {list(DATA_DIR.glob(f'{symbol}_*.csv'))}"
        )

    logger.info(f"Loading data from {file_path}")

    # Load CSV
    df = pd.read_csv(
        file_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # Validate columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter by date range
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # Validate data
    if df.empty:
        raise ValueError(
            f"No data found for {symbol} {timeframe} "
            f"between {start_date} and {end_date}"
        )

    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"Found NaN values in {symbol} data: {nan_counts[nan_counts > 0]}")
        # Forward fill NaN values
        df = df.fillna(method='ffill')

    logger.info(
        f"Loaded {len(df):,} bars for {symbol} "
        f"from {df.index[0]} to {df.index[-1]}"
    )

    return df


def load_multiple_symbols(
    symbols: List[str],
    timeframe: str = "15min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols.

    Args:
        symbols: List of symbol codes
        timeframe: Timeframe for all symbols
        start_date: Start date
        end_date: End date

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    data = {}
    failed = []

    for symbol in symbols:
        try:
            df = load_data(symbol, timeframe, start_date, end_date)
            data[symbol] = df
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            failed.append(symbol)

    if failed:
        logger.warning(f"Failed to load symbols: {failed}")

    logger.info(f"Successfully loaded {len(data)}/{len(symbols)} symbols")

    return data


def get_available_symbols(timeframe: str = "15min") -> List[str]:
    """
    Get list of available symbols for a timeframe.

    Args:
        timeframe: Timeframe to check

    Returns:
        List of symbol codes
    """
    pattern = f"*_{timeframe}.csv"
    files = list(DATA_DIR.glob(pattern))

    # Extract symbol from filename (e.g., 'ES_15min.csv' -> 'ES')
    symbols = [f.stem.replace(f'_{timeframe}', '') for f in files]
    symbols.sort()

    logger.info(f"Found {len(symbols)} symbols for {timeframe}: {symbols}")

    return symbols


def get_data_date_range(symbol: str, timeframe: str = "15min") -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the date range for a symbol's data.

    Args:
        symbol: Symbol code
        timeframe: Timeframe

    Returns:
        Tuple of (start_date, end_date)
    """
    df = load_data(symbol, timeframe)
    return df.index[0], df.index[-1]


def align_data(
    data_dict: dict[str, pd.DataFrame],
    method: str = "inner"
) -> dict[str, pd.DataFrame]:
    """
    Align multiple dataframes to have the same datetime index.

    Useful for cross-asset strategies that need synchronized data.

    Args:
        data_dict: Dictionary of symbol -> DataFrame
        method: Alignment method ('inner' or 'outer')

    Returns:
        Dictionary with aligned DataFrames
    """
    if not data_dict:
        return {}

    # Get common index
    if method == "inner":
        # Only keep dates present in all datasets
        common_index = data_dict[list(data_dict.keys())[0]].index
        for df in data_dict.values():
            common_index = common_index.intersection(df.index)
    else:
        # Keep all dates, fill missing with NaN
        all_indices = []
        for df in data_dict.values():
            all_indices.append(df.index)
        common_index = all_indices[0].union(*all_indices[1:])

    # Reindex all dataframes
    aligned = {}
    for symbol, df in data_dict.items():
        aligned[symbol] = df.reindex(common_index)

        # Forward fill for outer join
        if method == "outer":
            aligned[symbol] = aligned[symbol].fillna(method='ffill')

    logger.info(
        f"Aligned {len(data_dict)} symbols: "
        f"{len(common_index):,} common bars "
        f"({method} join)"
    )

    return aligned


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)

    print("Testing data loader...")
    print()

    # Test 1: Get available symbols
    print("Available symbols (15min):")
    symbols = get_available_symbols("15min")
    print(f"  {symbols[:10]}... ({len(symbols)} total)")
    print()

    # Test 2: Load ES data
    print("Loading ES 15-min data (2010-2024):")
    es_data = load_data("ES", "15min", "2010-01-01", "2024-12-31")
    print(f"  Shape: {es_data.shape}")
    print(f"  Date range: {es_data.index[0]} to {es_data.index[-1]}")
    print(f"  Columns: {list(es_data.columns)}")
    print()
    print(es_data.head())
    print()

    # Test 3: Load multiple symbols
    print("Loading multiple symbols:")
    multi_data = load_multiple_symbols(
        ["ES", "NQ", "YM", "RTY"],
        timeframe="15min",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    for symbol, df in multi_data.items():
        print(f"  {symbol}: {len(df):,} bars")
