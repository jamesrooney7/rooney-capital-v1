"""
Data loader utilities for backtesting with Backtrader.

Loads resampled hourly and daily bar data into Backtrader format
with proper naming conventions for IbsStrategy.
"""

import backtrader as bt
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PandasData(bt.feeds.PandasData):
    """
    Extended Pandas data feed for Backtrader.

    Maps CSV columns to Backtrader data lines.
    """
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'volume'),
        ('openinterest', -1),  # Not used
    )


def load_symbol_data(
    symbol: str,
    data_dir: str = 'data/resampled',
    start_date: str = None,
    end_date: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hourly and daily data for a symbol.

    Args:
        symbol: Symbol name (e.g., 'ES', 'NQ')
        data_dir: Directory containing resampled CSV files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    data_path = Path(data_dir)

    hourly_path = data_path / f"{symbol}_hourly.csv"
    daily_path = data_path / f"{symbol}_daily.csv"

    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly data not found: {hourly_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily data not found: {daily_path}")

    # Load hourly data
    hourly_df = pd.read_csv(
        hourly_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # Load daily data
    daily_df = pd.read_csv(
        daily_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        hourly_df = hourly_df[hourly_df.index >= start_dt]
        daily_df = daily_df[daily_df.index >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date)
        hourly_df = hourly_df[hourly_df.index <= end_dt]
        daily_df = daily_df[daily_df.index <= end_dt]

    logger.info(f"Loaded {symbol}: {len(hourly_df):,} hourly bars, {len(daily_df):,} daily bars")

    return hourly_df, daily_df


def add_symbol_to_cerebro(
    cerebro: bt.Cerebro,
    symbol: str,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    start_date: datetime = None,
    end_date: datetime = None
):
    """
    Add hourly and daily data feeds for a symbol to Cerebro.

    IbsStrategy expects data feeds named:
    - {SYMBOL}_hour (e.g., ES_hour)
    - {SYMBOL}_day (e.g., ES_day)

    Args:
        cerebro: Backtrader Cerebro instance
        symbol: Symbol name (e.g., 'ES', 'NQ')
        hourly_df: Hourly bars DataFrame
        daily_df: Daily bars DataFrame
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
    """
    # Create Backtrader data feeds
    hourly_data = PandasData(
        dataname=hourly_df,
        fromdate=start_date,
        todate=end_date,
        name=f"{symbol}_hour"  # IbsStrategy expects this naming
    )

    daily_data = PandasData(
        dataname=daily_df,
        fromdate=start_date,
        todate=end_date,
        name=f"{symbol}_day"  # IbsStrategy expects this naming
    )

    # Add to cerebro
    cerebro.adddata(hourly_data, name=f"{symbol}_hour")
    cerebro.adddata(daily_data, name=f"{symbol}_day")

    logger.info(f"Added {symbol}_hour and {symbol}_day to Cerebro")


def load_all_symbols(
    symbols: list[str],
    data_dir: str = 'data/resampled',
    start_date: str = None,
    end_date: str = None
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load data for multiple symbols.

    Args:
        symbols: List of symbol names
        data_dir: Directory containing resampled CSV files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Dict mapping symbol -> (hourly_df, daily_df)
    """
    data = {}

    for symbol in symbols:
        try:
            hourly_df, daily_df = load_symbol_data(
                symbol,
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date
            )
            data[symbol] = (hourly_df, daily_df)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {symbol}: {e}")

    return data


def setup_cerebro_with_data(
    cerebro: bt.Cerebro,
    symbols: list[str],
    data_dir: str = 'data/resampled',
    start_date: str = None,
    end_date: str = None
):
    """
    Setup Cerebro with data for all specified symbols.

    Args:
        cerebro: Backtrader Cerebro instance
        symbols: List of symbol names to load
        data_dir: Directory containing resampled CSV files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
    """
    # Convert date strings to datetime if provided
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    # Load all symbol data
    all_data = load_all_symbols(symbols, data_dir, start_date, end_date)

    # Add each symbol to cerebro
    for symbol, (hourly_df, daily_df) in all_data.items():
        add_symbol_to_cerebro(
            cerebro,
            symbol,
            hourly_df,
            daily_df,
            start_dt,
            end_dt
        )

    logger.info(f"Loaded {len(all_data)} symbols into Cerebro")
