#!/usr/bin/env python3
"""
Extract training data from Strategy Factory vectorized strategies.

This script runs a vectorized strategy and extracts:
1. All indicator values at entry time as features
2. Trade outcomes (exit time, PnL, return, binary label)

Output format matches train_rf_cpcv_bo.py expectations.

Usage:
    # Extract for a specific strategy and symbol from database winners
    python -m research.strategy_factory.ml_integration.extract_training_data \\
        --strategy RSI2_MeanReversion \\
        --symbol ES \\
        --params '{"rsi_length": 2, "rsi_oversold": 10, "rsi_overbought": 65}'

    # Extract using database winner (auto-loads params)
    python -m research.strategy_factory.ml_integration.extract_training_data \\
        --from-db \\
        --strategy-id 21 \\
        --symbol ES

    # Extract all winners from a run
    python -m research.strategy_factory.ml_integration.extract_training_data \\
        --from-winners winners.json \\
        --output-dir data/training/factory
"""

import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.strategy_factory.ml_integration.feature_extractor import (
    FeatureExtractor,
    calculate_additional_features,
    calculate_all_features,
    CROSS_ASSET_SYMBOLS
)
from research.strategy_factory.engine.data_loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Strategy class registry - maps strategy names to classes
STRATEGY_REGISTRY = {}


def register_strategies():
    """Dynamically register all strategy classes."""
    global STRATEGY_REGISTRY

    strategies_path = Path(__file__).parent.parent / 'strategies'
    # Auto-discover all strategy modules in the strategies directory
    strategy_modules = [
        f.stem for f in strategies_path.glob('*.py')
        if f.stem not in ('__init__', 'base') and not f.stem.startswith('_')
    ]

    for module_name in strategy_modules:
        try:
            module = __import__(
                f'research.strategy_factory.strategies.{module_name}',
                fromlist=['*']
            )
            # Find strategy class (class that inherits from BaseStrategy)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    hasattr(attr, 'param_grid') and
                    attr_name != 'BaseStrategy'):
                    STRATEGY_REGISTRY[attr_name] = attr
                    STRATEGY_REGISTRY[attr_name.lower()] = attr
                    logger.debug(f"Registered strategy: {attr_name}")
        except Exception as e:
            logger.debug(f"Could not load {module_name}: {e}")

    logger.info(f"Registered {len(STRATEGY_REGISTRY)//2} strategy classes")


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    if not STRATEGY_REGISTRY:
        register_strategies()

    # Try exact match first
    if strategy_name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_name]

    # Try case-insensitive match
    lower_name = strategy_name.lower().replace('_', '').replace('-', '')
    for name, cls in STRATEGY_REGISTRY.items():
        if name.lower().replace('_', '').replace('-', '') == lower_name:
            return cls

    raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(set(STRATEGY_REGISTRY.values()))}")


def extract_training_data(
    strategy_name: str,
    symbol: str,
    params: Dict[str, Any],
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "15min",
    output_path: Optional[str] = None,
    add_extra_features: bool = True,
    data_dir: str = "data/resampled"
) -> pd.DataFrame:
    """
    Extract training data for a vectorized strategy.

    Args:
        strategy_name: Name of the strategy class
        symbol: Symbol to extract (e.g., 'ES')
        params: Strategy parameters
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Data timeframe (default: 15min)
        output_path: Output CSV path (optional)
        add_extra_features: Whether to add additional ML features
        data_dir: Data directory

    Returns:
        DataFrame with training data in ML-ready format
    """
    logger.info(f"Extracting training data for {strategy_name} on {symbol}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Get strategy class and instantiate
    strategy_class = get_strategy_class(strategy_name)
    strategy = strategy_class(params=params)

    # Load data
    data = load_data(symbol, timeframe, start_date, end_date)
    logger.info(f"Loaded {len(data):,} bars")

    # Add extra features if requested (including cross-asset features)
    if add_extra_features:
        data = calculate_all_features(data, symbol, add_cross_asset=True)
        logger.info("Added additional ML features (including cross-asset)")

    # Extract features and trades
    extractor = FeatureExtractor(
        commission_per_side=1.00,
        slippage_ticks=1.0
    )

    df = extractor.extract(strategy, data, symbol)

    if df.empty:
        logger.warning("No trades extracted!")
        return df

    # Log summary
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Trades: {len(df):,}")
    logger.info(f"Features: {len(df.columns) - 10}")  # Subtract core columns

    if 'y_binary' in df.columns:
        wins = (df['y_binary'] == 1).sum()
        losses = (df['y_binary'] == 0).sum()
        win_rate = wins / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"Win Rate: {win_rate:.1f}% ({wins:,} wins, {losses:,} losses)")

    if 'y_pnl_usd' in df.columns:
        total_pnl = df['y_pnl_usd'].sum()
        avg_pnl = df['y_pnl_usd'].mean()
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Avg PnL/Trade: ${avg_pnl:.2f}")

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

    logger.info("=" * 80)

    return df


def extract_from_winners(
    winners_path: str,
    output_dir: str = "data/training/factory",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "15min"
) -> Dict[str, pd.DataFrame]:
    """
    Extract training data for all winners from a winners.json file.

    Args:
        winners_path: Path to winners.json file (from extract_winners.py)
        output_dir: Output directory for CSV files
        start_date: Start date
        end_date: End date
        timeframe: Data timeframe

    Returns:
        Dict mapping symbol -> DataFrame
    """
    with open(winners_path, 'r') as f:
        winners = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_winners = sum(len(w) for w in winners.values())

    logger.info(f"Extracting training data for {total_winners} winners from {len(winners)} symbols")

    for symbol, symbol_winners in winners.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}: {len(symbol_winners)} winners")
        logger.info(f"{'='*60}")

        for winner in symbol_winners:
            strategy_name = winner['strategy_name']
            params = winner['params']

            try:
                # Create output filename
                param_str = '_'.join(f"{k}{v}" for k, v in sorted(params.items())
                                     if k not in ['stop_loss_atr', 'take_profit_atr', 'max_bars_held'])
                output_file = output_dir / f"{symbol}_{strategy_name}_{param_str}_training.csv"

                # Extract
                df = extract_training_data(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    params=params,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    output_path=str(output_file)
                )

                key = f"{symbol}_{strategy_name}"
                results[key] = df

            except Exception as e:
                logger.error(f"Failed to extract {strategy_name} for {symbol}: {e}")
                continue

    logger.info(f"\nExtracted training data for {len(results)} strategy-symbol combinations")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract training data from Strategy Factory vectorized strategies'
    )

    # Strategy specification
    parser.add_argument('--strategy', type=str, help='Strategy class name (e.g., RSI2MeanReversion)')
    parser.add_argument('--symbol', type=str, help='Symbol to extract (e.g., ES)')
    parser.add_argument('--params', type=str, help='Strategy parameters as JSON or Python dict string')

    # Database/winners source
    parser.add_argument('--from-winners', type=str, help='Path to winners.json file')
    parser.add_argument('--from-db', action='store_true', help='Load from database (requires --strategy-id and --symbol)')
    parser.add_argument('--strategy-id', type=int, help='Strategy ID from database')

    # Date range
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='15min', help='Data timeframe')

    # Output
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--output-dir', type=str, default='data/training/factory', help='Output directory for batch extraction')

    # Options
    parser.add_argument('--no-extra-features', action='store_true', help='Skip adding extra ML features')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')

    args = parser.parse_args()

    try:
        # Mode 1: Extract from winners file
        if args.from_winners:
            results = extract_from_winners(
                winners_path=args.from_winners,
                output_dir=args.output_dir,
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe
            )
            logger.info(f"Extracted {len(results)} training datasets")
            return 0

        # Mode 2: Extract single strategy
        if args.strategy and args.symbol:
            # Parse params
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError:
                    # Try Python literal
                    params = ast.literal_eval(args.params)
            else:
                params = {}

            # Determine output path
            output_path = args.output
            if not output_path:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{args.symbol}_{args.strategy}_training.csv"

            df = extract_training_data(
                strategy_name=args.strategy,
                symbol=args.symbol,
                params=params,
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe,
                output_path=str(output_path),
                add_extra_features=not args.no_extra_features,
                data_dir=args.data_dir
            )

            if df.empty:
                logger.error("No training data extracted!")
                return 1

            return 0

        # No valid mode specified
        parser.print_help()
        return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
