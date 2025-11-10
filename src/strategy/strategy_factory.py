"""
Strategy Factory - Loads and configures trading strategies.

This module provides functions to dynamically load strategy classes
and create their configuration parameters for the strategy worker.
"""

import logging
from typing import Any, Dict, Type, Optional
import backtrader as bt

logger = logging.getLogger(__name__)


def load_strategy(strategy_type: str) -> Type[bt.Strategy]:
    """
    Load a strategy class by name.

    Args:
        strategy_type: Strategy type identifier (e.g., 'ibs', 'breakout')

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy type is not recognized
    """
    # Normalize strategy name
    strategy_type = strategy_type.lower().strip()

    # Map strategy names to their classes
    if strategy_type == 'ibs':
        from src.strategy.ibs_strategy import IbsStrategy
        return IbsStrategy
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def create_strategy_config(
    strategy_name: str,
    symbol: str,
    config: Dict[str, Any],
    portfolio_coordinator: Any,
    ml_model: Optional[Any] = None,
    ml_features: Optional[list] = None,
    ml_threshold: Optional[float] = None,
    callbacks: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create strategy configuration parameters.

    Args:
        strategy_name: Name of the strategy instance
        symbol: Primary trading symbol
        config: Strategy configuration dict from YAML
        portfolio_coordinator: Portfolio coordinator instance
        ml_model: Optional ML model for filtering
        ml_features: Optional list of ML feature names
        ml_threshold: Optional ML probability threshold
        callbacks: Optional dict of callback functions

    Returns:
        Dictionary of strategy parameters to pass to Cerebro.addstrategy()
    """
    # Base parameters that all strategies need
    params = {
        'symbol': symbol,
        'portfolio_coordinator': portfolio_coordinator,
    }

    # Add ML model parameters if provided
    if ml_model is not None:
        params['ml_model'] = ml_model
    if ml_features is not None:
        params['ml_features'] = ml_features
    if ml_threshold is not None:
        params['ml_threshold'] = ml_threshold

    # Add callbacks if provided
    if callbacks:
        params.update(callbacks)

    # Add strategy-specific config parameters
    # These come from the YAML config file
    if config:
        # Add any additional parameters from config
        # Filter out non-strategy params like 'instruments', 'enabled', etc.
        strategy_params = {
            k: v for k, v in config.items()
            if k not in ['instruments', 'enabled', 'strategy_type', 'redis_host',
                        'redis_port', 'broker_account', 'discord_webhook_url',
                        'heartbeat_file', 'heartbeat_interval', 'models_path',
                        'load_historical_warmup', 'historical_lookback_days',
                        'historical_hourly_lookback_days', 'starting_cash',
                        'max_positions', 'daily_stop_loss']
        }
        params.update(strategy_params)

    logger.debug(f"Created strategy config for {strategy_name}/{symbol}: {list(params.keys())}")

    return params
