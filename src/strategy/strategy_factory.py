"""
Strategy factory for multi-alpha architecture.

Dynamically loads and instantiates strategies based on configuration.
This allows adding new strategies without modifying core code.
"""

import logging
from typing import Dict, Any, Optional, Type

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# Registry of available strategies
_STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(name: str, strategy_class: Type[BaseStrategy]):
    """
    Register a strategy class with the factory.

    Args:
        name: Strategy name (e.g., "ibs", "breakout")
        strategy_class: Strategy class (must inherit from BaseStrategy)
    """
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError(
            f"Strategy class {strategy_class.__name__} must inherit from BaseStrategy"
        )

    _STRATEGY_REGISTRY[name.lower()] = strategy_class
    logger.info(f"Registered strategy: {name} -> {strategy_class.__name__}")


def get_registered_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Get all registered strategies.

    Returns:
        Dict mapping strategy names to classes
    """
    return dict(_STRATEGY_REGISTRY)


def load_strategy(
    strategy_name: str,
    **kwargs
) -> BaseStrategy:
    """
    Load and instantiate a strategy by name.

    Args:
        strategy_name: Name of strategy (e.g., "ibs", "breakout")
        **kwargs: Strategy parameters to pass to constructor

    Returns:
        Instantiated strategy object

    Raises:
        ValueError: If strategy not found in registry
    """
    strategy_name_lower = strategy_name.lower()

    if strategy_name_lower not in _STRATEGY_REGISTRY:
        available = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available strategies: {available}"
        )

    strategy_class = _STRATEGY_REGISTRY[strategy_name_lower]

    logger.info(
        f"Loading strategy '{strategy_name}' ({strategy_class.__name__}) "
        f"with parameters: {list(kwargs.keys())}"
    )

    # Instantiate strategy
    # Note: Backtrader strategies are instantiated differently when added to Cerebro
    # This function returns the class with params, not an instance
    return strategy_class


def create_strategy_config(
    strategy_name: str,
    symbol: str,
    config: Dict[str, Any],
    portfolio_coordinator=None,
    ml_model=None,
    ml_features=None,
    ml_threshold=None,
    callbacks: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create strategy configuration dict for Backtrader.

    This prepares all parameters needed to add a strategy to Cerebro.

    Args:
        strategy_name: Name of the strategy
        symbol: Trading symbol
        config: Strategy-specific configuration
        portfolio_coordinator: Optional portfolio coordinator
        ml_model: Optional ML model for filtering
        ml_features: Optional list of ML feature names
        ml_threshold: Optional ML probability threshold
        callbacks: Optional dict of callback functions

    Returns:
        Dict of parameters to pass to strategy
    """
    params = {
        'strategy_name': strategy_name,
        'symbol': symbol,
        'portfolio_coordinator': portfolio_coordinator,
        'ml_model': ml_model,
        'ml_features': ml_features,
        'ml_threshold': ml_threshold,
    }

    # Add callbacks if provided
    if callbacks:
        params['on_order_callback'] = callbacks.get('on_order')
        params['on_trade_callback'] = callbacks.get('on_trade')

    # Merge strategy-specific config
    params.update(config.get('strategy_params', {}))

    # Add common parameters
    params['size'] = config.get('size', 1)
    params['max_bars_in_trade'] = config.get('max_bars_in_trade', 100)

    return params


# Auto-registration of built-in strategies
def _register_builtin_strategies():
    """Register built-in strategies on module import."""
    try:
        from .ibs_strategy import IbsStrategy
        register_strategy("ibs", IbsStrategy)
    except ImportError as e:
        logger.warning(f"Could not import IbsStrategy: {e}")

    # Add future strategies here
    # try:
    #     from .breakout_strategy import BreakoutStrategy
    #     register_strategy("breakout", BreakoutStrategy)
    # except ImportError:
    #     pass


# Register built-in strategies when module is imported
_register_builtin_strategies()
