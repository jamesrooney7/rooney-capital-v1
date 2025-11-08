"""
Configuration module for multi-alpha architecture.

Handles loading, validation, and management of configuration files.
"""

from .config_loader import load_config, RuntimeConfig
from .strategy_schema import StrategyConfig, validate_strategy_config

__all__ = [
    'load_config',
    'RuntimeConfig',
    'StrategyConfig',
    'validate_strategy_config',
]
