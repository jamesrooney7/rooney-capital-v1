"""
Configuration module for multi-alpha architecture.

Handles loading, validation, and management of configuration files.
"""

from .config_loader import load_config, RuntimeConfig
from .strategy_schema import StrategyConfig, validate_strategy_config

# Import legacy config constants from src/config.py module
# (not to be confused with this src/config/ package)
import sys
from pathlib import Path
import importlib.util

# Load src/config.py as a module to avoid package/module name conflict
_config_module_path = Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_legacy_config", _config_module_path)
_legacy_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy_config)

# Re-export constants needed by IbsStrategy
COMMISSION_PER_SIDE = _legacy_config.COMMISSION_PER_SIDE
PAIR_MAP = _legacy_config.PAIR_MAP

__all__ = [
    'load_config',
    'RuntimeConfig',
    'StrategyConfig',
    'validate_strategy_config',
    'COMMISSION_PER_SIDE',
    'PAIR_MAP',
]
