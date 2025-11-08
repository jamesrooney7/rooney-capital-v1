"""
Configuration loader for multi-alpha architecture.

Loads and parses YAML configuration files with environment variable expansion.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from .strategy_schema import (
    StrategyConfig,
    InstrumentConfig,
    validate_strategy_config,
    validate_instrument_config
)

logger = logging.getLogger(__name__)


@dataclass
class DataHubConfig:
    """Configuration for the data hub."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    publish_channels: List[str] = field(default_factory=lambda: ["market"])
    heartbeat_file: str = "/var/run/pine/data_hub_heartbeat.json"
    heartbeat_interval: int = 30


@dataclass
class DatabentoConfig:
    """Configuration for Databento data source."""

    api_key: str
    dataset: str = "GLBX.MDP3"
    schema: str = "trades"
    stype_in: str = "product_id"


@dataclass
class RuntimeConfig:
    """
    Complete runtime configuration for multi-alpha system.

    This encompasses all configuration needed to run:
    - Data hub
    - Multiple strategy workers
    - Instrument specifications
    """

    # Databento configuration
    databento: DatabentoConfig

    # Data hub configuration
    data_hub: DataHubConfig

    # Instrument configurations
    instruments: Dict[str, InstrumentConfig]

    # Strategy configurations
    strategies: Dict[str, StrategyConfig]

    # Dashboard configuration
    dashboard_port: int = 5000
    dashboard_strategies: List[str] = field(default_factory=list)

    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name."""
        return self.strategies.get(name)

    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """Get list of enabled strategies."""
        return [s for s in self.strategies.values() if s.enabled]

    def get_instrument(self, symbol: str) -> Optional[InstrumentConfig]:
        """Get instrument configuration by symbol."""
        return self.instruments.get(symbol)


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.

    Supports ${VAR_NAME} syntax.

    Args:
        value: Configuration value (str, dict, list, or other)

    Returns:
        Value with environment variables expanded
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                logger.warning(
                    f"Environment variable '{var_name}' not set, using empty string"
                )
                return ""
            return env_value

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]

    else:
        return value


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    logger.info(f"Loading configuration from {file_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Expand environment variables
    config = expand_env_vars(config)

    return config


def load_config(file_path: str) -> RuntimeConfig:
    """
    Load and validate complete runtime configuration.

    Args:
        file_path: Path to YAML configuration file

    Returns:
        Validated RuntimeConfig object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If configuration is invalid
    """
    # Load raw YAML
    config_dict = load_yaml_file(file_path)

    # Parse Databento config
    databento_dict = config_dict.get('databento', {})
    if 'api_key' not in databento_dict:
        raise ValueError("databento.api_key is required")

    databento_config = DatabentoConfig(
        api_key=databento_dict['api_key'],
        dataset=databento_dict.get('dataset', 'GLBX.MDP3'),
        schema=databento_dict.get('schema', 'trades'),
        stype_in=databento_dict.get('stype_in', 'product_id'),
    )

    # Parse data hub config
    data_hub_dict = config_dict.get('data_hub', {})
    data_hub_config = DataHubConfig(
        redis_host=data_hub_dict.get('redis_host', 'localhost'),
        redis_port=data_hub_dict.get('redis_port', 6379),
        publish_channels=data_hub_dict.get('publish_channels', ['market']),
        heartbeat_file=data_hub_dict.get(
            'heartbeat_file',
            '/var/run/pine/data_hub_heartbeat.json'
        ),
        heartbeat_interval=data_hub_dict.get('heartbeat_interval', 30),
    )

    # Parse instruments
    instruments_dict = config_dict.get('instruments', {})
    instruments = {}

    for symbol, instr_config in instruments_dict.items():
        instr_config['symbol'] = symbol  # Ensure symbol is set
        instruments[symbol] = validate_instrument_config(instr_config)

    logger.info(f"Loaded {len(instruments)} instrument configurations")

    # Parse strategies
    strategies_dict = config_dict.get('strategies', {})
    strategies = {}

    for strategy_name, strategy_config in strategies_dict.items():
        strategy_config['name'] = strategy_name  # Ensure name is set

        # Inherit global redis settings if not specified
        if 'redis_host' not in strategy_config:
            strategy_config['redis_host'] = data_hub_config.redis_host
        if 'redis_port' not in strategy_config:
            strategy_config['redis_port'] = data_hub_config.redis_port

        strategies[strategy_name] = validate_strategy_config(strategy_config)

    logger.info(f"Loaded {len(strategies)} strategy configurations")

    # Parse dashboard config
    dashboard_dict = config_dict.get('dashboard', {})
    dashboard_port = dashboard_dict.get('port', 5000)
    dashboard_strategies = dashboard_dict.get('strategies', list(strategies.keys()))

    # Create RuntimeConfig
    runtime_config = RuntimeConfig(
        databento=databento_config,
        data_hub=data_hub_config,
        instruments=instruments,
        strategies=strategies,
        dashboard_port=dashboard_port,
        dashboard_strategies=dashboard_strategies,
    )

    logger.info("Configuration loaded and validated successfully")

    return runtime_config


def load_config_from_env() -> RuntimeConfig:
    """
    Load configuration from path specified in environment variable.

    Looks for ROONEY_CONFIG environment variable.
    Falls back to config.multi_alpha.yml in current directory.

    Returns:
        Validated RuntimeConfig object
    """
    config_path = os.getenv('ROONEY_CONFIG', 'config.multi_alpha.yml')
    return load_config(config_path)
