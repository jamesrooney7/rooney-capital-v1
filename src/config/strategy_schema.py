"""
Strategy configuration schema and validation.

Defines the structure and validation rules for strategy configurations
in the multi-alpha system.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """
    Configuration for a single strategy.

    This dataclass defines all parameters needed to run a strategy worker.
    """

    # Strategy identification
    name: str  # Strategy name (e.g., "ibs", "breakout")
    enabled: bool = True  # Whether strategy is active

    # Broker/execution
    broker_account: str = ""  # TradersPost webhook URL or broker account ID
    starting_cash: float = 150000.0  # Initial capital allocation

    # Strategy-specific
    models_path: str = ""  # Path to ML models directory
    instruments: List[str] = field(default_factory=list)  # Instruments to trade
    strategy_params: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific parameters

    # Risk management
    max_positions: int = 2  # Max concurrent positions
    daily_stop_loss: float = 2500.0  # Daily portfolio stop loss ($)
    max_bars_in_trade: int = 100  # Max time in trade (bars)

    # Data feeds
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Monitoring
    heartbeat_file: str = ""  # Path to heartbeat file
    heartbeat_interval: int = 30  # Heartbeat interval (seconds)

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.heartbeat_file:
            self.heartbeat_file = f"/var/run/pine/{self.name}_worker_heartbeat.json"

        if not self.models_path:
            self.models_path = f"src/models/{self.name}/"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'broker_account': self.broker_account,
            'starting_cash': self.starting_cash,
            'models_path': self.models_path,
            'instruments': self.instruments,
            'strategy_params': self.strategy_params,
            'max_positions': self.max_positions,
            'daily_stop_loss': self.daily_stop_loss,
            'max_bars_in_trade': self.max_bars_in_trade,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'heartbeat_file': self.heartbeat_file,
            'heartbeat_interval': self.heartbeat_interval,
        }


def validate_strategy_config(config_dict: Dict[str, Any]) -> StrategyConfig:
    """
    Validate and parse strategy configuration dictionary.

    Args:
        config_dict: Raw configuration dictionary from YAML

    Returns:
        Validated StrategyConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields
    if 'name' not in config_dict:
        raise ValueError("Strategy config must have 'name' field")

    name = config_dict['name']

    # Validate broker account if enabled
    enabled = config_dict.get('enabled', True)
    if enabled and not config_dict.get('broker_account'):
        logger.warning(
            f"Strategy '{name}' is enabled but has no broker_account configured"
        )

    # Validate instruments
    instruments = config_dict.get('instruments', [])
    if enabled and not instruments:
        logger.warning(
            f"Strategy '{name}' is enabled but has no instruments configured"
        )

    # Validate risk parameters
    max_positions = config_dict.get('max_positions', 2)
    if max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions}")

    daily_stop_loss = config_dict.get('daily_stop_loss', 2500.0)
    if daily_stop_loss <= 0:
        raise ValueError(f"daily_stop_loss must be > 0, got {daily_stop_loss}")

    # Create StrategyConfig object
    try:
        strategy_config = StrategyConfig(
            name=name,
            enabled=enabled,
            broker_account=config_dict.get('broker_account', ''),
            starting_cash=config_dict.get('starting_cash', 150000.0),
            models_path=config_dict.get('models_path', ''),
            instruments=instruments,
            strategy_params=config_dict.get('strategy_params', {}),
            max_positions=max_positions,
            daily_stop_loss=daily_stop_loss,
            max_bars_in_trade=config_dict.get('max_bars_in_trade', 100),
            redis_host=config_dict.get('redis_host', 'localhost'),
            redis_port=config_dict.get('redis_port', 6379),
            heartbeat_file=config_dict.get('heartbeat_file', ''),
            heartbeat_interval=config_dict.get('heartbeat_interval', 30),
        )

        logger.info(f"Validated configuration for strategy '{name}'")
        return strategy_config

    except Exception as e:
        raise ValueError(f"Failed to validate strategy config for '{name}': {e}")


@dataclass
class InstrumentConfig:
    """Configuration for a single trading instrument."""

    symbol: str
    size: int = 1  # Position size (number of contracts)
    commission: float = 4.0  # Commission per side
    multiplier: int = 50  # Contract multiplier
    margin: float = 4000.0  # Margin requirement

    # Databento settings
    databento_product_id: str = ""  # e.g., "ES.FUT"
    databento_dataset: str = "GLBX.MDP3"

    # Tradovate settings
    tradovate_symbol: str = ""  # e.g., "ES"

    def __post_init__(self):
        """Set defaults based on symbol."""
        if not self.databento_product_id:
            self.databento_product_id = f"{self.symbol}.FUT"

        if not self.tradovate_symbol:
            self.tradovate_symbol = self.symbol


def validate_instrument_config(config_dict: Dict[str, Any]) -> InstrumentConfig:
    """
    Validate and parse instrument configuration.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        Validated InstrumentConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    if 'symbol' not in config_dict:
        raise ValueError("Instrument config must have 'symbol' field")

    try:
        return InstrumentConfig(
            symbol=config_dict['symbol'],
            size=config_dict.get('size', 1),
            commission=config_dict.get('commission', 4.0),
            multiplier=config_dict.get('multiplier', 50),
            margin=config_dict.get('margin', 4000.0),
            databento_product_id=config_dict.get('databento_product_id', ''),
            databento_dataset=config_dict.get('databento_dataset', 'GLBX.MDP3'),
            tradovate_symbol=config_dict.get('tradovate_symbol', ''),
        )
    except Exception as e:
        raise ValueError(f"Failed to validate instrument config: {e}")
