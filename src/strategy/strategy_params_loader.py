"""
Strategy Parameters Loader.

Utility functions for loading optimized strategy parameters from config files.
Used by both backtesting (extract_training_data.py) and live trading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path('config/strategy_params.json')


def load_strategy_params(
    symbol: str,
    config_path: Optional[Path] = None
) -> Dict:
    """
    Load optimized strategy parameters for a symbol.

    Args:
        symbol: Symbol to load parameters for (e.g., 'ES')
        config_path: Optional path to config file (default: config/strategy_params.json)

    Returns:
        Dict with strategy parameters, or empty dict if not found
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # If config doesn't exist, return empty dict (will use defaults)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using default parameters.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Get symbol-specific params
        if symbol not in config:
            logger.warning(f"No optimized parameters found for {symbol}. Using defaults.")
            return {}

        symbol_params = config[symbol]

        # Check if optimized
        if not symbol_params.get('_optimized', False):
            logger.warning(f"Parameters for {symbol} not marked as optimized. Using defaults.")
            return {}

        # Check approval status and warn user
        approved_for_ml = symbol_params.get('_approved_for_ml', True)  # Default True for backwards compatibility
        decision = symbol_params.get('_decision', 'UNKNOWN')

        if not approved_for_ml:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"⚠️  WARNING: Strategy parameters for {symbol} did NOT pass all automated checks")
            logger.warning(f"Decision: {decision}")
            logger.warning(f"Message: {symbol_params.get('_message', 'N/A')}")
            logger.warning(f"")
            logger.warning(f"Metrics:")
            logger.warning(f"  Walk-Forward Sharpe: {symbol_params.get('_walk_forward_sharpe', 'N/A')}")
            logger.warning(f"  Held-Out Sharpe: {symbol_params.get('_heldout_sharpe', 'N/A')}")
            logger.warning(f"  Total OOS Trades: {symbol_params.get('_total_oos_trades', 'N/A')}")
            logger.warning(f"")
            logger.warning(f"You are proceeding at your own discretion. Review results carefully!")
            logger.warning(f"{'='*80}\n")

        # Extract only the parameter values (filter out metadata fields starting with _)
        params = {
            k: v for k, v in symbol_params.items()
            if not k.startswith('_')
        }

        logger.info(f"Loaded optimized parameters for {symbol} (Decision: {decision}):")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")

        return params

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}


def get_strategy_params_for_backtrader(
    symbol: str,
    config_path: Optional[Path] = None
) -> Dict:
    """
    Load strategy parameters formatted for Backtrader IbsStrategy.

    Maps optimized parameter names to Backtrader parameter names.

    Args:
        symbol: Symbol to load parameters for
        config_path: Optional path to config file

    Returns:
        Dict with Backtrader-compatible parameter names
    """
    params = load_strategy_params(symbol, config_path)

    if not params:
        return {}

    # Map from optimized names to Backtrader names
    bt_params = {}

    # IBS entry/exit thresholds
    if 'ibs_entry_low' in params:
        bt_params['ibs_entry_low'] = params['ibs_entry_low']
    if 'ibs_entry_high' in params:
        bt_params['ibs_entry_high'] = params['ibs_entry_high']
    if 'ibs_exit_low' in params:
        bt_params['ibs_exit_low'] = params['ibs_exit_low']
    if 'ibs_exit_high' in params:
        bt_params['ibs_exit_high'] = params['ibs_exit_high']

    # ATR-based stops/targets
    if 'stop_atr_mult' in params:
        bt_params['stop_type'] = 'ATR'  # Switch to ATR-based stops
        bt_params['stop_atr_mult'] = params['stop_atr_mult']
        if 'atr_period' in params:
            bt_params['stop_atr_len'] = int(params['atr_period'])

    if 'target_atr_mult' in params:
        bt_params['tp_type'] = 'ATR'  # Switch to ATR-based targets
        bt_params['tp_atr_mult'] = params['target_atr_mult']
        if 'atr_period' in params:
            bt_params['tp_atr_len'] = int(params['atr_period'])

    # Max holding bars (convert to bar stop)
    if 'max_holding_bars' in params:
        bt_params['enable_bar_stop'] = True
        bt_params['bar_stop_bars'] = int(params['max_holding_bars'])

    # Auto-close hour
    if 'auto_close_hour' in params:
        bt_params['enable_auto_close'] = True
        bt_params['auto_close_time'] = int(params['auto_close_hour']) * 100  # Convert to HHMM format

    return bt_params


def merge_with_defaults(
    symbol: str,
    user_params: Optional[Dict] = None,
    config_path: Optional[Path] = None
) -> Dict:
    """
    Merge optimized parameters with user-provided overrides.

    Priority: user_params > config file > defaults

    Args:
        symbol: Symbol to load parameters for
        user_params: Optional user-provided parameter overrides
        config_path: Optional path to config file

    Returns:
        Merged parameter dict
    """
    # Load from config
    config_params = get_strategy_params_for_backtrader(symbol, config_path)

    # Merge with user params (user params take priority)
    if user_params:
        config_params.update(user_params)

    return config_params
