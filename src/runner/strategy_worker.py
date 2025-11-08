"""
Strategy Worker - Main entry point for running a single strategy.

This is the core process that runs one strategy independently:
- Loads strategy configuration
- Connects to Redis data hub for market data
- Creates Backtrader Cerebro with Redis feeds
- Loads and runs the strategy
- Routes orders to TradersPost
- Writes heartbeat files for monitoring
- Handles graceful shutdown

Usage:
    python -m src.runner.strategy_worker --strategy ibs --config config.multi_alpha.yml
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import backtrader as bt

# Import configuration
from src.config import load_config, RuntimeConfig, StrategyConfig

# Import feeds
from src.feeds import RedisLiveData, RedisResampledData

# Import strategy factory
from src.strategy.strategy_factory import load_strategy, create_strategy_config

# Import portfolio coordinator (will adapt for multi-alpha)
from src.runner.portfolio_coordinator import PortfolioCoordinator

# Import TradersPost client (will adapt for multi-alpha)
from src.runner.traderspost_client import TradersPostClient

# Import ML model loader
try:
    from src.models.loader import load_model_bundle
except ImportError:
    load_model_bundle = None

logger = logging.getLogger(__name__)


class StrategyWorker:
    """
    Worker process that runs a single strategy.

    Responsibilities:
    - Load strategy configuration
    - Connect to Redis data hub
    - Set up Backtrader with Redis feeds
    - Load ML models
    - Run strategy event loop
    - Handle graceful shutdown
    """

    def __init__(
        self,
        strategy_name: str,
        config: RuntimeConfig,
        log_level: str = "INFO"
    ):
        """
        Initialize strategy worker.

        Args:
            strategy_name: Name of strategy to run (e.g., "ibs")
            config: Complete runtime configuration
            log_level: Logging level
        """
        self.strategy_name = strategy_name
        self.config = config
        self.log_level = log_level

        # Get strategy-specific config
        self.strategy_config = config.get_strategy(strategy_name)
        if not self.strategy_config:
            raise ValueError(f"Strategy '{strategy_name}' not found in config")

        if not self.strategy_config.enabled:
            raise ValueError(f"Strategy '{strategy_name}' is disabled in config")

        # Backtrader instance
        self.cerebro: Optional[bt.Cerebro] = None

        # Components
        self.portfolio_coordinator: Optional[PortfolioCoordinator] = None
        self.traderspost_client: Optional[TradersPostClient] = None

        # ML models per instrument
        self.ml_models: Dict[str, Any] = {}

        # Heartbeat
        self.heartbeat_file = self.strategy_config.heartbeat_file
        self.heartbeat_interval = self.strategy_config.heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # State
        self.start_time: Optional[datetime] = None
        self.running = False

        logger.info(
            f"StrategyWorker initialized for '{strategy_name}' "
            f"(instruments={len(self.strategy_config.instruments)})"
        )

    def setup(self):
        """Set up all components before running."""
        logger.info(f"Setting up strategy worker for '{self.strategy_name}'...")

        # Create portfolio coordinator
        self._setup_portfolio_coordinator()

        # Create TradersPost client
        self._setup_traderspost_client()

        # Load ML models
        self._load_ml_models()

        # Create Backtrader Cerebro
        self._setup_cerebro()

        logger.info("Strategy worker setup complete")

    def _setup_portfolio_coordinator(self):
        """Create portfolio coordinator for this strategy."""
        self.portfolio_coordinator = PortfolioCoordinator(
            max_positions=self.strategy_config.max_positions,
            daily_stop_loss=self.strategy_config.daily_stop_loss,
            emergency_exit_callback=self._on_emergency_exit,
            daily_summary_callback=self._on_daily_summary
        )

        logger.info(
            f"Portfolio coordinator: max_positions={self.strategy_config.max_positions}, "
            f"daily_stop_loss=${self.strategy_config.daily_stop_loss}"
        )

    def _setup_traderspost_client(self):
        """Create TradersPost client for this strategy."""
        webhook_url = self.strategy_config.broker_account

        if not webhook_url:
            logger.warning("No broker_account (webhook) configured - orders will not be sent")

        self.traderspost_client = TradersPostClient(
            webhook_url=webhook_url,
            max_retries=3,
            backoff_factor=0.5,
            timeout=10.0
        )

        logger.info(f"TradersPost client configured: {webhook_url[:50]}..." if webhook_url else "TradersPost client: NO WEBHOOK")

    def _load_ml_models(self):
        """Load ML models for each instrument."""
        if not load_model_bundle:
            logger.warning("ML model loader not available, skipping model loading")
            return

        models_path = Path(self.strategy_config.models_path)

        if not models_path.exists():
            logger.warning(f"Models directory not found: {models_path}")
            return

        for symbol in self.strategy_config.instruments:
            try:
                bundle = load_model_bundle(symbol, base_dir=str(models_path))
                self.ml_models[symbol] = bundle
                logger.info(
                    f"Loaded ML model for {symbol}: "
                    f"{len(bundle.features)} features, threshold={bundle.threshold}"
                )
            except Exception as e:
                logger.warning(f"Failed to load ML model for {symbol}: {e}")

        logger.info(f"Loaded {len(self.ml_models)}/{len(self.strategy_config.instruments)} ML models")

    def _setup_cerebro(self):
        """Create and configure Backtrader Cerebro instance."""
        self.cerebro = bt.Cerebro()

        # Set broker cash
        self.cerebro.broker.setcash(self.strategy_config.starting_cash)

        # Add data feeds for each instrument
        self._add_data_feeds()

        # Add strategy
        self._add_strategy()

        logger.info("Cerebro configured successfully")

    def _add_data_feeds(self):
        """Add Redis data feeds for all instruments."""
        for symbol in self.strategy_config.instruments:
            # Get instrument config
            instr_config = self.config.get_instrument(symbol)
            if not instr_config:
                logger.warning(f"No instrument config for {symbol}, skipping")
                continue

            # Add minute feed
            data_minute = RedisLiveData(
                symbol=symbol,
                timeframe_str='1min',
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name=f"{symbol}_minute"
            )
            self.cerebro.adddata(data_minute, name=f"{symbol}_minute")

            # Add hourly resampled feed
            data_hourly = RedisResampledData(
                symbol=symbol,
                timeframe_str='hourly',
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name=f"{symbol}_hour"
            )
            self.cerebro.adddata(data_hourly, name=f"{symbol}_hour")

            # Add daily resampled feed
            data_daily = RedisResampledData(
                symbol=symbol,
                timeframe_str='daily',
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name=f"{symbol}_day"
            )
            self.cerebro.adddata(data_daily, name=f"{symbol}_day")

            logger.info(f"Added Redis feeds for {symbol} (minute/hourly/daily)")

    def _add_strategy(self):
        """Add strategy to Cerebro."""
        # Load strategy class from factory using strategy_type
        strategy_type = getattr(self.strategy_config, 'strategy_type', self.strategy_name)
        strategy_class = load_strategy(strategy_type)

        # Prepare strategy configuration
        # For now, we'll use the first instrument as the primary symbol
        # In a full implementation, strategies might iterate over all instruments
        primary_symbol = self.strategy_config.instruments[0] if self.strategy_config.instruments else "ES"

        # Get ML model for primary symbol
        ml_bundle = self.ml_models.get(primary_symbol)

        # Create strategy params
        strategy_params = create_strategy_config(
            strategy_name=self.strategy_name,
            symbol=primary_symbol,
            config=self.strategy_config.to_dict(),
            portfolio_coordinator=self.portfolio_coordinator,
            ml_model=ml_bundle.model if ml_bundle else None,
            ml_features=ml_bundle.features if ml_bundle else None,
            ml_threshold=ml_bundle.threshold if ml_bundle else None,
            callbacks={
                'on_order': self._on_order,
                'on_trade': self._on_trade
            }
        )

        # Add strategy to Cerebro
        self.cerebro.addstrategy(strategy_class, **strategy_params)

        logger.info(f"Added strategy '{self.strategy_name}' for {primary_symbol}")

    def _on_order(self, symbol: str, order):
        """Callback when order is placed/executed."""
        # Send to TradersPost
        if self.traderspost_client and order.status in [order.Completed]:
            try:
                payload = {
                    'strategy': self.strategy_name,
                    'symbol': symbol,
                    'action': 'buy' if order.isbuy() else 'sell',
                    'quantity': order.executed.size,
                    'price': order.executed.price,
                    'timestamp': datetime.utcnow().isoformat(),
                }

                self.traderspost_client.post_order(payload)
                logger.info(f"Sent order to TradersPost: {symbol} {payload['action']} {payload['quantity']}")

            except Exception as e:
                logger.error(f"Failed to send order to TradersPost: {e}")

    def _on_trade(self, symbol: str, trade):
        """Callback when trade is closed."""
        # Send to TradersPost
        if self.traderspost_client and trade.isclosed:
            try:
                payload = {
                    'strategy': self.strategy_name,
                    'symbol': symbol,
                    'pnl': trade.pnl,
                    'pnl_pct': (trade.pnl / abs(trade.value) * 100) if trade.value else 0,
                    'timestamp': datetime.utcnow().isoformat(),
                }

                self.traderspost_client.post_trade(payload)
                logger.info(f"Sent trade to TradersPost: {symbol} P&L=${trade.pnl:.2f}")

            except Exception as e:
                logger.error(f"Failed to send trade to TradersPost: {e}")

    def _on_emergency_exit(self, reason: str, context: Dict[str, Any]):
        """Callback when emergency exit triggered (e.g., daily stop loss hit)."""
        logger.critical(
            f"EMERGENCY EXIT triggered for '{self.strategy_name}': {reason} | Context: {context}"
        )

        # Could send alert notification here
        # For now, just log

    def _on_daily_summary(self):
        """Callback when trading day rolls over."""
        logger.info(f"Trading day rollover for '{self.strategy_name}'")

    def run(self):
        """Run the strategy worker."""
        logger.info(f"Starting strategy worker for '{self.strategy_name}'...")

        self.start_time = datetime.utcnow()
        self.running = True

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.strategy_name}",
            daemon=True
        )
        self.heartbeat_thread.start()

        # Run Cerebro (blocks until stopped)
        try:
            logger.info("Running Cerebro event loop...")
            self.cerebro.run()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Cerebro error: {e}", exc_info=True)

        finally:
            self.running = False
            self.stop_event.set()

            # Wait for heartbeat thread
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2)

            logger.info(f"Strategy worker for '{self.strategy_name}' stopped")

    def _heartbeat_loop(self):
        """Background thread that writes heartbeat file periodically."""
        while not self.stop_event.is_set():
            try:
                self._write_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # Sleep in small chunks for quick exit
            for _ in range(self.heartbeat_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

    def _write_heartbeat(self):
        """Write heartbeat status to file."""
        heartbeat_data = {
            'strategy': self.strategy_name,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'running' if self.running else 'stopped',
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'broker_value': self.cerebro.broker.getvalue() if self.cerebro else 0,
            'broker_cash': self.cerebro.broker.getcash() if self.cerebro else 0,
        }

        # Add portfolio coordinator stats
        if self.portfolio_coordinator:
            status = self.portfolio_coordinator.get_status()
            heartbeat_data['portfolio'] = {
                'open_positions': status['open_positions_count'],
                'daily_pnl': status['daily_pnl'],
                'stopped_out': status['stopped_out'],
                'total_entries_allowed': status['stats']['total_entries_allowed'],
                'total_entries_blocked': status['stats']['total_entries_blocked'],
            }

        # Write to file
        heartbeat_path = Path(self.heartbeat_file)
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        with open(heartbeat_path, 'w') as f:
            json.dump(heartbeat_data, f, indent=2)

    def stop(self):
        """Stop the strategy worker gracefully."""
        logger.info(f"Stopping strategy worker for '{self.strategy_name}'...")
        self.running = False
        self.stop_event.set()

        # Stop Cerebro (if running)
        # Note: Cerebro doesn't have a native stop method for live mode
        # In practice, we'd need to signal the strategy to close positions and exit

        logger.info("Strategy worker stopped")


def main():
    """Main entry point for strategy worker."""
    parser = argparse.ArgumentParser(
        description="Strategy Worker - Run a single trading strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Name of strategy to run (e.g., 'ibs', 'breakout')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.multi_alpha.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Create log directory if it doesn't exist
    log_dir = Path("/var/log/rooney")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f'%(asctime)s - [{args.strategy}] - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{args.strategy}_worker.log")
        ]
    )

    logger.info("=" * 80)
    logger.info(f"Starting Strategy Worker: {args.strategy}")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create worker
    try:
        worker = StrategyWorker(
            strategy_name=args.strategy,
            config=config,
            log_level=args.log_level
        )
    except Exception as e:
        logger.error(f"Failed to create strategy worker: {e}")
        sys.exit(1)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup worker
    try:
        worker.setup()
    except Exception as e:
        logger.error(f"Failed to setup strategy worker: {e}", exc_info=True)
        sys.exit(1)

    # Run worker
    logger.info(f"Strategy worker '{args.strategy}' is running. Press Ctrl+C to stop.")
    try:
        worker.run()
    except Exception as e:
        logger.error(f"Strategy worker error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
