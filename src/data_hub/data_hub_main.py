"""
Data Hub Main Process for Multi-Alpha Architecture.

This is the central data distribution process that:
- Connects to Databento live stream
- Aggregates ticks into OHLCV bars
- Publishes bars to Redis pub/sub channels
- Caches latest bars for strategy worker warm-up
- Runs independently from strategy workers

Usage:
    python -m src.data_hub.data_hub_main --config config.multi_alpha.yml
"""

import argparse
import datetime as dt
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any

from databento import Live, SymbolMappingMsg, TradeMsg

from .redis_client import RedisClient
from src.config import load_config, RuntimeConfig

logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Aggregates trade ticks into 1-minute OHLCV bars.

    Responsibilities:
    - Collect ticks and build 1-minute bars
    - Emit completed minute bars for publishing to Redis

    Note: Workers resample minute bars to hourly/daily themselves (matches legacy behavior)
    """

    def __init__(self, on_bar_completed):
        """
        Initialize bar aggregator.

        Args:
            on_bar_completed: Callback function(symbol, timeframe, bar_data)
        """
        self.on_bar_completed = on_bar_completed

        # Track current minute bars being built
        self._minute_bars: Dict[str, Dict[str, Any]] = {}

        # Track last emitted timestamps
        self._last_minute: Dict[str, dt.datetime] = {}

        # Track contract symbols
        self._contract_symbols: Dict[str, str] = {}  # root -> contract symbol

        self._lock = threading.Lock()

    def apply_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        timestamp: dt.datetime,
        instrument_id: Optional[int] = None,
        contract_symbol: Optional[str] = None
    ) -> None:
        """
        Apply a trade tick to build minute bars.

        Args:
            symbol: Root symbol (e.g., 'ES')
            price: Trade price
            size: Trade size
            timestamp: Trade timestamp (UTC)
            instrument_id: Optional Databento instrument ID
            contract_symbol: Optional specific contract (e.g., 'ESZ4')
        """
        with self._lock:
            # Track contract symbol
            if contract_symbol:
                self._contract_symbols[symbol] = contract_symbol

            # Build minute bar
            minute = timestamp.replace(second=0, microsecond=0)
            self._update_bar(
                self._minute_bars,
                symbol,
                minute,
                price,
                size,
                '1min',
                instrument_id,
                contract_symbol
            )

    def _update_bar(
        self,
        bars_dict: Dict[str, Dict[str, Any]],
        symbol: str,
        period: dt.datetime,
        price: float,
        size: float,
        timeframe: str,
        instrument_id: Optional[int],
        contract_symbol: Optional[str]
    ) -> None:
        """Update or create a bar in the given dictionary."""
        key = f"{symbol}_{period.isoformat()}"

        if key not in bars_dict:
            # New bar - check if we should emit the previous one
            self._check_emit_previous(bars_dict, symbol, period, timeframe)

            # Create new bar
            bars_dict[key] = {
                'symbol': symbol,
                'timestamp': period,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size,
                'instrument_id': instrument_id,
                'contract_symbol': contract_symbol or self._contract_symbols.get(symbol),
                'timeframe': timeframe
            }
        else:
            # Update existing bar
            bar = bars_dict[key]
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += size
            if instrument_id:
                bar['instrument_id'] = instrument_id
            if contract_symbol:
                bar['contract_symbol'] = contract_symbol

    def _check_emit_previous(
        self,
        bars_dict: Dict[str, Dict[str, Any]],
        symbol: str,
        new_period: dt.datetime,
        timeframe: str
    ) -> None:
        """Check if we should emit the previous bar for this symbol."""
        # Find previous bar for this symbol
        prev_bar = None
        prev_key = None

        for key, bar in bars_dict.items():
            if bar['symbol'] == symbol and bar['timestamp'] < new_period:
                if prev_bar is None or bar['timestamp'] > prev_bar['timestamp']:
                    prev_bar = bar
                    prev_key = key

        if prev_bar and prev_key:
            # Emit the completed bar
            self._emit_bar(prev_bar)
            # Remove from dict
            del bars_dict[prev_key]

    def _emit_bar(self, bar: Dict[str, Any]) -> None:
        """Emit a completed minute bar via callback."""
        symbol = bar['symbol']
        timeframe = bar['timeframe']

        # Create clean bar data for publishing
        bar_data = {
            'symbol': symbol,
            'timestamp': bar['timestamp'].isoformat(),
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'instrument_id': bar.get('instrument_id'),
            'contract_symbol': bar.get('contract_symbol'),
            'timeframe': timeframe
        }

        # Call the callback to publish
        self.on_bar_completed(symbol, timeframe, bar_data)

        # Update last emitted timestamp
        self._last_minute[symbol] = bar['timestamp']

    def flush(self) -> None:
        """Flush all pending minute bars (called on shutdown)."""
        with self._lock:
            # Emit all pending minute bars
            all_bars = list(self._minute_bars.values())

            # Sort by timestamp
            all_bars.sort(key=lambda b: (b['symbol'], b['timestamp']))

            for bar in all_bars:
                self._emit_bar(bar)

            # Clear all
            self._minute_bars.clear()


class DataHub:
    """
    Central data hub for multi-alpha architecture.

    Connects to Databento, aggregates bars, publishes to Redis.
    """

    def __init__(
        self,
        databento_api_key: str,
        databento_dataset: str,
        product_codes: list,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        heartbeat_interval: int = 30
    ):
        """
        Initialize data hub.

        Args:
            databento_api_key: Databento API key
            databento_dataset: Dataset name (e.g., 'GLBX.MDP3')
            product_codes: List of product IDs to subscribe to
            redis_host: Redis server host
            redis_port: Redis server port
            heartbeat_interval: Heartbeat publish interval (seconds)
        """
        self.databento_api_key = databento_api_key
        self.databento_dataset = databento_dataset
        self.product_codes = product_codes
        self.heartbeat_interval = heartbeat_interval

        # Initialize Redis client
        self.redis_client = RedisClient(host=redis_host, port=redis_port)

        # Initialize bar aggregator
        self.aggregator = BarAggregator(on_bar_completed=self._on_bar_completed)

        # Databento client
        self._databento_client: Optional[Live] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Tracking
        self._instrument_to_symbol: Dict[int, str] = {}  # instrument_id -> root symbol
        self._instrument_to_contract: Dict[int, str] = {}  # instrument_id -> contract symbol
        self._start_time: Optional[dt.datetime] = None
        self._last_trade_time: Optional[dt.datetime] = None
        self._trade_count: int = 0
        self._bar_count: int = 0

        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None

        logger.info("DataHub initialized for dataset=%s products=%s",
                    databento_dataset, product_codes)

    def _on_bar_completed(
        self,
        symbol: str,
        timeframe: str,
        bar_data: Dict[str, Any]
    ) -> None:
        """Callback when a bar is completed - publish to Redis."""
        try:
            # Publish to Redis and get subscriber count
            num_subscribers = self.redis_client.publish_and_cache(symbol, timeframe, bar_data)
            self._bar_count += 1

            # Log subscriber count for monitoring
            if num_subscribers == 0:
                logger.warning(
                    "Published %s %s bar but NO SUBSCRIBERS are listening",
                    symbol,
                    timeframe
                )
            else:
                logger.info(
                    "Published %s %s bar to %d subscriber(s): OHLCV=%.2f/%.2f/%.2f/%.2f V=%.0f",
                    symbol,
                    timeframe,
                    num_subscribers,
                    bar_data['open'],
                    bar_data['high'],
                    bar_data['low'],
                    bar_data['close'],
                    bar_data['volume']
                )

        except Exception as e:
            logger.error(f"Failed to publish bar for {symbol} {timeframe}: {e}")

    def start(self) -> None:
        """Start the data hub."""
        if self._thread and self._thread.is_alive():
            logger.warning("DataHub already running")
            return

        self._stop_event.clear()
        self._start_time = dt.datetime.now(dt.timezone.utc)

        # Start Databento subscriber thread
        self._thread = threading.Thread(
            target=self._run_databento,
            name="databento-subscriber",
            daemon=True
        )
        self._thread.start()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat,
            name="heartbeat",
            daemon=True
        )
        self._heartbeat_thread.start()

        logger.info("DataHub started")

    def stop(self) -> None:
        """Stop the data hub gracefully."""
        logger.info("Stopping DataHub...")
        self._stop_event.set()

        # Notify workers that data hub is shutting down
        try:
            num_workers = self.redis_client.publish_shutdown()
            logger.info(f"Notified {num_workers} workers of shutdown")
        except Exception as e:
            logger.warning(f"Failed to send shutdown signal: {e}")

        # Stop Databento client
        if self._databento_client:
            try:
                self._databento_client.stop()
            except Exception:
                pass

        # Wait for threads
        if self._thread:
            self._thread.join(timeout=5)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)

        # Flush pending bars
        self.aggregator.flush()

        # Close Redis
        self.redis_client.close()

        logger.info("DataHub stopped")

    def _run_databento(self) -> None:
        """Main Databento subscription loop."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_stream()
            except Exception as e:
                logger.error(f"Databento stream error: {e}", exc_info=True)

                if self._stop_event.is_set():
                    break

                # Reconnect after delay
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)

    def _connect_and_stream(self) -> None:
        """Connect to Databento and process messages."""
        logger.info(
            "Connecting to Databento: dataset=%s products=%s",
            self.databento_dataset,
            self.product_codes
        )

        # Create Databento client
        client = Live(
            key=self.databento_api_key,
            reconnect_policy="none"
        )
        self._databento_client = client

        # Subscribe
        client.subscribe(
            dataset=self.databento_dataset,
            schema="trades",
            symbols=self.product_codes,
            stype_in="parent"
        )

        # Process metadata for symbol mappings
        self._process_metadata(client)

        logger.info("Databento connected, starting stream...")

        # Process messages
        try:
            for record in client:
                if self._stop_event.is_set():
                    break

                self._handle_record(record)
        finally:
            try:
                client.stop()
            except Exception:
                pass
            self._databento_client = None

    def _process_metadata(self, client: Live) -> None:
        """Process Databento metadata to extract symbol mappings."""
        try:
            metadata = client.metadata
        except Exception:
            metadata = None

        if not metadata:
            return

        # Process mappings
        for mapping in getattr(metadata, "mappings", []) or []:
            instrument_id = getattr(mapping, "instrument_id", None)
            symbol = getattr(mapping, "symbol", None) or getattr(mapping, "stype_in_symbol", None)
            raw_symbol = getattr(mapping, "raw_symbol", None)

            if instrument_id and symbol:
                # Extract root symbol (strip digits)
                root = "".join(ch for ch in symbol if not ch.isdigit()) or symbol
                self._instrument_to_symbol[instrument_id] = root

                if raw_symbol:
                    self._instrument_to_contract[instrument_id] = raw_symbol

                logger.info(
                    "Metadata mapping: instrument=%s symbol=%s root=%s contract=%s",
                    instrument_id, symbol, root, raw_symbol
                )

    def _handle_record(self, record) -> None:
        """Handle a single Databento record."""
        if isinstance(record, SymbolMappingMsg):
            self._handle_symbol_mapping(record)
        elif isinstance(record, TradeMsg):
            self._handle_trade(record)

    def _handle_symbol_mapping(self, msg: SymbolMappingMsg) -> None:
        """Handle symbol mapping message (contract rolls)."""
        instrument_id = getattr(msg, "instrument_id", None)
        symbol = getattr(msg, "stype_in_symbol", None) or getattr(msg, "stype_out_symbol", None)
        raw_symbol = getattr(msg, "raw_symbol", None)

        if not instrument_id:
            return

        # Check if symbol is valid before processing
        if not symbol:
            logger.warning(f"Symbol mapping message has no symbol for instrument {instrument_id}")
            return

        # Extract root symbol
        root = "".join(ch for ch in symbol if not ch.isdigit()) or symbol

        if root:
            self._instrument_to_symbol[instrument_id] = root

        if raw_symbol:
            self._instrument_to_contract[instrument_id] = raw_symbol

        logger.info(
            "Symbol mapping: instrument=%s symbol=%s root=%s contract=%s",
            instrument_id, symbol, root, raw_symbol
        )

    def _handle_trade(self, trade: TradeMsg) -> None:
        """Handle trade message - aggregate into bars."""
        instrument_id = trade.instrument_id

        # Resolve symbol
        root_symbol = self._instrument_to_symbol.get(instrument_id)
        if not root_symbol:
            logger.debug(f"Unknown instrument {instrument_id}, skipping trade")
            return

        # Get price and size
        price = trade.pretty_price
        size = float(trade.size or 0)

        if price is None:
            return

        try:
            price = float(price)
        except (TypeError, ValueError):
            return

        # Get timestamp
        timestamp = dt.datetime.fromtimestamp(
            trade.ts_event / 1_000_000_000,
            tz=dt.timezone.utc
        )

        # Get contract symbol
        contract_symbol = self._instrument_to_contract.get(instrument_id)

        # Apply trade to aggregator
        self.aggregator.apply_trade(
            symbol=root_symbol,
            price=price,
            size=size,
            timestamp=timestamp,
            instrument_id=instrument_id,
            contract_symbol=contract_symbol
        )

        self._trade_count += 1
        self._last_trade_time = timestamp

    def _run_heartbeat(self) -> None:
        """Publish periodic heartbeat to Redis."""
        while not self._stop_event.is_set():
            try:
                self._publish_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # Sleep in small chunks so we can exit quickly
            for _ in range(self.heartbeat_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _publish_heartbeat(self) -> None:
        """Publish heartbeat status to Redis."""
        uptime = None
        if self._start_time:
            uptime = (dt.datetime.now(dt.timezone.utc) - self._start_time).total_seconds()

        heartbeat_data = {
            'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
            'status': 'running',
            'databento_connected': self._databento_client is not None,
            'dataset': self.databento_dataset,
            'products': self.product_codes,
            'uptime_seconds': uptime,
            'trade_count': self._trade_count,
            'bar_count': self._bar_count,
            'last_trade_time': self._last_trade_time.isoformat() if self._last_trade_time else None,
            'known_instruments': len(self._instrument_to_symbol),
            'redis_info': self.redis_client.get_info()
        }

        self.redis_client.publish_heartbeat(heartbeat_data)


def main():
    """Entry point for data hub process."""
    parser = argparse.ArgumentParser(description="Data Hub for Multi-Alpha Architecture")
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

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info("Starting Data Hub...")

    # Load configuration from YAML file
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Extract product codes from instruments
    product_codes = [
        instr.databento_product_id
        for instr in config.instruments.values()
    ]

    logger.info(f"Data hub will subscribe to {len(product_codes)} instruments")

    # Create data hub from config
    data_hub = DataHub(
        databento_api_key=config.databento.api_key,
        databento_dataset=config.databento.dataset,
        product_codes=product_codes,
        redis_host=config.data_hub.redis_host,
        redis_port=config.data_hub.redis_port,
        heartbeat_interval=config.data_hub.heartbeat_interval
    )

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        data_hub.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start data hub
    data_hub.start()

    # Keep running
    logger.info("Data Hub running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        data_hub.stop()


if __name__ == '__main__':
    main()
