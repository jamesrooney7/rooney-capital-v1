"""
Redis-based live data feeds for Backtrader.

These feeds subscribe to Redis pub/sub channels published by the data hub
and adapt them to Backtrader's DataBase interface.
"""

import collections
import datetime as dt
import json
import logging
import queue
import threading
from typing import Optional, Any, Dict

import backtrader as bt
import redis

logger = logging.getLogger(__name__)


class RedisLiveData(bt.feeds.DataBase):
    """
    Backtrader live feed powered by Redis pub/sub.

    Subscribes to Redis channels: market:{symbol}:1min
    Receives 1-minute bars published by the data hub.
    """

    params = (
        ("symbol", None),  # Trading symbol (e.g., 'ES')
        ("redis_host", "localhost"),
        ("redis_port", 6379),
        ("redis_db", 0),
        ("qcheck", 0.5),  # Queue check interval
    )

    def __init__(self, **kwargs):
        # Set Backtrader timeframe to 1-minute
        kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
        kwargs.setdefault("compression", 1)

        super().__init__(**kwargs)

        if not self.p.symbol:
            raise ValueError("symbol parameter is required")

        # Subscribe to 1-minute channel only (workers resample to hourly/daily)
        self.channel = f"market:{self.p.symbol}:1min"

        # Redis clients (need separate for pub/sub and regular operations)
        self.redis_pubsub_client: Optional[redis.Redis] = None
        self.redis_cache_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None

        # Queue for receiving bars from Redis thread
        self._queue: queue.Queue = queue.Queue(maxsize=1024)

        # Subscriber thread
        self._sub_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # State
        self._latest_dt: Optional[dt.datetime] = None
        self._stopped = False
        self._warmup_bars: collections.deque = collections.deque()
        self._qcheck: Optional[float] = None  # Will be set based on warmup state

        logger.info(
            "Initialized RedisLiveData for %s on channel %s",
            self.p.symbol,
            self.channel
        )

    def start(self):
        """Start the feed - connect to Redis and begin subscription."""
        super().start()

        # Connect to Redis
        self._connect_redis()

        # Load warmup bars from cache
        self._load_warmup_from_cache()

        # Start subscriber thread
        self._stop_event.clear()
        self._sub_thread = threading.Thread(
            target=self._subscribe_loop,
            name=f"redis-sub-{self.p.symbol}",
            daemon=True
        )
        self._sub_thread.start()

        logger.info("Started RedisLiveData for %s", self.p.symbol)

    def stop(self):
        """Stop the feed - unsubscribe and close connections."""
        logger.info("Stopping RedisLiveData for %s", self.p.symbol)

        self._stopped = True
        self._stop_event.set()

        # Unsubscribe
        if self.pubsub:
            try:
                self.pubsub.unsubscribe(self.channel)
                self.pubsub.close()
            except Exception as e:
                logger.debug(f"Error unsubscribing: {e}")

        # Wait for thread
        if self._sub_thread:
            self._sub_thread.join(timeout=2)

        # Close connections
        if self.redis_pubsub_client:
            try:
                self.redis_pubssub_client.close()
            except Exception as e:
                logger.debug(f"Error closing pubsub client: {e}")

        if self.redis_cache_client:
            try:
                self.redis_cache_client.close()
            except Exception as e:
                logger.debug(f"Error closing cache client: {e}")

        super().stop()

    def _connect_redis(self):
        """Establish Redis connections."""
        try:
            # Client for pub/sub (must be separate)
            self.redis_pubsub_client = redis.Redis(
                host=self.p.redis_host,
                port=self.p.redis_port,
                db=self.p.redis_db,
                decode_responses=True
            )

            # Client for cache reads
            self.redis_cache_client = redis.Redis(
                host=self.p.redis_host,
                port=self.p.redis_port,
                db=self.p.redis_db,
                decode_responses=True
            )

            # Test connection
            self.redis_cache_client.ping()

            logger.info("Connected to Redis at %s:%s", self.p.redis_host, self.p.redis_port)

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _load_warmup_from_cache(self):
        """Load latest cached bar from Redis for warmup."""
        try:
            key = f"market:{self.p.symbol}:1min:latest"
            cached_data = self.redis_cache_client.get(key)

            if cached_data:
                bar_data = json.loads(cached_data)
                self._warmup_bars.append(bar_data)
                logger.info(
                    "Loaded 1 warmup bar for %s from cache",
                    self.p.symbol
                )
        except Exception as e:
            logger.warning(f"Failed to load warmup bar from cache: {e}")

    def _subscribe_loop(self):
        """Background thread that subscribes to Redis and queues bars."""
        try:
            # Create pub/sub object
            self.pubsub = self.redis_pubsub_client.pubsub()

            # Subscribe to both data channel AND control channel
            self.pubsub.subscribe(self.channel, "datahub:control")
            logger.info("Subscribed to Redis channels: %s, datahub:control", self.channel)

            print(f"[DEBUG] {self.p.symbol}: Subscription thread started, entering listen loop", flush=True)

            # Listen for messages
            message_count = 0
            for message in self.pubsub.listen():
                if self._stop_event.is_set():
                    break

                message_count += 1
                if message_count <= 5 or message_count % 100 == 0:
                    print(f"[DEBUG] {self.p.symbol}: Received message #{message_count}, type={message['type']}", flush=True)

                if message['type'] != 'message':
                    continue  # Skip subscription confirmations

                channel = message['channel']

                print(f"[DEBUG] {self.p.symbol}: Processing message from channel {channel}", flush=True)

                try:
                    # Handle control signals
                    if channel == 'datahub:control':
                        self._handle_control_signal(message['data'])
                        continue

                    # Parse bar data from data channel
                    bar_data = json.loads(message['data'])

                    print(f"[DEBUG] {self.p.symbol}: Received bar: {bar_data.get('timestamp', 'unknown')}", flush=True)

                    # Queue the bar
                    try:
                        self._queue.put_nowait(bar_data)
                    except queue.Full:
                        logger.warning(
                            "Queue full for %s, dropping bar",
                            self.p.symbol
                        )

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Redis subscription error: {e}", exc_info=True)
        finally:
            logger.info("Redis subscription loop ended for %s", self.p.symbol)

    def _handle_control_signal(self, signal_data: str):
        """
        Handle control signals from data hub.

        Matches legacy QueueSignal behavior for RESET and SHUTDOWN.

        Args:
            signal_data: JSON-encoded signal
        """
        try:
            signal = json.loads(signal_data)
            signal_type = signal.get('type')
            symbol = signal.get('symbol')

            if signal_type == 'shutdown':
                logger.info("Received SHUTDOWN signal from data hub for %s", self.p.symbol)
                self._stopped = True
                self._stop_event.set()

            elif signal_type == 'reset':
                # Reset signal: clear stale state
                if symbol is None or symbol == self.p.symbol:
                    logger.info("Received RESET signal, clearing feed state for %s", self.p.symbol)
                    self._latest_dt = None  # Allow fresh data

            else:
                logger.debug(f"Ignoring unknown control signal: {signal_type}")

        except Exception as e:
            logger.error(f"Failed to handle control signal: {e}")

    def _load(self) -> Optional[bool]:
        """
        Load next bar into Backtrader lines.

        Returns:
            True if bar loaded, False if stopped, None if no data yet
        """
        if self._stopped:
            return False

        # First drain warmup bars
        if self._warmup_bars:
            self._qcheck = 0.0  # Fast warmup
            bar_data = self._warmup_bars.popleft()
            return self._populate_lines(bar_data)

        # Restore normal qcheck
        if not self._qcheck:
            self._qcheck = self.p.qcheck or 0.5

        # Get bar from queue
        timeout = self._qcheck or 0.5
        try:
            bar_data = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None  # No data yet

        return self._populate_lines(bar_data)

    def extend_warmup(self, bars) -> int:
        """
        Append historical bars to be consumed before live data.

        Args:
            bars: Iterable of Bar objects (from src.runner.databento_bridge)

        Returns:
            Number of bars appended
        """
        appended = 0
        for bar in bars:
            # Convert Bar object to dict for Redis feed
            if hasattr(bar, 'timestamp') and hasattr(bar, 'open'):
                # Bar dataclass from databento_bridge
                bar_data = {
                    'timestamp': bar.timestamp.isoformat() if hasattr(bar.timestamp, 'isoformat') else str(bar.timestamp),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                }
            elif isinstance(bar, dict):
                bar_data = bar
            else:
                logger.debug("Ignoring warmup payload of type %s", type(bar))
                continue

            self._warmup_bars.append(bar_data)
            appended += 1

        if appended:
            logger.info(
                "Buffered %d warmup bars for %s (1min)",
                appended,
                self.p.symbol
            )
        return appended

    def warmup_backlog_size(self) -> int:
        """Return the number of buffered warmup bars awaiting consumption."""
        return len(self._warmup_bars)

    def _populate_lines(self, bar_data: Dict[str, Any]) -> bool:
        """
        Populate Backtrader lines from bar data.

        Args:
            bar_data: Dict with OHLCV data

        Returns:
            True if successful
        """
        try:
            # Parse timestamp
            timestamp_str = bar_data['timestamp']
            timestamp = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            # Check for duplicate/stale bars
            if self._latest_dt and timestamp <= self._latest_dt:
                logger.debug(
                    "Skipping stale bar for %s @ %s",
                    self.p.symbol,
                    timestamp
                )
                return None

            # Populate lines
            self.lines.datetime[0] = bt.date2num(timestamp)
            self.lines.open[0] = float(bar_data['open'])
            self.lines.high[0] = float(bar_data['high'])
            self.lines.low[0] = float(bar_data['low'])
            self.lines.close[0] = float(bar_data['close'])
            self.lines.volume[0] = float(bar_data['volume'])

            self._latest_dt = timestamp

            logger.debug(
                "Loaded %s bar @ %s: OHLCV=%.2f/%.2f/%.2f/%.2f V=%.0f",
                self.p.symbol,
                timestamp,
                bar_data['open'],
                bar_data['high'],
                bar_data['low'],
                bar_data['close'],
                bar_data['volume']
            )

            return True

        except Exception as e:
            logger.error(f"Failed to populate lines from bar data: {e}", exc_info=True)
            return None


class RedisResampledData(bt.feeds.DataBase):
    """
    Hybrid feed that aggregates minute bars into hourly/daily.

    This feed operates in two modes:
    1. Warmup mode: Consumes pre-aggregated hourly/daily bars from warmup queue
    2. Live mode: Aggregates minute bars from source feed into hourly/daily bars

    This matches the legacy ResampledLiveData behavior, allowing us to:
    - Load pre-aggregated hourly/daily bars during warmup (fast)
    - Seamlessly switch to aggregating live minute bars (correct)
    - Preserve full OHLC data for indicators
    """

    params = (
        ("symbol", None),
        ("source_feed", None),  # The minute-resolution feed to aggregate from
        ("session_end_hour", 23),  # Hour when daily bar closes (UTC)
        ("session_end_minute", 0),
        ("bar_interval_minutes", 1440),  # 1440=daily, 60=hourly
        ("qcheck", 0.5),
    )

    def __init__(self, **kwargs):
        # Set Backtrader timeframe based on interval
        interval = kwargs.get("bar_interval_minutes", 1440)
        if interval >= 1440:
            kwargs.setdefault("timeframe", bt.TimeFrame.Days)
            kwargs.setdefault("compression", 1)
        else:
            kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
            kwargs.setdefault("compression", interval)

        super().__init__(**kwargs)

        self._warmup_bars = collections.deque()
        self._current_bar: Optional[Dict[str, Any]] = None
        self._last_bar_timestamp: Optional[dt.datetime] = None
        self._stopped = False

    def start(self):
        """Start the resampled feed."""
        super().start()
        logger.info(
            "Starting RedisResampledData for %s (%d-min bars)",
            self.p.symbol,
            self.p.bar_interval_minutes
        )

    def stop(self):
        """Stop the resampled feed."""
        logger.info("Stopping RedisResampledData for %s", self.p.symbol)
        super().stop()
        self._stopped = True

    def _load(self) -> Optional[bool]:
        """Load next bar (from warmup queue or by aggregating from source)."""
        if self._stopped:
            return False

        # Mode 1: Warmup - drain pre-aggregated bars from queue
        if self._warmup_bars:
            self._qcheck = 0.0  # Fast warmup
            bar_data = self._warmup_bars.popleft()
            return self._populate_lines_from_dict(bar_data)

        # Mode 2: Live - aggregate minute bars from source feed
        if not self._qcheck:
            self._qcheck = self.p.qcheck or 0.5  # Restore normal qcheck

        if self.p.source_feed is None:
            return None  # No source feed configured

        # Try to aggregate minute bars into our timeframe
        return self._aggregate_from_source()

    def _aggregate_from_source(self) -> Optional[bool]:
        """Aggregate minute bars from source feed into hourly/daily bars."""
        source = self.p.source_feed

        # Check if source has data available
        if len(source) == 0:
            return None

        # Get current minute bar from source
        minute_timestamp = bt.num2date(source.datetime[0])

        # Determine if this minute bar belongs to a new aggregation period
        if self._should_start_new_bar(minute_timestamp):
            # Close and emit the previous bar if it exists
            if self._current_bar is not None:
                result = self._populate_lines_from_dict(self._current_bar)
                self._current_bar = None
                self._last_bar_timestamp = minute_timestamp
                return result
            self._last_bar_timestamp = minute_timestamp

        # Aggregate this minute bar into current bar
        self._aggregate_minute_bar(
            timestamp=minute_timestamp,
            open_price=source.open[0],
            high_price=source.high[0],
            low_price=source.low[0],
            close_price=source.close[0],
            volume=source.volume[0]
        )

        # Check if we should close this bar (end of period)
        if self._is_bar_complete(minute_timestamp):
            result = self._populate_lines_from_dict(self._current_bar)
            self._current_bar = None
            return result

        return None  # Bar still building, not ready to emit

    def _should_start_new_bar(self, timestamp: dt.datetime) -> bool:
        """Check if this timestamp starts a new aggregation period."""
        if self._current_bar is None:
            return True  # No current bar, start fresh

        current_start = self._current_bar['timestamp']

        if self.p.bar_interval_minutes >= 1440:
            # Daily bars: new bar if different day
            return timestamp.date() != current_start.date()
        else:
            # Hourly bars: new bar if crossed hour boundary
            hours_diff = (timestamp - current_start).total_seconds() / 3600
            return hours_diff >= (self.p.bar_interval_minutes / 60)

    def _is_bar_complete(self, timestamp: dt.datetime) -> bool:
        """Check if current bar is complete and should be emitted."""
        if self.p.bar_interval_minutes >= 1440:
            # Daily bar: complete at session end (23:00 UTC)
            return (timestamp.hour == self.p.session_end_hour and
                    timestamp.minute == self.p.session_end_minute)
        else:
            # Hourly bar: complete at top of next hour
            return timestamp.minute == 0

    def _aggregate_minute_bar(
        self,
        timestamp: dt.datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float
    ) -> None:
        """Add minute bar data to current aggregated bar."""
        if self._current_bar is None:
            # Start new bar
            self._current_bar = {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
        else:
            # Update existing bar
            self._current_bar['high'] = max(self._current_bar['high'], high_price)
            self._current_bar['low'] = min(self._current_bar['low'], low_price)
            self._current_bar['close'] = close_price
            self._current_bar['volume'] += volume
            self._current_bar['timestamp'] = timestamp  # Use latest timestamp

    def _populate_lines_from_dict(self, bar_dict: Dict[str, Any]) -> bool:
        """Populate feed lines from a dictionary."""
        self.lines.datetime[0] = bt.date2num(bar_dict['timestamp'])
        self.lines.open[0] = bar_dict['open']
        self.lines.high[0] = bar_dict['high']
        self.lines.low[0] = bar_dict['low']
        self.lines.close[0] = bar_dict['close']
        self.lines.volume[0] = bar_dict['volume']
        return True

    def extend_warmup(self, bars) -> int:
        """
        Add pre-aggregated bars for warmup (called before live trading).

        Args:
            bars: Iterable of Bar objects (from databento warmup)

        Returns:
            Number of bars added to warmup queue
        """
        appended = 0
        for bar in bars:
            # Convert Bar object to dict for internal use
            if hasattr(bar, 'timestamp') and hasattr(bar, 'open'):
                bar_dict = {
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                }
                self._warmup_bars.append(bar_dict)
                appended += 1
            else:
                logger.debug("Ignoring warmup payload of type %s", type(bar))

        if appended:
            logger.debug(
                "Buffered %d warmup bars for %s (%d-min)",
                appended,
                self.p.symbol,
                self.p.bar_interval_minutes
            )
        return appended

    def warmup_backlog_size(self) -> int:
        """Return number of buffered warmup bars awaiting consumption."""
        return len(self._warmup_bars)
