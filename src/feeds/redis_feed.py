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

    Subscribes to Redis channels: market:{symbol}:{timeframe}
    Receives bars published by the data hub.
    """

    params = (
        ("symbol", None),  # Trading symbol (e.g., 'ES')
        ("timeframe_str", "1min"),  # Timeframe string ('1min', 'hourly', 'daily')
        ("redis_host", "localhost"),
        ("redis_port", 6379),
        ("redis_db", 0),
        ("qcheck", 0.5),  # Queue check interval
        ("warmup_bars", 100),  # Number of bars to load from cache on startup
    )

    def __init__(self, **kwargs):
        # Set Backtrader timeframe from timeframe_str
        timeframe_str = kwargs.get("timeframe_str", "1min")
        if timeframe_str == "1min":
            kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
            kwargs.setdefault("compression", 1)
        elif timeframe_str == "hourly":
            kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
            kwargs.setdefault("compression", 60)
        elif timeframe_str == "daily":
            kwargs.setdefault("timeframe", bt.TimeFrame.Days)
            kwargs.setdefault("compression", 1)

        super().__init__(**kwargs)

        if not self.p.symbol:
            raise ValueError("symbol parameter is required")

        self.channel = f"market:{self.p.symbol}:{self.p.timeframe_str}"

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
            "Initialized RedisLiveData for %s (%s) on channel %s",
            self.p.symbol,
            self.p.timeframe_str,
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
                self.redis_pubsub_client.close()
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
            key = f"market:{self.p.symbol}:{self.p.timeframe_str}:latest"
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

            # Subscribe to channel
            self.pubsub.subscribe(self.channel)
            logger.info("Subscribed to Redis channel: %s", self.channel)

            # Listen for messages
            for message in self.pubsub.listen():
                if self._stop_event.is_set():
                    break

                if message['type'] != 'message':
                    continue  # Skip subscription confirmations

                try:
                    # Parse bar data
                    bar_data = json.loads(message['data'])

                    # Queue the bar
                    try:
                        self._queue.put_nowait(bar_data)
                    except queue.Full:
                        logger.warning(
                            "Queue full for %s, dropping bar",
                            self.p.symbol
                        )

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode bar data: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Redis subscription error: {e}", exc_info=True)
        finally:
            logger.info("Redis subscription loop ended for %s", self.p.symbol)

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
            bars: Iterable of bar dictionaries or Databento records

        Returns:
            Number of bars appended
        """
        appended = 0
        for bar in bars:
            # Convert Databento record to dict if needed
            if hasattr(bar, 'ts_event'):
                # Databento OHLCV record
                bar_data = {
                    'timestamp': dt.datetime.fromtimestamp(bar.ts_event / 1e9, tz=dt.timezone.utc).isoformat(),
                    'open': float(bar.open) / 1e9,  # Databento fixed-point prices
                    'high': float(bar.high) / 1e9,
                    'low': float(bar.low) / 1e9,
                    'close': float(bar.close) / 1e9,
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
                "Buffered %d warmup bars for %s (%s)",
                appended,
                self.p.symbol,
                self.p.timeframe_str
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
    Resampled feed that reads from minute Redis feed and aggregates to hourly/daily.

    This is a simplified version - for now, we'll just subscribe to the
    pre-aggregated hourly/daily channels published by the data hub.

    NOTE: This class delegates to RedisLiveData internally, which is redundant
    since RedisLiveData already supports all timeframes directly. Consider using
    RedisLiveData(timeframe_str='hourly') instead of RedisResampledData.
    This class may be deprecated in future versions.
    """

    params = (
        ("symbol", None),
        ("timeframe_str", "hourly"),  # 'hourly' or 'daily'
        ("redis_host", "localhost"),
        ("redis_port", 6379),
        ("redis_db", 0),
        ("qcheck", 0.5),
    )

    def __init__(self, **kwargs):
        # Set Backtrader timeframe before calling parent
        timeframe_str = kwargs.get("timeframe_str", "hourly")
        if timeframe_str == "hourly":
            kwargs.setdefault("timeframe", bt.TimeFrame.Minutes)
            kwargs.setdefault("compression", 60)
        elif timeframe_str == "daily":
            kwargs.setdefault("timeframe", bt.TimeFrame.Days)
            kwargs.setdefault("compression", 1)

        super().__init__(**kwargs)

        # After super().__init__, params are available as self.p.*
        # Create underlying RedisLiveData feed with params from self.p
        self._redis_feed = RedisLiveData(
            symbol=self.p.symbol,
            timeframe_str=self.p.timeframe_str,
            redis_host=self.p.redis_host,
            redis_port=self.p.redis_port,
            redis_db=self.p.redis_db,
            qcheck=self.p.qcheck
        )

    def start(self):
        """Start the resampled feed."""
        super().start()
        self._redis_feed.start()

    def stop(self):
        """Stop the resampled feed."""
        self._redis_feed.stop()
        super().stop()

    def extend_warmup(self, bars) -> int:
        """Append historical bars to underlying feed."""
        return self._redis_feed.extend_warmup(bars)

    def warmup_backlog_size(self) -> int:
        """Return the number of buffered warmup bars awaiting consumption."""
        return self._redis_feed.warmup_backlog_size()

    def _load(self) -> Optional[bool]:
        """Load next bar by delegating to Redis feed."""
        result = self._redis_feed._load()

        if result is True:
            # Copy data from redis feed to our lines
            self.lines.datetime[0] = self._redis_feed.lines.datetime[0]
            self.lines.open[0] = self._redis_feed.lines.open[0]
            self.lines.high[0] = self._redis_feed.lines.high[0]
            self.lines.low[0] = self._redis_feed.lines.low[0]
            self.lines.close[0] = self._redis_feed.lines.close[0]
            self.lines.volume[0] = self._redis_feed.lines.volume[0]

        return result
