"""
Redis client wrapper for data hub pub/sub and caching.

This module provides a clean interface for:
- Publishing market data to Redis channels
- Caching latest bars for strategy worker warmup
- Managing connections and error handling
"""

import json
import logging
from typing import Dict, Any, Optional, List
import redis
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Wrapper around Redis client for market data pub/sub.

    Responsibilities:
    - Publish bars to channels (market:{symbol}:{timeframe})
    - Cache latest bars for warmup
    - Handle connection failures gracefully
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        """
        Initialize Redis client.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            decode_responses: Whether to decode responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password

        # Create Redis connection
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Verify connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def publish_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_data: Dict[str, Any]
    ) -> int:
        """
        Publish a bar to Redis pub/sub channel.

        Channel format: market:{symbol}:{timeframe}
        Example: market:ES:1min

        Args:
            symbol: Trading symbol (e.g., 'ES', 'NQ')
            timeframe: Bar timeframe (e.g., '1min', 'hourly', 'daily')
            bar_data: Dict containing OHLCV data and metadata

        Returns:
            Number of subscribers that received the message
        """
        channel = f"market:{symbol}:{timeframe}"

        try:
            # Add timestamp if not present
            if 'published_at' not in bar_data:
                bar_data['published_at'] = datetime.utcnow().isoformat()

            # Serialize to JSON
            message = json.dumps(bar_data)

            # Publish to channel
            num_subscribers = self.client.publish(channel, message)

            logger.debug(
                f"Published {symbol} {timeframe} bar to {num_subscribers} subscribers"
            )

            return num_subscribers

        except Exception as e:
            logger.error(f"Failed to publish bar for {symbol} {timeframe}: {e}")
            raise

    def cache_latest_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_data: Dict[str, Any],
        ttl: int = 86400  # 24 hours default
    ) -> None:
        """
        Cache the latest bar for a symbol/timeframe.

        This allows strategy workers to fetch the latest bar on startup
        without waiting for the next bar to arrive.

        Key format: market:{symbol}:{timeframe}:latest

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            bar_data: Bar data to cache
            ttl: Time-to-live in seconds (default 24 hours)
        """
        key = f"market:{symbol}:{timeframe}:latest"

        try:
            # Serialize to JSON
            value = json.dumps(bar_data)

            # Set with TTL
            self.client.setex(key, ttl, value)

            logger.debug(f"Cached latest bar for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Failed to cache bar for {symbol} {timeframe}: {e}")
            # Don't raise - caching failure shouldn't stop publishing

    def get_latest_bar(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest cached bar for a symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Bar data dict, or None if not cached
        """
        key = f"market:{symbol}:{timeframe}:latest"

        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol} {timeframe}: {e}")
            return None

    def publish_and_cache(
        self,
        symbol: str,
        timeframe: str,
        bar_data: Dict[str, Any]
    ) -> int:
        """
        Publish bar to channel AND cache as latest.

        Convenience method that combines publish_bar() and cache_latest_bar().

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            bar_data: Bar data

        Returns:
            Number of subscribers that received the message
        """
        # Publish first (critical path)
        num_subscribers = self.publish_bar(symbol, timeframe, bar_data)

        # Then cache (non-critical)
        self.cache_latest_bar(symbol, timeframe, bar_data)

        return num_subscribers

    def publish_heartbeat(
        self,
        heartbeat_data: Dict[str, Any]
    ) -> None:
        """
        Publish data hub heartbeat to monitoring channel.

        Args:
            heartbeat_data: Heartbeat status information
        """
        channel = "datahub:heartbeat"

        try:
            message = json.dumps(heartbeat_data)
            self.client.publish(channel, message)

        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}")
            # Don't raise - heartbeat failure shouldn't crash data hub

    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server info.

        Returns:
            Dict with server info (memory usage, connected clients, etc.)
        """
        try:
            info = self.client.info()
            return {
                'redis_version': info.get('redis_version'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'uptime_in_seconds': info.get('uptime_in_seconds'),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}

    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
