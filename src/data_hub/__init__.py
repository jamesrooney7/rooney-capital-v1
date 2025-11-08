"""
Data Hub module for multi-alpha architecture.

The data hub is responsible for:
- Connecting to Databento live stream
- Aggregating ticks into OHLCV bars
- Publishing bars to Redis pub/sub
- Caching latest bars for strategy worker warmup
"""

from .redis_client import RedisClient
from .data_hub_main import DataHub

__all__ = ['RedisClient', 'DataHub']
