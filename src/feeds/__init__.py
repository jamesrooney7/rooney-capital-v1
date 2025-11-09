"""
Feed adapters for multi-alpha architecture.

Provides Backtrader-compatible data feeds that read from Redis pub/sub.
"""

from .redis_feed import RedisLiveData, RedisResampledData

__all__ = ['RedisLiveData', 'RedisResampledData']
