"""Shared dependencies for FastAPI dependency injection."""

from __future__ import annotations

import redis.asyncio as redis

# Redis client (initialized in main.py on startup)
redis_client: redis.Redis | None = None


def set_redis_client(client: redis.Redis) -> None:
    """Set the Redis client (called from main.py on startup)."""
    global redis_client
    redis_client = client


def get_redis() -> redis.Redis:
    """Get Redis client dependency."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized")
    return redis_client
