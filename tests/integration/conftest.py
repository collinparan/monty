"""Fixtures for integration tests.

This module provides fixtures for:
- Database connections (PostgreSQL + PGVector)
- Redis client
- Mocked Snowflake service
- FastAPI test client
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app, get_redis

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test database URL - use test database
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/shs_test",
)
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create async database engine for tests."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for each test."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    """Create Redis client for tests."""
    client = aioredis.from_url(
        TEST_REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    try:
        await client.ping()
        yield client
    except Exception:
        # If Redis not available, use a mock
        mock_redis = MagicMock(spec=aioredis.Redis)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        mock_redis.delete = AsyncMock()
        mock_redis.ping = AsyncMock()
        yield mock_redis
    finally:
        try:
            await client.flushdb()
            await client.close()
        except Exception:
            pass


@pytest.fixture
def mock_snowflake_data() -> list[dict[str, Any]]:
    """Generate mock technician data from Snowflake."""
    regions = ["US-WEST", "US-EAST", "US-CENTRAL", "US-SOUTH"]
    statuses = ["ACTIVE", "ACTIVE", "ACTIVE", "INACTIVE", "TERMINATED"]

    return [
        {
            "technician_id": str(uuid4()),
            "external_id": f"TECH-{i:04d}",
            "region": regions[i % len(regions)],
            "status": statuses[i % len(statuses)],
            "tenure_days": i * 30 + 10,
            "hire_date": (datetime.utcnow() - timedelta(days=i * 30 + 10)).isoformat(),
            "termination_date": None
            if statuses[i % len(statuses)] != "TERMINATED"
            else datetime.utcnow().isoformat(),
            "avg_jobs_per_week": 5 + (i % 10),
            "avg_rating": 3.5 + (i % 5) * 0.3,
            "total_jobs": 100 + i * 20,
            "completion_rate": 0.85 + (i % 10) * 0.01,
            "first_call_resolution": 0.70 + (i % 15) * 0.02,
        }
        for i in range(50)
    ]


@pytest.fixture
def mock_snowflake_service(mock_snowflake_data):
    """Create mocked Snowflake service."""
    with patch("app.services.snowflake.execute_query") as mock_query:
        mock_query.return_value = mock_snowflake_data

        yield mock_query


@pytest.fixture
def mock_regional_data() -> list[dict[str, Any]]:
    """Generate mock regional metrics from Snowflake."""
    return [
        {
            "region": "US-WEST",
            "technician_count": 120,
            "active_count": 100,
            "avg_tenure_days": 180,
            "avg_rating": 4.2,
            "avg_jobs_per_week": 8.5,
            "churn_rate_30d": 0.05,
        },
        {
            "region": "US-EAST",
            "technician_count": 150,
            "active_count": 130,
            "avg_tenure_days": 200,
            "avg_rating": 4.0,
            "avg_jobs_per_week": 7.8,
            "churn_rate_30d": 0.08,
        },
        {
            "region": "US-CENTRAL",
            "technician_count": 80,
            "active_count": 70,
            "avg_tenure_days": 150,
            "avg_rating": 4.1,
            "avg_jobs_per_week": 9.2,
            "churn_rate_30d": 0.06,
        },
        {
            "region": "US-SOUTH",
            "technician_count": 100,
            "active_count": 85,
            "avg_tenure_days": 170,
            "avg_rating": 3.9,
            "avg_jobs_per_week": 8.0,
            "churn_rate_30d": 0.10,
        },
    ]


@pytest.fixture
def mock_time_series_data() -> list[dict[str, Any]]:
    """Generate mock time series data for Prophet."""
    base_date = datetime(2023, 1, 1)
    return [
        {
            "ds": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "y": 100 + i * 0.5 + (i % 7) * 2,  # Trend + weekly seasonality
        }
        for i in range(365)
    ]


@pytest.fixture
async def test_client(
    db_session, redis_client, mock_snowflake_service
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with mocked dependencies."""

    # Override dependencies
    async def override_get_db():
        yield db_session

    def override_get_redis():
        return redis_client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_training_data():
    """Generate sample DataFrame for ML training."""
    import pandas as pd

    return pd.DataFrame(
        {
            "tenure_days": [30, 60, 90, 120, 150, 180, 210, 240, 270, 300] * 10,
            "avg_jobs_per_week": [5, 6, 7, 8, 9, 10, 11, 12, 8, 7] * 10,
            "avg_rating": [3.5, 4.0, 4.2, 3.8, 4.5, 3.9, 4.1, 4.3, 3.7, 4.0] * 10,
            "completion_rate": [0.80, 0.85, 0.90, 0.82, 0.95, 0.88, 0.91, 0.93, 0.79, 0.87] * 10,
            "churned": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] * 10,
        }
    )


@pytest.fixture
def sample_forecast_data():
    """Generate sample DataFrame for Prophet forecasting."""
    import pandas as pd

    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(365)]
    values = [100 + i * 0.3 + (i % 7) * 2 for i in range(365)]

    return pd.DataFrame({"ds": dates, "y": values})
