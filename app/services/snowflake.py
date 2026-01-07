"""Snowflake data service for querying technician data."""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis.asyncio as redis

from app.config import get_settings
from app.snowflake import execute_query

settings = get_settings()


class SnowflakeService:
    """Service for querying technician data from Snowflake with caching."""

    def __init__(self, redis_client: redis.Redis | None = None):
        """Initialize service with optional Redis client for caching."""
        self.redis = redis_client
        self.cache_ttl = settings.snowflake_cache_ttl_seconds
        self.schema = f"{settings.snowflake_database}.{settings.snowflake_schema}"

    def _cache_key(self, query: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key from query and parameters."""
        key_data = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        return f"sf:cache:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def _get_cached(self, cache_key: str) -> list[dict[str, Any]] | None:
        """Get cached query result."""
        if not self.redis:
            return None

        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)  # type: ignore[arg-type]
        except Exception:
            pass
        return None

    async def _set_cached(
        self, cache_key: str, data: list[dict[str, Any]], ttl: int | None = None
    ) -> None:
        """Cache query result."""
        if not self.redis:
            return

        with contextlib.suppress(Exception):
            await self.redis.setex(cache_key, ttl or self.cache_ttl, json.dumps(data, default=str))

    async def query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Snowflake query with optional caching."""
        cache_key = self._cache_key(query, params)

        # Try cache first
        if use_cache:
            cached = await self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Execute query
        result = await execute_query(query, params)

        # Cache result
        if use_cache and result:
            await self._set_cached(cache_key, result, cache_ttl)

        return result

    async def get_technician_data(
        self,
        region: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get technician data with optional filters.

        Returns technician records with their metrics and attributes.
        """
        # Build query dynamically based on filters
        conditions = []
        params = {}

        if region:
            conditions.append("region = %(region)s")
            params["region"] = region

        if status:
            conditions.append("status = %(status)s")
            params["status"] = status

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                technician_id,
                external_id,
                region,
                status,
                tenure_days,
                hire_date,
                termination_date,
                avg_jobs_per_week,
                avg_rating,
                completion_rate,
                territory_zip_codes,
                skills,
                certifications,
                equipment_owned,
                created_at,
                updated_at
            FROM {self.schema}.TECHNICIANS
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit}
        """

        return await self.query(query, params if params else None)

    async def get_regional_metrics(
        self,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated metrics by region.

        Returns regional summaries including technician counts, retention rates, etc.
        """
        region_filter = "WHERE region = %(region)s" if region else ""
        params = {"region": region} if region else None

        query = f"""
            SELECT
                region,
                COUNT(*) as total_technicians,
                COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_technicians,
                COUNT(CASE WHEN status = 'CHURNED' THEN 1 END) as churned_technicians,
                AVG(tenure_days) as avg_tenure_days,
                AVG(avg_jobs_per_week) as avg_jobs_per_week,
                AVG(avg_rating) as avg_rating,
                AVG(completion_rate) as avg_completion_rate,
                COUNT(CASE WHEN tenure_days > 180 THEN 1 END) * 100.0 / COUNT(*) as retention_rate_6mo
            FROM {self.schema}.TECHNICIANS
            {region_filter}
            GROUP BY region
            ORDER BY total_technicians DESC
        """

        return await self.query(query, params)

    async def get_historical_trends(
        self,
        metric: str = "headcount",
        region: str | None = None,
        granularity: str = "month",
    ) -> list[dict[str, Any]]:
        """Get historical trends for forecasting.

        Args:
            metric: One of 'headcount', 'hires', 'terminations', 'jobs'
            region: Optional region filter
            granularity: 'day', 'week', or 'month'

        Returns time series data suitable for Prophet forecasting.
        """
        date_trunc = granularity.upper()
        region_filter = "AND region = %(region)s" if region else ""
        params = {"region": region} if region else None

        if metric == "headcount":
            query = f"""
                SELECT
                    DATE_TRUNC('{date_trunc}', snapshot_date) as ds,
                    COUNT(DISTINCT technician_id) as y
                FROM {self.schema}.TECHNICIAN_SNAPSHOTS
                WHERE snapshot_date >= DATEADD(year, -2, CURRENT_DATE())
                {region_filter}
                GROUP BY DATE_TRUNC('{date_trunc}', snapshot_date)
                ORDER BY ds
            """
        elif metric == "hires":
            query = f"""
                SELECT
                    DATE_TRUNC('{date_trunc}', hire_date) as ds,
                    COUNT(*) as y
                FROM {self.schema}.TECHNICIANS
                WHERE hire_date >= DATEADD(year, -2, CURRENT_DATE())
                {region_filter}
                GROUP BY DATE_TRUNC('{date_trunc}', hire_date)
                ORDER BY ds
            """
        elif metric == "terminations":
            query = f"""
                SELECT
                    DATE_TRUNC('{date_trunc}', termination_date) as ds,
                    COUNT(*) as y
                FROM {self.schema}.TECHNICIANS
                WHERE termination_date IS NOT NULL
                AND termination_date >= DATEADD(year, -2, CURRENT_DATE())
                {region_filter}
                GROUP BY DATE_TRUNC('{date_trunc}', termination_date)
                ORDER BY ds
            """
        elif metric == "jobs":
            query = f"""
                SELECT
                    DATE_TRUNC('{date_trunc}', job_date) as ds,
                    COUNT(*) as y
                FROM {self.schema}.JOBS
                WHERE job_date >= DATEADD(year, -2, CURRENT_DATE())
                {region_filter}
                GROUP BY DATE_TRUNC('{date_trunc}', job_date)
                ORDER BY ds
            """
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return await self.query(query, params)

    async def get_technician_features(
        self,
        technician_ids: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Get feature data for ML model training.

        Returns a dataset with all features needed for retention prediction.
        """
        id_filter = ""
        params = None

        if technician_ids:
            placeholders = ", ".join([f"'{tid}'" for tid in technician_ids])
            id_filter = f"WHERE technician_id IN ({placeholders})"

        query = f"""
            SELECT
                t.technician_id,
                t.region,
                t.tenure_days,
                t.avg_jobs_per_week,
                t.avg_rating,
                t.completion_rate,
                t.status,

                -- Territory metrics
                r.avg_job_density,
                r.competition_index,
                r.avg_payout_per_job,
                r.seasonal_variance,

                -- Historical performance
                COALESCE(h.jobs_last_30d, 0) as jobs_last_30d,
                COALESCE(h.jobs_last_90d, 0) as jobs_last_90d,
                COALESCE(h.revenue_last_30d, 0) as revenue_last_30d,
                COALESCE(h.complaints_last_90d, 0) as complaints_last_90d,

                -- Outcome (target variable)
                CASE
                    WHEN t.status = 'CHURNED' THEN 1
                    ELSE 0
                END as churned

            FROM {self.schema}.TECHNICIANS t
            LEFT JOIN {self.schema}.REGIONS r ON t.region = r.region
            LEFT JOIN {self.schema}.TECHNICIAN_HISTORY h ON t.technician_id = h.technician_id
            {id_filter}
            ORDER BY t.created_at DESC
            LIMIT {limit}
        """

        return await self.query(query, params, use_cache=False)

    async def invalidate_cache(self, pattern: str = "*") -> int:
        """Invalidate cached queries matching pattern."""
        if not self.redis:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(f"sf:cache:{pattern}"):
                keys.append(key)

            if keys:
                return await self.redis.delete(*keys)
        except Exception:
            pass
        return 0
