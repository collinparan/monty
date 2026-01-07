"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import close_db, engine
from app.routers import dashboard_router, models_router

settings = get_settings()

# Redis client (initialized on startup)
redis_client: redis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown."""
    global redis_client

    # Startup
    # Initialize Redis
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_max_connections,
    )

    # Test Redis connection
    try:
        await redis_client.ping()  # type: ignore[misc]
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")

    yield

    # Shutdown
    # Close Redis
    if redis_client:
        await redis_client.close()

    # Close database
    await close_db()
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SHS Technician Analytics",
        description="Sears Home Services Technician Recruitment & Retention Analytics Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for dashboard
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Include API routers
    app.include_router(models_router)
    app.include_router(dashboard_router)

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> JSONResponse:
        """Health check endpoint."""
        checks = {
            "status": "healthy",
            "app": settings.app_name,
            "env": settings.app_env,
        }

        # Check Redis
        if redis_client:
            try:
                await redis_client.ping()  # type: ignore[misc]
                checks["redis"] = "connected"
            except Exception:
                checks["redis"] = "disconnected"
        else:
            checks["redis"] = "not initialized"

        return JSONResponse(content=checks)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to API documentation."""
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/docs")

    return app


# Create app instance
app = create_app()


def get_redis() -> redis.Redis:
    """Get Redis client dependency."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized")
    return redis_client
