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
from app.dependencies import get_redis, set_redis_client
from app.logging_config import LoggingMiddleware, get_logger, setup_logging

settings = get_settings()

# Initialize structured logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("application_starting", version="0.1.0", env=settings.app_env)

    # Initialize Redis
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_max_connections,
    )
    set_redis_client(redis_client)

    # Test Redis connection
    try:
        await redis_client.ping()  # type: ignore[misc]
        logger.info("redis_connected", url=settings.redis_url.split("@")[-1])
    except Exception as e:
        logger.warning("redis_connection_failed", error=str(e))

    logger.info("application_started")
    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Close Redis
    try:
        redis_client = get_redis()
        await redis_client.close()
        logger.info("redis_disconnected")
    except RuntimeError:
        pass

    # Close database
    await close_db()
    await engine.dispose()
    logger.info("database_disconnected")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Import routers inside function to avoid circular imports
    from app.routers.agent import router as agent_router
    from app.routers.dashboard import router as dashboard_router
    from app.routers.models import router as models_router

    app = FastAPI(
        title="SHS Technician Analytics",
        description="Sears Home Services Technician Recruitment & Retention Analytics Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Logging middleware (must be added first so it wraps all requests)
    app.add_middleware(LoggingMiddleware)

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
    app.include_router(agent_router)

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
        try:
            redis_client = get_redis()
            await redis_client.ping()  # type: ignore[misc]
            checks["redis"] = "connected"
        except RuntimeError:
            checks["redis"] = "not initialized"
        except Exception:
            checks["redis"] = "disconnected"

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
