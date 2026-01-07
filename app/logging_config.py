"""Structured logging configuration with structlog.

Provides JSON-formatted logging with request ID propagation,
timing metrics, and configurable log levels.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import structlog
from structlog.types import Processor

from app.config import get_settings

settings = get_settings()

# Context variable for request ID propagation
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in context, generating one if not provided."""
    if request_id is None:
        request_id = str(uuid4())
    request_id_var.set(request_id)
    return request_id


def add_request_id(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add request ID to log event."""
    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add application context to log event."""
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""
    # Determine if we should use JSON or console output
    use_json = settings.app_env != "development" or not sys.stdout.isatty()

    # Configure processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_request_id,
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        # JSON output for production
        shared_processors.append(
            structlog.processors.format_exc_info,
        )
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for development
        shared_processors.append(
            structlog.dev.set_exc_info,
        )
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger by name."""
    return structlog.get_logger(name)


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(operation: str) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        operation: Name of the operation being timed

    Usage:
        @log_execution_time("model_training")
        async def train_model():
            ...
    """

    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)

        if asyncio_iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"{operation}_completed",
                        operation=operation,
                        duration_ms=round(elapsed_ms, 2),
                    )
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.error(
                        f"{operation}_failed",
                        operation=operation,
                        duration_ms=round(elapsed_ms, 2),
                        error=str(e),
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"{operation}_completed",
                        operation=operation,
                        duration_ms=round(elapsed_ms, 2),
                    )
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.error(
                        f"{operation}_failed",
                        operation=operation,
                        duration_ms=round(elapsed_ms, 2),
                        error=str(e),
                    )
                    raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def asyncio_iscoroutinefunction(func: Callable[..., Any]) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


class LoggingMiddleware:
    """ASGI middleware for request logging and timing."""

    def __init__(self, app):
        """Initialize middleware.

        Args:
            app: ASGI application
        """
        self.app = app
        self.logger = get_logger("http")

    async def __call__(self, scope, receive, send):
        """Process request and log metrics."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"").decode()

        # Set request ID from header or generate new one
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode() or None
        request_id = set_request_id(request_id)

        # Start timing
        start_time = time.perf_counter()

        # Track response status
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                # Add request ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log request
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.logger.info(
                "http_request",
                method=method,
                path=path,
                query=query_string if query_string else None,
                status=status_code,
                duration_ms=round(elapsed_ms, 2),
            )


# Metrics tracking (in-memory for simplicity; in production use Prometheus or similar)
class MetricsTracker:
    """Simple in-memory metrics tracker."""

    def __init__(self):
        """Initialize metrics storage."""
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}
        self._lock = None  # Would use asyncio.Lock in production

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[metric] = self._counters.get(metric, 0) + value

    def record_timing(self, metric: str, value_ms: float) -> None:
        """Record a timing metric in milliseconds."""
        if metric not in self._timings:
            self._timings[metric] = []
        self._timings[metric].append(value_ms)
        # Keep only last 1000 values
        if len(self._timings[metric]) > 1000:
            self._timings[metric] = self._timings[metric][-1000:]

    def get_counter(self, metric: str) -> int:
        """Get counter value."""
        return self._counters.get(metric, 0)

    def get_timing_stats(self, metric: str) -> Optional[dict[str, float]]:
        """Get timing statistics."""
        values = self._timings.get(metric, [])
        if not values:
            return None
        return {
            "count": len(values),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
            "avg_ms": round(sum(values) / len(values), 2),
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self._counters.copy(),
            "timings": {metric: self.get_timing_stats(metric) for metric in self._timings},
        }


# Global metrics instance
metrics = MetricsTracker()
