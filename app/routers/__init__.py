"""API routers."""

from app.routers.dashboard import router as dashboard_router
from app.routers.models import router as models_router

__all__ = ["dashboard_router", "models_router"]
