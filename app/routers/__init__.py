"""API routers."""

from app.routers.agent import router as agent_router
from app.routers.dashboard import router as dashboard_router
from app.routers.models import router as models_router

__all__ = ["agent_router", "dashboard_router", "models_router"]
