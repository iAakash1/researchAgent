"""HTTP routers, one module per resource."""

from fastapi import APIRouter

from researchagent.api.routes import health, research

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(research.router)

__all__ = ["api_router"]
