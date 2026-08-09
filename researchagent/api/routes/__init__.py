"""HTTP routers, one module per resource."""

from fastapi import APIRouter

from researchagent.api.routes import evidence, health, library, research

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(research.router)
api_router.include_router(library.router)
api_router.include_router(evidence.router)

__all__ = ["api_router"]
