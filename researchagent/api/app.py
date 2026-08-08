"""FastAPI application factory.

The API is the outermost layer: it owns the container's lifecycle and translates
errors, and holds no business logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from researchagent.api.errors import register_exception_handlers
from researchagent.api.routes import api_router
from researchagent.container import Container, build_container
from researchagent.core.constants import APP_DESCRIPTION, APP_NAME, APP_VERSION
from researchagent.core.logging import get_logger
from researchagent.core.settings import Settings, get_settings

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    """Build the ASGI app. Tests inject a pre-built container to skip real wiring."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(settings)
        logger.info("api_startup", version=APP_VERSION)
        try:
            yield
        finally:
            await app.state.container.aclose()
            logger.info("api_shutdown")

    resolved = settings or (container.settings if container else get_settings())

    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
