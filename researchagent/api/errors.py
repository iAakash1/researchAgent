"""Translation of domain errors into HTTP responses.

Handlers live at the edge so no inner layer knows what a status code is.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.logging import get_logger

logger = get_logger(__name__)


async def researchagent_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ResearchAgentError)  # noqa: S101 - handler is type-registered
    logger.warning(
        "request_failed",
        path=request.url.path,
        error_code=exc.code,
        error=exc.message,
        context=exc.context,
    )
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("request_crashed", path=request.url.path, error_type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error"}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ResearchAgentError, researchagent_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
