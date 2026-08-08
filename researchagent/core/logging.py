"""Structured logging.

Every log line downstream of a workflow run carries ``run_id`` and ``agent`` without
the call sites passing them, via structlog contextvars bound by the agent base class.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from contextvars import Token
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, reset_contextvars

from researchagent.core.settings import LogFormat, Settings

_configured = False


def configure_logging(settings: Settings) -> None:
    """Idempotently configure stdlib logging + structlog for the process."""
    global _configured
    if _configured:
        return

    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format is LogFormat.JSON
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def reset_logging() -> None:
    """Test hook: allow reconfiguration with different settings."""
    global _configured
    _configured = False
    structlog.reset_defaults()
    clear_contextvars()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


class log_context:  # noqa: N801 - used as a context manager, reads like a statement
    """Bind key/values to every log line emitted inside the block."""

    def __init__(self, **values: Any) -> None:
        self._values = values
        self._tokens: Mapping[str, Token[Any]] = {}

    def __enter__(self) -> None:
        # Tokens (not plain unbind) so nested contexts restore the outer value.
        self._tokens = bind_contextvars(**self._values)

    def __exit__(self, *exc_info: object) -> None:
        reset_contextvars(**self._tokens)
