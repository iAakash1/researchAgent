"""Async retry with exponential backoff and full jitter.

Only errors that declare themselves retryable are retried; a schema violation from a
7B model is retryable, a missing config file is not.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.logging import get_logger

logger = get_logger(__name__)


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_seconds: float = Field(default=1.0, ge=0)
    max_delay_seconds: float = Field(default=30.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Delay before ``attempt`` (1-based: attempt 2 is the first retry)."""
        raw = self.initial_delay_seconds * (self.backoff_multiplier ** (attempt - 2))
        capped = min(raw, self.max_delay_seconds)
        return random.uniform(0, capped) if self.jitter else capped  # noqa: S311


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, ResearchAgentError):
        return error.retryable
    return isinstance(error, TimeoutError | ConnectionError)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    operation_name: str,
    on_retry: Callable[[int, BaseException], Awaitable[None]] | None = None,
) -> tuple[T, int]:
    """Run ``operation`` under ``policy``; returns ``(result, attempts_used)``.

    Re-raises the last error once attempts are exhausted or the error is not retryable.
    """
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation(), attempt
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt == policy.max_attempts:
                raise

            delay = policy.delay_for(attempt + 1)
            logger.warning(
                "operation_retry",
                operation=operation_name,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay_seconds=round(delay, 3),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if on_retry is not None:
                await on_retry(attempt, exc)
            await asyncio.sleep(delay)

    raise AssertionError("unreachable") from last_error
