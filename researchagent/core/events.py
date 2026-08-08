"""In-process async event bus.

Producers (agents, workflow nodes) publish facts; consumers (metrics, SSE streaming,
persistence) subscribe. Producers never import consumers, so adding observability
never touches agent code.

A subscriber that raises is logged and skipped: observability must not break the run.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from researchagent.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_RETRIED = "agent.retried"
    LLM_CALL_COMPLETED = "llm.call.completed"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"


class Event(BaseModel):
    """Immutable record of something that happened."""

    model_config = {"frozen": True}

    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    source: str | None = None
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


Subscriber = Callable[[Event], Awaitable[None]]


class EventBus:
    """Fan-out bus with per-type and wildcard subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType | None, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: EventType | None, handler: Subscriber) -> Callable[[], None]:
        """Register ``handler``; ``event_type=None`` receives every event.

        Returns an unsubscribe callable.
        """
        self._subscribers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._subscribers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """Deliver to all matching subscribers concurrently, isolating failures."""
        handlers = [*self._subscribers.get(event.type, []), *self._subscribers.get(None, [])]
        if not handlers:
            return

        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "event_subscriber_failed",
                    event_type=event.type,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(result),
                    error_type=type(result).__name__,
                )

    def subscriber_count(self, event_type: EventType | None = None) -> int:
        return len(self._subscribers.get(event_type, []))
