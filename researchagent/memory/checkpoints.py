"""Workflow checkpointing.

LangGraph persists state after every node through a checkpointer. That is what makes a
run resumable and inspectable after a failure, and it is where human-in-the-loop
interrupts will hook in.

``memory`` is process-local and dies with the API. A durable Postgres saver lands with
the persistence work; the registry is the seam that makes it a config change.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from researchagent.config.schemas import CheckpointerKind
from researchagent.core.logging import get_logger
from researchagent.core.registry import Registry

logger = get_logger(__name__)

CheckpointerFactory = Callable[[], BaseCheckpointSaver[str] | None]

CHECKPOINTERS: Registry[CheckpointerFactory] = Registry("checkpointer")

CHECKPOINTERS.add(CheckpointerKind.NONE.value, lambda: None)
CHECKPOINTERS.add(CheckpointerKind.MEMORY.value, InMemorySaver)


def build_checkpointer(kind: CheckpointerKind) -> BaseCheckpointSaver[str] | None:
    """Instantiate the configured checkpointer. ``None`` disables persistence."""
    checkpointer = CHECKPOINTERS.get(kind.value)()
    logger.debug("checkpointer_built", kind=kind.value, enabled=checkpointer is not None)
    return checkpointer
