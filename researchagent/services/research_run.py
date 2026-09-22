"""The application use case that connects corpus building to audited reasoning."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from researchagent.core.events import Event, EventBus
from researchagent.core.exceptions import RunNotFoundError
from researchagent.core.logging import get_logger
from researchagent.schemas.result import ResearchResult, ResearchRunSnapshot
from researchagent.schemas.workflow import ResearchConstraints, RunStatus
from researchagent.services.result import ResearchResultBuilder
from researchagent.workflows.reasoning_runner import ReasoningRunner
from researchagent.workflows.runner import WorkflowRunner

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SequencedRunEvent:
    sequence: int
    event: Event


class ResearchRunService:
    def __init__(
        self,
        workflow: WorkflowRunner,
        reasoning: ReasoningRunner,
        results: ResearchResultBuilder,
        event_bus: EventBus,
    ) -> None:
        self._workflow = workflow
        self._reasoning = reasoning
        self._results = results
        self._completed: dict[str, ResearchResult] = {}
        self._failures: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._events: dict[str, list[SequencedRunEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[SequencedRunEvent]]] = {}
        self._unsubscribe = event_bus.subscribe(None, self._capture_event)

    def start(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None = None,
        feedback: list[str] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create one owned background run and return its stable identifier."""
        run_id = str(uuid4())
        self._events[run_id] = []
        self._subscribers[run_id] = set()
        self._tasks[run_id] = asyncio.create_task(
            self._execute(
                goal,
                constraints=constraints,
                feedback=feedback,
                run_id=run_id,
                session_id=session_id,
            )
        )
        return run_id

    async def _execute(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None,
        feedback: list[str] | None,
        run_id: str,
        session_id: str | None,
    ) -> None:
        try:
            await self.run(
                goal,
                constraints=constraints,
                feedback=feedback,
                run_id=run_id,
                session_id=session_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failures[run_id] = str(exc)
            logger.exception("research_run_failed", run_id=run_id)

    async def run(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None = None,
        feedback: list[str] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> ResearchResult:
        state = await self._workflow.run(
            goal,
            constraints=constraints,
            feedback=feedback,
            run_id=run_id,
            session_id=session_id,
        )
        if state.succeeded:
            state = await self._reasoning.run(state)
        result = await self._results.build(state)
        self._completed[result.run_id] = result
        self._events.setdefault(result.run_id, [])
        self._subscribers.setdefault(result.run_id, set())
        return result

    def get(self, run_id: str) -> ResearchRunSnapshot:
        if run_id not in self._tasks and run_id not in self._completed:
            raise RunNotFoundError("No research run with this identifier", run_id=run_id)

        result = self._completed.get(run_id)
        if result is not None:
            return ResearchRunSnapshot(run_id=run_id, status=result.status, result=result)

        error = self._failures.get(run_id)
        if error is not None:
            return ResearchRunSnapshot(run_id=run_id, status=RunStatus.FAILED, error=error)

        return ResearchRunSnapshot(run_id=run_id, status=RunStatus.RUNNING)

    async def stream(self, run_id: str, *, after: int = 0) -> AsyncIterator[SequencedRunEvent]:
        """Replay and follow progress without taking ownership of the run task."""
        self.get(run_id)
        queue: asyncio.Queue[SequencedRunEvent] = asyncio.Queue()
        subscribers = self._subscribers[run_id]
        subscribers.add(queue)
        backlog = tuple(self._events[run_id])
        cursor = after
        try:
            for item in backlog:
                if item.sequence > cursor:
                    cursor = item.sequence
                    yield item

            while True:
                if self._is_finished(run_id) and queue.empty():
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                if item.sequence > cursor:
                    cursor = item.sequence
                    yield item
        finally:
            subscribers.discard(queue)

    async def aclose(self) -> None:
        """Stop owned runs during application shutdown."""
        self._unsubscribe()
        pending = [task for task in self._tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _capture_event(self, event: Event) -> None:
        if event.run_id is None or event.run_id not in self._events:
            return
        item = SequencedRunEvent(len(self._events[event.run_id]) + 1, event)
        self._events[event.run_id].append(item)
        for queue in tuple(self._subscribers[event.run_id]):
            queue.put_nowait(item)

    def _is_finished(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return run_id in self._completed or run_id in self._failures or bool(task and task.done())
