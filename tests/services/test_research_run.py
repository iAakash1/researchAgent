from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from researchagent.core.events import EventBus, EventType, StagePayload
from researchagent.schemas.result import ResearchResult
from researchagent.schemas.workflow import ResearchState, RunStatus
from researchagent.services.research_run import ResearchRunService

GOAL = "Study how research streams reconnect safely"


async def test_stream_disconnect_does_not_cancel_run_and_reconnect_replays_progress() -> None:
    event_bus = EventBus()
    started = asyncio.Event()
    release = asyncio.Event()

    class ControlledWorkflow:
        calls = 0

        async def run(self, goal: str, *, run_id: str, **_: object) -> ResearchState:
            self.calls += 1
            await event_bus.emit(
                EventType.WORKFLOW_STARTED,
                StagePayload(stage="planning"),
                run_id=run_id,
            )
            started.set()
            await release.wait()
            return ResearchState(run_id=run_id, goal=goal, status=RunStatus.FAILED)

    class UnusedReasoning:
        async def run(self, state: ResearchState) -> ResearchState:
            raise AssertionError(f"reasoning should not run for {state.run_id}")

    class ResultBuilder:
        async def build(self, state: ResearchState) -> ResearchResult:
            return ResearchResult(
                run_id=state.run_id,
                research_goal=state.goal,
                status=state.status,
                failure="controlled failure",
            )

    workflow = ControlledWorkflow()
    service = ResearchRunService(
        cast(Any, workflow),
        cast(Any, UnusedReasoning()),
        cast(Any, ResultBuilder()),
        event_bus,
    )
    run_id = service.start(GOAL)
    await started.wait()

    first_connection = service.stream(run_id)
    first_event = await anext(first_connection)
    await cast(Any, first_connection).aclose()

    assert first_event.sequence == 1
    assert first_event.event.type is EventType.WORKFLOW_STARTED
    assert service.get(run_id).status is RunStatus.RUNNING
    assert workflow.calls == 1

    reconnected = service.stream(run_id)
    replayed = await anext(reconnected)
    assert replayed.event.event_id == first_event.event.event_id

    release.set()
    with pytest.raises(StopAsyncIteration):
        await anext(reconnected)

    snapshot = service.get(run_id)
    assert snapshot.status is RunStatus.FAILED
    assert snapshot.result is not None
    assert workflow.calls == 1
    await service.aclose()
