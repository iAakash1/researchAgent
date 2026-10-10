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


async def test_completed_run_records_conclusions_and_survives_a_recording_failure() -> None:
    from researchagent.core.exceptions import RepositoryError
    from researchagent.schemas.reasoning import ReasoningSession

    class Workflow:
        async def run(self, goal: str, *, run_id: str | None = None, **_: object) -> ResearchState:
            return ResearchState(run_id=run_id or "R", goal=goal, status=RunStatus.COMPLETED)

    class Reasoning:
        async def run(self, state: ResearchState) -> ResearchState:
            return state.model_copy(update={"reasoning": ReasoningSession()})

    class ResultBuilder:
        async def build(self, state: ResearchState) -> ResearchResult:
            return ResearchResult(
                run_id=state.run_id, research_goal=state.goal, status=state.status
            )

    class Recorder:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.recorded: list[str] = []

        async def record_run(self, state: ResearchState) -> tuple[()]:
            self.recorded.append(state.run_id)
            if self.fail:
                raise RepositoryError("disk full")
            return ()

    for fail in (False, True):
        recorder = Recorder(fail)
        service = ResearchRunService(
            cast(Any, Workflow()),
            cast(Any, Reasoning()),
            cast(Any, ResultBuilder()),
            EventBus(),
            conclusions=cast(Any, recorder),
        )
        result = await service.run(GOAL, run_id=f"R-{fail}")

        assert recorder.recorded == [f"R-{fail}"]
        assert result.status is RunStatus.COMPLETED
        await service.aclose()
