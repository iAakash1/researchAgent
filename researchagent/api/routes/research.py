"""Research workflow endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import uuid4

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchagent.api.dependencies import ContainerDep, ResearchRunServiceDep, WorkflowRunnerDep
from researchagent.core.events import Event
from researchagent.core.exceptions import RunNotFoundError, WorkflowExecutionError
from researchagent.core.logging import get_logger
from researchagent.models.research import ResearchPlan
from researchagent.schemas.result import ResearchResult
from researchagent.schemas.workflow import (
    AcquisitionReport,
    DiscoveryReport,
    DocumentReport,
    EvidenceReport,
    KnowledgeReport,
    ResearchConstraints,
    ResearchState,
    RunStatus,
    StageFailure,
    StageRecord,
)
from researchagent.services.ranking import ScoredPaper

router = APIRouter(prefix="/research", tags=["research"])
logger = get_logger(__name__)


class PlanRequest(BaseModel):
    goal: str = Field(min_length=8, max_length=2000, examples=["Agentic AI in healthcare"])
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    # Reviewer critique from an earlier run; the Planner is instructed to address it.
    feedback: list[str] = Field(default_factory=list, max_length=20)
    session_id: str | None = None


class PlanResponse(BaseModel):
    run_id: str
    status: RunStatus
    plan: ResearchPlan | None
    candidates: list[ScoredPaper] = Field(default_factory=list)
    discovery: DiscoveryReport | None = None
    documents: DocumentReport | None = None
    knowledge: KnowledgeReport | None = None
    evidence: EvidenceReport | None = None
    acquisition: AcquisitionReport | None = None
    history: list[StageRecord]
    failure: StageFailure | None = None

    @classmethod
    def from_state(cls, state: ResearchState) -> PlanResponse:
        return cls(
            run_id=state.run_id,
            status=state.status,
            plan=state.plan,
            candidates=state.candidates,
            discovery=state.discovery,
            documents=state.documents,
            knowledge=state.knowledge,
            evidence=state.evidence,
            acquisition=state.acquisition,
            history=state.history,
            failure=state.failure,
        )


@router.post("/plan", response_model=PlanResponse, status_code=status.HTTP_200_OK)
async def create_plan(request: PlanRequest, runner: WorkflowRunnerDep) -> PlanResponse:
    """Run the workflow to completion and return the resulting plan."""
    state = await runner.run(
        request.goal,
        constraints=request.constraints,
        feedback=request.feedback,
        session_id=request.session_id,
    )

    if state.failure is not None:
        # The run is checkpointed and inspectable; the HTTP layer decides it is an error.
        raise WorkflowExecutionError(
            "Research workflow failed",
            run_id=state.run_id,
            stage=state.failure.stage.value,
            agent=state.failure.agent,
            cause=state.failure.code,
            detail=state.failure.message,
        )

    return PlanResponse.from_state(state)


@router.post("/run", response_model=ResearchResult)
async def create_research_run(
    request: PlanRequest, service: ResearchRunServiceDep
) -> ResearchResult:
    """Run acquisition through reviewer checks and return the frontend result."""
    return await service.run(
        request.goal,
        constraints=request.constraints,
        feedback=request.feedback,
        session_id=request.session_id,
    )


@router.get("/results/{run_id}", response_model=ResearchResult)
async def get_research_result(run_id: str, service: ResearchRunServiceDep) -> ResearchResult:
    """Fetch a completed structured result while this API process is running."""
    return service.get(run_id)


@router.post("/run/stream")
async def stream_research_run(
    request: PlanRequest,
    service: ResearchRunServiceDep,
    container: ContainerDep,
) -> StreamingResponse:
    """Stream genuine workflow events, followed by the structured final result."""
    run_id = str(uuid4())

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def capture(event: Event) -> None:
            if event.run_id == run_id:
                await queue.put(event)

        unsubscribe = container.event_bus.subscribe(None, capture)
        task = asyncio.create_task(
            service.run(
                request.goal,
                constraints=request.constraints,
                feedback=request.feedback,
                run_id=run_id,
                session_id=request.session_id,
            )
        )
        try:
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                yield f"event: progress\ndata: {event.model_dump_json()}\n\n"

            while not queue.empty():
                event = queue.get_nowait()
                yield f"event: progress\ndata: {event.model_dump_json()}\n\n"

            try:
                result = task.result()
            except Exception as exc:
                logger.exception("research_stream_failed", run_id=run_id)
                payload = json.dumps({"run_id": run_id, "message": str(exc)})
                yield f"event: error\ndata: {payload}\n\n"
                return
            yield f"event: result\ndata: {result.model_dump_json()}\n\n"
        finally:
            unsubscribe()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/plan/stream")
async def stream_plan(request: PlanRequest, runner: WorkflowRunnerDep) -> StreamingResponse:
    """Same run, streamed as server-sent events — one event per completed stage."""

    async def events() -> AsyncIterator[str]:
        async for update in runner.stream(
            request.goal,
            constraints=request.constraints,
            feedback=request.feedback,
            session_id=request.session_id,
        ):
            yield f"event: stage\ndata: {update.model_dump_json()}\n\n"
        yield f"event: done\ndata: {json.dumps({'status': 'finished'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}", response_model=PlanResponse)
async def get_run(run_id: str, runner: WorkflowRunnerDep) -> PlanResponse:
    """Load a previous run from its checkpoint."""
    if not runner.checkpointing_enabled:
        raise RunNotFoundError(
            "Checkpointing is disabled; set checkpointer in config/workflow.yaml",
            run_id=run_id,
        )
    return PlanResponse.from_state(await runner.get_state(run_id))
