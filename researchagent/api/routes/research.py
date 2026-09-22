"""Research workflow endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchagent.api.dependencies import ResearchRunServiceDep, WorkflowRunnerDep
from researchagent.core.exceptions import RunNotFoundError, WorkflowExecutionError
from researchagent.models.research import ResearchPlan
from researchagent.schemas.result import ResearchRunCreated, ResearchRunSnapshot
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


@router.post("/run", response_model=ResearchRunCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_research_run(
    request: PlanRequest, service: ResearchRunServiceDep
) -> ResearchRunCreated:
    """Create one background research run and return its stable identifier."""
    run_id = service.start(
        request.goal,
        constraints=request.constraints,
        feedback=request.feedback,
        session_id=request.session_id,
    )
    return ResearchRunCreated(run_id=run_id)


@router.get("/results/{run_id}", response_model=ResearchRunSnapshot)
async def get_research_result(run_id: str, service: ResearchRunServiceDep) -> ResearchRunSnapshot:
    """Fetch the current status and optional final result for a run."""
    return service.get(run_id)


@router.get("/runs/{run_id}/stream")
async def stream_research_run(
    run_id: str,
    request: Request,
    service: ResearchRunServiceDep,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    """Attach to an existing run without creating or owning its task."""
    service.get(run_id)
    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None and last_event_id.isdigit():
        after = max(after, int(last_event_id))

    async def events() -> AsyncIterator[str]:
        async for item in service.stream(run_id, after=after):
            yield (
                f"id: {item.sequence}\nevent: progress\ndata: {item.event.model_dump_json()}\n\n"
            )

        finished = service.get(run_id)
        if finished.result is not None:
            yield f"event: result\ndata: {finished.result.model_dump_json()}\n\n"
        elif finished.error is not None:
            payload = json.dumps({"run_id": run_id, "message": finished.error})
            yield f"event: failed\ndata: {payload}\n\n"

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
