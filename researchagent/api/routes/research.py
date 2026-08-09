"""Research workflow endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchagent.api.dependencies import WorkflowRunnerDep
from researchagent.core.exceptions import RunNotFoundError, WorkflowExecutionError
from researchagent.core.logging import get_logger
from researchagent.models.research import ResearchPlan
from researchagent.schemas.workflow import (
    DiscoveryReport,
    DocumentReport,
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
