"""Workflow state — the contract passed between LangGraph nodes.

LangGraph merges the partial dict a node returns into this model. Fields annotated with
a reducer accumulate across nodes; every other field is last-write-wins, which is why
nodes must return only what they actually own.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field

from researchagent.models.research import ResearchPlan
from researchagent.services.ranking import ScoredPaper


class WorkflowStage(StrEnum):
    """Pipeline stages that have a node. Grows one member per released stage."""

    PLANNING = "planning"
    DISCOVERY = "discovery"
    DOCUMENT_INTELLIGENCE = "document_intelligence"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    # Prerequisites were not met. Distinct from FAILED: nothing went wrong, the stage
    # simply had nothing valid to work on.
    BLOCKED = "blocked"


class ResearchConstraints(BaseModel):
    """Caller-supplied bounds on the research. All optional; the agent applies defaults."""

    max_research_questions: int | None = Field(default=None, ge=1, le=10)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    focus_areas: list[str] = Field(default_factory=list, max_length=10)
    exclusions: list[str] = Field(default_factory=list, max_length=10)


class StageRecord(BaseModel):
    """Audit trail entry: one stage execution."""

    stage: WorkflowStage
    agent: str
    status: StageStatus
    latency_ms: float
    attempts: int = 1
    note: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StageFailure(BaseModel):
    """Why the run stopped. Recorded in state rather than raised, so a failed run stays
    inspectable and resumable from its checkpoint."""

    stage: WorkflowStage
    agent: str
    code: str
    message: str


class DiscoveryReport(BaseModel):
    """Summary of one discovery pass, including providers that failed.

    Kept in state because a shortlist assembled while two indexes were down is a
    materially different result, and the reviewer must be able to see that.
    """

    sources_queried: list[str] = Field(default_factory=list)
    sources_failed: list[str] = Field(default_factory=list)
    papers_returned: int = 0
    duplicates_removed: int = 0
    candidates: int = 0


class DocumentReport(BaseModel):
    """Summary of the document intelligence stage, including what could not be parsed."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    mean_confidence: float = 0.0
    ready_for_extraction: tuple[str, ...] = ()
    failures: tuple[DocumentFailure, ...] = ()


class DocumentFailure(BaseModel):
    """Why one paper did not become a usable document."""

    paper_id: str
    code: str
    message: str
    remedy: str | None = None


class ResearchState(BaseModel):
    """State threaded through the whole research workflow."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None

    goal: str = Field(min_length=8)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)

    # Reviewer critique fed back into a re-plan; empty on the first pass.
    feedback: list[str] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)

    status: RunStatus = RunStatus.PENDING
    current_stage: WorkflowStage | None = None

    plan: ResearchPlan | None = None
    # Ranked discovery output. Papers themselves live in the repository; state carries
    # the run's shortlist so later stages need no second lookup.
    candidates: list[ScoredPaper] = Field(default_factory=list)
    discovery: DiscoveryReport | None = None
    # Canonical documents are large; state carries the per-paper verdicts and the
    # documents themselves live in the document repository.
    documents: DocumentReport | None = None

    history: Annotated[list[StageRecord], operator.add] = Field(default_factory=list)
    failure: StageFailure | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.COMPLETED and self.failure is None
