"""Frontend-facing, evidence-first view of a completed or partial research run."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.reasoning import Citation, FindingStatus, ReviewDecision
from researchagent.schemas.workflow import RunStatus


class PaperResult(BaseModel):
    id: str
    title: str
    provider: str
    year: int | None = None
    url: str | None = None
    score: float
    selected: bool
    accessible: bool
    processed: bool
    failure_reason: str | None = None


class EvidenceResult(BaseModel):
    id: str
    paper_id: str
    quote: str
    location: str
    stance: str
    bundle_id: str


class ContradictionResult(BaseModel):
    id: str
    description: str
    left_paper_id: str
    right_paper_id: str
    left_quotes: tuple[str, ...] = ()
    right_quotes: tuple[str, ...] = ()


class FindingResult(BaseModel):
    id: str
    statement: str
    status: FindingStatus
    verification_status: str | None = None
    reviewer_accepted: bool = False
    supporting_sources: tuple[str, ...] = ()
    conflicting_sources: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    provenance: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class FailedPaperResult(BaseModel):
    paper_id: str
    reason: str


class ResearchResult(BaseModel):
    run_id: str
    research_goal: str
    status: RunStatus
    progress_stage: str | None = None
    discovered_papers: int = 0
    selected_papers: int = 0
    processed_papers: int = 0
    papers: tuple[PaperResult, ...] = ()
    failed_papers: tuple[FailedPaperResult, ...] = ()
    findings: tuple[FindingResult, ...] = ()
    evidence: tuple[EvidenceResult, ...] = ()
    contradictions: tuple[ContradictionResult, ...] = ()
    reviewer_status: ReviewDecision | None = None
    limitations: tuple[str, ...] = ()
    research_gaps: tuple[str, ...] = ()
    final_summary: str = ""
    failure: str | None = None
    history: tuple[str, ...] = Field(default_factory=tuple)
