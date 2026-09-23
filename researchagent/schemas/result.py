"""Frontend-facing, evidence-first view of a completed or partial research run."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.reasoning import Citation, FindingStatus, ReviewDecision
from researchagent.schemas.workflow import RunStatus
from researchagent.services.ranking import RelevanceDecision


class PaperResult(BaseModel):
    id: str
    title: str
    provider: str
    year: int | None = None
    doi: str | None = None
    source_url: str | None = None
    pdf_url: str | None = None
    url: str | None = None
    score: float
    relevance_score: float
    relevance_decision: RelevanceDecision
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


class ModelCallResult(BaseModel):
    agent: str | None = None
    provider: str
    model: str
    latency_ms: float
    attempts: int = 1
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


class ModelUsageResult(BaseModel):
    calls: tuple[ModelCallResult, ...] = ()
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


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
    model_usage: ModelUsageResult = Field(default_factory=ModelUsageResult)
    failure: str | None = None
    history: tuple[str, ...] = Field(default_factory=tuple)


class ResearchRunCreated(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.RUNNING


class ResearchRunSnapshot(BaseModel):
    run_id: str
    status: RunStatus
    result: ResearchResult | None = None
    error: str | None = None
