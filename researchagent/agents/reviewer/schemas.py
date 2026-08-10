"""Reviewer agent contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.reasoning import ResearchFinding, ReviewResult, VerificationResult
from researchagent.models.research import ResearchQuestion


class CritiqueDraft(BaseModel):
    """The model's opinion. One input to the decision, never the decision itself."""

    critique: str = ""
    concerns: list[str] = Field(default_factory=list, max_length=8)
    overclaiming_finding_ids: list[str] = Field(default_factory=list, max_length=10)
    recommend_more_evidence: bool = False


class ReviewerInput(BaseModel):
    goal: str = Field(min_length=8)
    questions: tuple[ResearchQuestion, ...] = ()
    findings: tuple[ResearchFinding, ...] = ()
    verifications: tuple[VerificationResult, ...] = ()
    resolved_evidence_ids: frozenset[str] = frozenset()
    iteration: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}


class ReviewerOutput(BaseModel):
    result: ReviewResult
