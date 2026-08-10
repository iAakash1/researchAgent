"""Reasoning agent contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import Hypothesis, ResearchFinding
from researchagent.models.research import ResearchQuestion


class ClaimDraft(BaseModel):
    """One claim as the model produces it.

    Citations are plain id strings because that is what a model can reliably emit. They
    are resolved against the actual bundles afterwards — a claim citing an id that does
    not exist keeps the claim and loses the citation, which is what demotes it to a
    hypothesis rather than letting it pass as a finding.
    """

    statement: str
    reasoning: str = ""
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    knowledge_object_ids: list[str] = Field(default_factory=list, max_length=20)
    contradicting_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ReasoningDraft(BaseModel):
    """Phase output: what the evidence supports, and what it merely suggests."""

    claims: list[ClaimDraft] = Field(default_factory=list, max_length=8)
    # Statements the agent believes but cannot support from the bundles.
    open_hypotheses: list[ClaimDraft] = Field(default_factory=list, max_length=5)
    insufficient_evidence: bool = False
    notes: str = ""


class ReasoningInput(BaseModel):
    question: ResearchQuestion
    goal: str = Field(min_length=8)
    bundles: tuple[EvidenceBundle, ...] = ()
    iteration: int = Field(default=0, ge=0)
    # Verification's critique of a previous attempt, when re-reasoning.
    critique: tuple[str, ...] = ()

    model_config = {"arbitrary_types_allowed": True}


class ReasoningOutput(BaseModel):
    """Findings that survived citation resolution, plus what did not."""

    question_id: str
    findings: tuple[ResearchFinding, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    # Claims the model made whose citations resolved to nothing. Counted, never used —
    # this is the agent's own fabrication rate.
    discarded_claims: tuple[str, ...] = ()
    insufficient_evidence: bool = False
    notes: str = ""
