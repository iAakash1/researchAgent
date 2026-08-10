"""Retrieval agent contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.models.research import ResearchQuestion


class RetrievalStrategy(StrEnum):
    """How to look. The agent picks; it never names an index or a backend."""

    # One broad query. Cheap, and usually enough for a well-specified question.
    DIRECT = "direct"
    # Several narrower queries decomposing the question. For multi-part questions.
    DECOMPOSED = "decomposed"
    # Follow relationships from named entities. For "which X was used with Y".
    GRAPH_EXPANSION = "graph_expansion"
    # Deliberately hunt for disagreement. For questions about competing claims.
    CONTRADICTION_SEEKING = "contradiction_seeking"


class RetrievalPlanDraft(BaseModel):
    """What the model is asked to decide. Deliberately shallow."""

    strategy: RetrievalStrategy = RetrievalStrategy.DIRECT
    queries: list[str] = Field(default_factory=list, max_length=6)
    kinds: list[str] = Field(default_factory=list, max_length=6)
    entities: list[str] = Field(
        default_factory=list, max_length=6, description="Named methods/datasets to expand from"
    )
    rationale: str = ""


class SufficiencyDraft(BaseModel):
    """The model's judgement on whether what came back is enough."""

    sufficient: bool = False
    missing: list[str] = Field(default_factory=list, max_length=6)
    rationale: str = ""


class RetrievalInput(BaseModel):
    question: ResearchQuestion
    goal: str = Field(min_length=8)
    iteration: int = Field(default=0, ge=0)
    # What a previous round already found, so the agent widens rather than repeats.
    previous_bundle_ids: tuple[str, ...] = ()
    gaps: tuple[str, ...] = Field(
        default=(), description="What verification said was missing, if this is a retry"
    )


class RetrievalDecision(BaseModel):
    """The agent's reasoning about retrieval, recorded for the audit trail."""

    model_config = {"frozen": True}

    strategy: RetrievalStrategy
    queries: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    rationale: str = ""


class RetrievalOutput(BaseModel):
    """Bundles plus the reasoning that produced them."""

    question_id: str
    decision: RetrievalDecision
    bundle_ids: tuple[str, ...] = ()
    # Bundles that failed validation are counted, never used.
    rejected_bundle_ids: tuple[str, ...] = ()
    objects_found: int = 0
    evidence_found: int = 0
    contradictions_found: int = 0
    graph_citations: tuple[str, ...] = ()
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficient: bool = False
    unresolved: tuple[str, ...] = ()
