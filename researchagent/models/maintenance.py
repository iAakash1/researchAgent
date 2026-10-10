"""Conclusion maintenance (RACM): persisted, versioned conclusions and their diffs.

A *conclusion* is a reviewer-accepted ``ResearchFinding``, persisted so it can outlive the
run that produced it. Each one is resolved down to the evidence it rests on:

    Conclusion -SUPPORTED_BY-> KnowledgeObject -GROUNDED_IN-> SourceLocation -IN-> Paper

which is what lets a later change to a paper (a retraction, an expression of concern, a
new paper that disagrees) be traced mechanically to exactly the conclusions it touches.

History is append-only. A maintenance run never edits a conclusion: it appends a new
version and a :class:`ConclusionChange` naming the observation that caused it. A change
with no trigger cannot be constructed, so "this conclusion changed" is always answerable
with "because of this".
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator

from researchagent.core.evidence import SourceLocation
from researchagent.core.validation import Confidence
from researchagent.models.bundle import Contradiction
from researchagent.models.evidence import EvidenceRole
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus
from researchagent.models.reasoning import ReviewResult, VerificationVerdict
from researchagent.models.research import ResearchPlan


class ConclusionState(StrEnum):
    """Where a conclusion stands after its latest version."""

    ACTIVE = "active"
    WEAKENED = "weakened"
    INVALIDATED = "invalidated"
    NEEDS_REVIEW = "needs_review"
    # A newly found paper disagrees with its support. Both sides are carried.
    CONTRADICTED = "contradicted"


class ChangeType(StrEnum):
    KEPT = "kept"
    WEAKENED = "weakened"
    INVALIDATED = "invalidated"
    NEEDS_REVIEW = "needs_review"
    NEWLY_CONTRADICTED = "newly_contradicted"
    NEW = "new"

    @property
    def resulting_state(self) -> ConclusionState:
        return {
            ChangeType.KEPT: ConclusionState.ACTIVE,
            ChangeType.NEW: ConclusionState.ACTIVE,
            ChangeType.WEAKENED: ConclusionState.WEAKENED,
            ChangeType.INVALIDATED: ConclusionState.INVALIDATED,
            ChangeType.NEEDS_REVIEW: ConclusionState.NEEDS_REVIEW,
            ChangeType.NEWLY_CONTRADICTED: ConclusionState.CONTRADICTED,
        }[self]


class MaintenanceMode(StrEnum):
    # RACM: propagate changes through the dependency graph, re-verify only what is hit.
    INCREMENTAL = "incremental"
    # Baseline: regenerate every conclusion from scratch.
    FULL_RERUN = "full_rerun"


class SupportLink(BaseModel):
    """One piece of evidence a conclusion rests on, resolved to its paper."""

    model_config = {"frozen": True}

    evidence_id: str = Field(min_length=1)
    knowledge_object_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    location: SourceLocation
    bundle_id: str | None = None
    doi: str | None = None
    quote: str = ""
    # FOUNDING: cited by the accepted finding. CORROBORATING: added later because a new
    # paper states the same entity (the cross-paper agreement rule).
    role: EvidenceRole = EvidenceRole.FOUNDING


class ExcludedSupport(BaseModel):
    """Evidence removed from support, and the observation that removed it."""

    model_config = {"frozen": True}

    link: SupportLink
    status: IntegrityStatus
    observation_id: str = Field(min_length=1)


class VerificationSnapshot(BaseModel):
    """The verifier's verdict on a conclusion version, as recorded at the time."""

    model_config = {"frozen": True}

    verdict: VerificationVerdict
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    verified_by: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TriggerKind(StrEnum):
    INTEGRITY = "integrity"
    NEW_PAPER = "new_paper"
    CONTRADICTION = "contradiction"
    REVERIFICATION = "reverification"
    REVIEW = "review"


class ChangeTrigger(BaseModel):
    """The observation that caused a change. Every change carries at least one."""

    model_config = {"frozen": True}

    kind: TriggerKind
    detail: str = Field(min_length=1)
    paper_id: str | None = None
    doi: str | None = None
    registry: str | None = None
    status: IntegrityStatus | None = None
    notice_doi: str | None = None
    notice_date: str | None = None
    observation_id: str | None = None
    contradiction_id: str | None = None

    @classmethod
    def from_observation(cls, observation: IntegrityObservation) -> ChangeTrigger:
        return cls(
            kind=TriggerKind.INTEGRITY,
            detail=observation.observation,
            paper_id=observation.paper_id,
            doi=observation.doi,
            registry=observation.registry,
            status=observation.status,
            notice_doi=observation.notice_doi,
            notice_date=observation.notice_date,
            observation_id=observation.id,
        )


class ConclusionRecord(BaseModel):
    """One version of one conclusion. Never edited; superseded by the next version."""

    model_config = {"frozen": True}

    conclusion_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    previous_version: int | None = None
    topic_id: str = Field(min_length=1)
    # The run that produced this version: the research run for v1, a maintenance run after.
    run_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    state: ConclusionState = ConclusionState.ACTIVE
    change_type: ChangeType | None = None

    supports: tuple[SupportLink, ...] = ()
    excluded: tuple[ExcludedSupport, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    review_flags: tuple[str, ...] = ()
    unresolved_evidence_ids: tuple[str, ...] = ()
    verification: VerificationSnapshot | None = None
    source_finding_id: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _history_is_linear(self) -> ConclusionRecord:
        if self.version == 1 and self.previous_version is not None:
            raise ValueError("version 1 has no predecessor")
        if self.version > 1 and self.previous_version != self.version - 1:
            raise ValueError("a version must follow the one before it")
        if self.state is not ConclusionState.INVALIDATED and not self.supports:
            raise ValueError("only an invalidated conclusion may stand on no evidence")
        return self

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(link.paper_id for link in self.supports))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(link.evidence_id for link in self.supports))

    @property
    def knowledge_object_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(link.knowledge_object_id for link in self.supports))

    @property
    def is_live(self) -> bool:
        return self.state is not ConclusionState.INVALIDATED


class MaintainedTopic(BaseModel):
    """A saved research topic whose conclusions are kept current.

    ``integrity_baseline`` is the last integrity observation this topic's conclusions
    have absorbed, per paper. Changes are measured against it, per topic, so two topics
    sharing a paper each see a retraction exactly once.
    """

    model_config = {"frozen": True}

    topic_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    plan: ResearchPlan
    # Papers with trusted knowledge when the topic was last brought up to date. A paper
    # outside this set is new to the topic.
    known_paper_ids: tuple[str, ...] = ()
    integrity_baseline: dict[str, IntegrityObservation] = Field(default_factory=dict)
    dependency_graph_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConclusionChange(BaseModel):
    """One conclusion's transition in one maintenance run."""

    model_config = {"frozen": True}

    conclusion_id: str = Field(min_length=1)
    statement: str = ""
    before_version: int | None = None
    after_version: int = Field(ge=1)
    change_type: ChangeType
    triggers: tuple[ChangeTrigger, ...] = Field(min_length=1)
    removed_evidence_ids: tuple[str, ...] = ()
    added_evidence_ids: tuple[str, ...] = ()
    verification: VerificationSnapshot | None = None
    contradictions: tuple[Contradiction, ...] = ()
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _versions_agree_with_type(self) -> ConclusionChange:
        if self.change_type is ChangeType.NEW and self.before_version is not None:
            raise ValueError("a NEW conclusion has no previous version")
        if self.change_type is not ChangeType.NEW and self.before_version is None:
            raise ValueError("only a NEW conclusion lacks a previous version")
        return self


class MaintenanceFailure(BaseModel):
    """A step that could not complete. Recorded as state, never raised past the run."""

    model_config = {"frozen": True}

    stage: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    conclusion_id: str | None = None
    paper_id: str | None = None


class MaintenanceUsage(BaseModel):
    """What the run spent, read from the existing per-run model-usage tracking."""

    model_config = {"frozen": True}

    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    # None when any call's cost could not be priced: unknown is not zero.
    estimated_cost_usd: float | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class ConclusionDiff(BaseModel):
    """The typed, auditable result of one maintenance run. An empty diff is valid."""

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: f"D-{uuid4().hex[:12]}")
    topic_id: str = Field(min_length=1)
    maintenance_run_id: str = Field(min_length=1)
    mode: MaintenanceMode
    changes: tuple[ConclusionChange, ...] = ()
    total_conclusions: int = Field(default=0, ge=0)
    affected_conclusions: int = Field(default=0, ge=0)
    papers_checked: int = Field(default=0, ge=0)
    integrity_changes: tuple[IntegrityObservation, ...] = ()
    lookup_failures: tuple[str, ...] = ()
    new_paper_ids: tuple[str, ...] = ()
    # Transitions no rule covers (e.g. a retraction reversed). Reported, never acted on.
    unhandled: tuple[str, ...] = ()
    failures: tuple[MaintenanceFailure, ...] = ()
    review: ReviewResult | None = None
    # full_rerun only: conclusions superseded by regeneration rather than by a rule.
    superseded_conclusion_ids: tuple[str, ...] = ()
    usage: MaintenanceUsage = Field(default_factory=MaintenanceUsage)
    dependency_graph_version: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts(self) -> dict[str, int]:
        """Changes per type, every type present (zero included) so diffs line up."""
        tally = {kind.value: 0 for kind in ChangeType}
        for change in self.changes:
            tally[change.change_type.value] += 1
        return tally

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def change_for(self, conclusion_id: str) -> ConclusionChange | None:
        return next((item for item in self.changes if item.conclusion_id == conclusion_id), None)
