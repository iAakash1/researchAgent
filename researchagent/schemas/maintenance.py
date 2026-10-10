"""Maintenance workflow contracts: the request and the LangGraph state.

State holds ids, verdicts and decisions — what a maintenance run decided and why — so a run
can be inspected after it ends. Every node returns a partial update of this model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from researchagent.models.bundle import Contradiction
from researchagent.models.graph import KnowledgeGraph
from researchagent.models.integrity import IntegrityChange, IntegrityObservation
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.maintenance import (
    ChangeTrigger,
    ChangeType,
    ConclusionDiff,
    ConclusionRecord,
    ExcludedSupport,
    MaintainedTopic,
    MaintenanceFailure,
    MaintenanceMode,
    SupportLink,
    VerificationSnapshot,
)
from researchagent.models.reasoning import ResearchFinding, ReviewResult, VerificationResult


class PropagationInput(BaseModel):
    """Everything one propagation pass needs, gathered by the workflow."""

    model_config = {"frozen": True}

    records: tuple[ConclusionRecord, ...] = ()
    graph: KnowledgeGraph
    changes: tuple[IntegrityChange, ...] = ()
    # Latest informative observation per paper; decides which support is excluded.
    effective: dict[str, IntegrityObservation] = Field(default_factory=dict)
    new_objects: tuple[KnowledgeObject, ...] = ()
    # The knowledge objects existing conclusions rest on, by id.
    supporting_objects: dict[str, KnowledgeObject] = Field(default_factory=dict)
    dois: dict[str, str | None] = Field(default_factory=dict)


class ReverificationRequest(BaseModel):
    """A conclusion that lost support but still has some: the verifier must look again."""

    model_config = {"frozen": True}

    record: ConclusionRecord
    remaining: tuple[SupportLink, ...] = Field(min_length=1)
    excluded: tuple[ExcludedSupport, ...] = ()
    added: tuple[SupportLink, ...] = ()
    triggers: tuple[ChangeTrigger, ...] = Field(min_length=1)
    # Other rules that fired too; combined with the verdict by precedence.
    also: tuple[ChangeType, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    review_flags: tuple[str, ...] = ()


class ConclusionOutcome(BaseModel):
    """What happens to one conclusion, before it is written."""

    model_config = {"frozen": True}

    record: ConclusionRecord
    change_type: ChangeType
    supports: tuple[SupportLink, ...] = ()
    excluded: tuple[ExcludedSupport, ...] = ()
    added: tuple[SupportLink, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    review_flags: tuple[str, ...] = ()
    triggers: tuple[ChangeTrigger, ...] = Field(min_length=1)
    verification: VerificationSnapshot | None = None
    reasons: tuple[str, ...] = ()


class PropagationPlan(BaseModel):
    model_config = {"frozen": True}

    outcomes: tuple[ConclusionOutcome, ...] = ()
    reverify: tuple[ReverificationRequest, ...] = ()
    affected_ids: tuple[str, ...] = ()
    unhandled: tuple[str, ...] = ()


class MaintenanceRequest(BaseModel):
    """What the caller asks for. Anything left unset comes from config/maintenance.yaml."""

    mode: MaintenanceMode | None = None
    # Ignore cached observations and ask the registry again.
    force_refresh: bool = False
    # Papers to treat as newly accepted. Empty means detect them (if configured).
    new_paper_ids: tuple[str, ...] = ()


class MaintenanceState(BaseModel):
    run_id: str = Field(default_factory=lambda: f"M-{uuid4().hex[:12]}")
    topic_id: str = Field(min_length=1)
    mode: MaintenanceMode = MaintenanceMode.INCREMENTAL
    force_refresh: bool = False
    requested_new_paper_ids: tuple[str, ...] = ()
    # Evaluation only: observations supplied instead of asking the registry.
    injected: tuple[IntegrityObservation, ...] = ()
    skip_registry: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    topic: MaintainedTopic | None = None
    records: tuple[ConclusionRecord, ...] = ()
    changes: tuple[IntegrityChange, ...] = ()
    effective: dict[str, IntegrityObservation] = Field(default_factory=dict)
    lookup_failures: tuple[str, ...] = ()
    papers_checked: int = 0
    new_paper_ids: tuple[str, ...] = ()

    plan: PropagationPlan | None = None
    outcomes: tuple[ConclusionOutcome, ...] = ()
    # Re-verification artefacts, kept so the reviewer sees exactly what was verified.
    reverified: dict[str, ResearchFinding] = Field(default_factory=dict)
    verifications: tuple[VerificationResult, ...] = ()
    resolved_evidence_ids: frozenset[str] = frozenset()
    review: ReviewResult | None = None

    new_findings: tuple[ResearchFinding, ...] = ()
    new_verifications: tuple[VerificationResult, ...] = ()

    failures: tuple[MaintenanceFailure, ...] = ()
    diff: ConclusionDiff | None = None
