"""EvidenceBundle — the canonical retrieval abstraction.

The smallest complete unit of trustworthy context. Not a chunk, not a paragraph, not a
prompt: a structured collection of evidence supporting one or more knowledge objects,
carrying its own provenance, coverage, confidence and disagreements.

This is what every future reasoning engine consumes. A reasoning engine handed text can
invent a conclusion and cite nothing; a reasoning engine handed a bundle has, for every
fact available to it, the paper, page, section and paragraph that states it — and is
told explicitly where the literature disagrees.

Two properties are deliberate:

* **Contradictions are carried, never resolved.** A bundle that silently dropped the
  paper disagreeing with the majority would manufacture a consensus. Disagreement is the
  most valuable thing a literature review can surface, so it is a first-class field.
* **Validation travels inside the bundle**, unlike other artefacts which use
  ``Validated[T]``. A bundle is the unit that leaves the system into a reasoning engine;
  a wrapper can be unwrapped, and a bundle whose verdict was dropped in transit would
  look exactly like one that passed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence, ValidationResult
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, KnowledgeRelation
from researchagent.models.query import ResearchQuery


class EvidenceStance(StrEnum):
    """How a piece of evidence relates to the claim it is bundled under."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    # Present and relevant, but does not settle the question either way. Recorded rather
    # than discarded: "we looked and the literature is silent" is a finding.
    UNKNOWN = "unknown"


class ContradictionKind(StrEnum):
    """What sort of disagreement was detected. Only mechanically checkable kinds exist."""

    # Two papers report materially different numbers for the same metric and dataset.
    VALUE_CONFLICT = "value_conflict"
    # Two papers disagree on a factual attribute (public/private, novel/prior).
    ATTRIBUTE_CONFLICT = "attribute_conflict"


class BundledEvidence(BaseModel):
    """One evidence item inside a bundle, with its stance and its origin."""

    model_config = {"frozen": True}

    evidence: Evidence
    paper_id: str = Field(min_length=1)
    knowledge_object_id: str | None = None
    stance: EvidenceStance = EvidenceStance.SUPPORTING
    # Why this item earned its place. Reuses the confidence primitive so a bundle's
    # relevance is auditable by exactly the same rules as everything else.
    relevance: Confidence = Field(default_factory=Confidence.unknown)

    @property
    def location(self) -> SourceLocation:
        return self.evidence.location

    @property
    def quote(self) -> str:
        return self.evidence.quote or ""


class Contradiction(BaseModel):
    """A detected disagreement between two knowledge objects, with both sides' evidence.

    Both sides are always carried. A contradiction that recorded only the losing claim,
    or only the winning one, would be an opinion rather than an observation.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    kind: ContradictionKind
    description: str = Field(min_length=1)
    left_object_id: str = Field(min_length=1)
    right_object_id: str = Field(min_length=1)
    left_paper_id: str = Field(min_length=1)
    right_paper_id: str = Field(min_length=1)
    left_evidence: tuple[Evidence, ...] = ()
    right_evidence: tuple[Evidence, ...] = ()
    detected_by: str = Field(min_length=1)
    confidence: Confidence = Field(default_factory=Confidence.unknown)

    @model_validator(mode="after")
    def _sides_differ(self) -> Contradiction:
        if self.left_object_id == self.right_object_id:
            raise ValueError("a contradiction must hold two distinct objects")
        return self

    @property
    def is_cross_paper(self) -> bool:
        """Disagreement between papers is far more significant than within one."""
        return self.left_paper_id != self.right_paper_id


class BundleCoverage(BaseModel):
    """What the bundle does and does not cover.

    Coverage is reported so a reasoning engine can tell a well-supported answer from a
    thin one, and so the reviewer can say "you concluded this from two papers".
    """

    model_config = {"frozen": True}

    papers_represented: tuple[str, ...] = ()
    kinds_covered: tuple[KnowledgeKind, ...] = ()
    objects_with_evidence: int = Field(default=0, ge=0)
    objects_without_evidence: int = Field(default=0, ge=0)
    # Papers considered but contributing nothing — the silence in the literature.
    papers_considered: int = Field(default=0, ge=0)
    question_id: str | None = None

    @property
    def paper_count(self) -> int:
        return len(self.papers_represented)

    @property
    def evidence_completeness(self) -> float:
        total = self.objects_with_evidence + self.objects_without_evidence
        return self.objects_with_evidence / total if total else 0.0


class BundleStatistics(BaseModel):
    model_config = {"frozen": True}

    knowledge_objects: int = Field(default=0, ge=0)
    evidence_items: int = Field(default=0, ge=0)
    supporting: int = Field(default=0, ge=0)
    contradicting: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    contradictions: int = Field(default=0, ge=0)
    distinct_papers: int = Field(default=0, ge=0)
    distinct_pages: int = Field(default=0, ge=0)

    @property
    def evidence_density(self) -> float:
        """Evidence items per knowledge object. Thin bundles are visibly thin."""
        return self.evidence_items / self.knowledge_objects if self.knowledge_objects else 0.0


class EvidenceBundle(BaseModel):
    """The canonical unit of trustworthy context."""

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    query: ResearchQuery

    knowledge_objects: tuple[KnowledgeObject, ...] = ()
    evidence: tuple[BundledEvidence, ...] = ()
    relations: tuple[KnowledgeRelation, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()

    coverage: BundleCoverage = Field(default_factory=BundleCoverage)
    statistics: BundleStatistics = Field(default_factory=BundleStatistics)
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    validation: ValidationResult

    built_by: str = Field(min_length=1)
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _evidence_belongs_to_the_bundle(self) -> EvidenceBundle:
        """Every referenced object must be present.

        A bundle whose evidence points at a knowledge object it does not carry is a
        bundle a reasoning engine cannot trace, which defeats the purpose of bundling.
        """
        known = {item.id for item in self.knowledge_objects}
        dangling = {
            item.knowledge_object_id
            for item in self.evidence
            if item.knowledge_object_id is not None and item.knowledge_object_id not in known
        }
        if dangling:
            raise ValueError(f"bundle evidence references absent objects: {sorted(dangling)}")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.knowledge_objects and not self.evidence

    @property
    def is_trusted(self) -> bool:
        return self.validation.success

    @property
    def has_disagreement(self) -> bool:
        return bool(self.contradictions)

    def evidence_for(self, object_id: str) -> tuple[BundledEvidence, ...]:
        return tuple(item for item in self.evidence if item.knowledge_object_id == object_id)

    def objects_of(self, kind: KnowledgeKind) -> tuple[KnowledgeObject, ...]:
        return tuple(item for item in self.knowledge_objects if item.kind is kind)

    def by_stance(self, stance: EvidenceStance) -> tuple[BundledEvidence, ...]:
        return tuple(item for item in self.evidence if item.stance is stance)

    def citations(self) -> tuple[str, ...]:
        """Every distinct provenance address in the bundle, in reading order.

        What a report's footnotes are generated from, and what the v0.9 verifier
        re-opens to check a claim.
        """
        seen: dict[str, None] = {}
        for item in sorted(
            self.evidence,
            key=lambda entry: (entry.paper_id, entry.location.page or 0),
        ):
            seen.setdefault(item.location.describe(), None)
        return tuple(seen)
