"""Knowledge objects — the canonical reasoning layer.

From v0.5 onward the system does not reason over PDFs, and it does not reason over plain
text. It reasons over :class:`KnowledgeObject` instances: typed, evidence-backed facts
extracted from validated documents.

The load-bearing invariant is enforced by the model itself, not by convention:

    A KnowledgeObject cannot be constructed without at least one piece of Evidence.

That single constraint is what stops a language model's fluent guess from entering the
system as a fact. An extraction whose quote could not be located in the source document
produces no evidence, and therefore cannot produce a knowledge object at all — the
failure happens at construction, not at review time.

Provenance is total: KnowledgeObject -> Evidence -> SourceLocation -> paragraph, section,
page, document, and the sha256 of the originating PDF.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence


class KnowledgeKind(StrEnum):
    """What a knowledge object asserts.

    One member per implemented extractor. The vocabulary grows with the extractors that
    populate it — a kind nothing can produce would be a promise, not a contract.
    """

    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    RESULT = "result"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"

    @property
    def is_claim_like(self) -> bool:
        """Kinds that assert something about the world, rather than naming an artefact.

        These carry the highest hallucination risk and get the strictest validation:
        inventing a plausible-sounding limitation is easy, inventing a dataset name that
        appears verbatim in the text is not.
        """
        return self in (KnowledgeKind.RESULT, KnowledgeKind.LIMITATION, KnowledgeKind.FUTURE_WORK)


class RelationPredicate(StrEnum):
    """Typed edges between knowledge objects.

    Deliberately few and concrete. These become knowledge-graph edges in v0.7, and an
    edge whose meaning is vague is an edge nothing can query.
    """

    EVALUATED_ON = "evaluated_on"  # method -> dataset
    MEASURED_BY = "measured_by"  # result -> metric
    REPORTED_ON = "reported_on"  # result -> dataset
    PRODUCED_BY = "produced_by"  # result -> method
    LIMITS = "limits"  # limitation -> method
    EXTENDS = "extends"  # future_work -> limitation

    @property
    def domain(self) -> KnowledgeKind:
        return _RELATION_TYPES[self][0]

    @property
    def range(self) -> KnowledgeKind:
        return _RELATION_TYPES[self][1]


_RELATION_TYPES: dict[RelationPredicate, tuple[KnowledgeKind, KnowledgeKind]] = {
    RelationPredicate.EVALUATED_ON: (KnowledgeKind.METHOD, KnowledgeKind.DATASET),
    RelationPredicate.MEASURED_BY: (KnowledgeKind.RESULT, KnowledgeKind.METRIC),
    RelationPredicate.REPORTED_ON: (KnowledgeKind.RESULT, KnowledgeKind.DATASET),
    RelationPredicate.PRODUCED_BY: (KnowledgeKind.RESULT, KnowledgeKind.METHOD),
    RelationPredicate.LIMITS: (KnowledgeKind.LIMITATION, KnowledgeKind.METHOD),
    RelationPredicate.EXTENDS: (KnowledgeKind.FUTURE_WORK, KnowledgeKind.LIMITATION),
}


class MethodDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.METHOD] = KnowledgeKind.METHOD
    category: str | None = Field(default=None, description="e.g. 'retrieval-augmented generation'")
    components: tuple[str, ...] = ()
    is_novel: bool | None = Field(default=None, description="Presented as this paper's own")


class DatasetDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.DATASET] = KnowledgeKind.DATASET
    domain: str | None = None
    size: str | None = Field(default=None, description="As stated, e.g. '1.2M examples'")
    url: str | None = None
    is_public: bool | None = None


class MetricDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.METRIC] = KnowledgeKind.METRIC
    unit: str | None = Field(default=None, description="e.g. '%', 'ms', 'F1'")
    higher_is_better: bool | None = None


class ResultDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.RESULT] = KnowledgeKind.RESULT
    metric_name: str | None = None
    dataset_name: str | None = None
    # Kept verbatim as printed *and* parsed. The v0.9 verifier re-checks the string
    # against the page; the float is what analysis compares.
    value_text: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    baseline_comparison: str | None = None


class LimitationDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.LIMITATION] = KnowledgeKind.LIMITATION
    # Whether the authors state this themselves, or it is a threat to validity they name.
    acknowledged_by_authors: bool = True
    affects: str | None = None


class FutureWorkDetails(BaseModel):
    model_config = {"frozen": True}
    kind: Literal[KnowledgeKind.FUTURE_WORK] = KnowledgeKind.FUTURE_WORK
    direction: str | None = None


KnowledgeDetails = Annotated[
    MethodDetails
    | DatasetDetails
    | MetricDetails
    | ResultDetails
    | LimitationDetails
    | FutureWorkDetails,
    Field(discriminator="kind"),
]


class KnowledgeRelation(BaseModel):
    """A typed, evidence-backed edge between two knowledge objects.

    Relations carry their own evidence because "the paper mentions both X and Y" is not
    the same claim as "the paper says X was evaluated on Y", and the knowledge graph must
    not conflate them.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    predicate: RelationPredicate
    subject_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()
    confidence: Confidence = Field(default_factory=Confidence.unknown)

    @model_validator(mode="after")
    def _no_self_reference(self) -> KnowledgeRelation:
        if self.subject_id == self.object_id:
            raise ValueError("a knowledge relation cannot point at itself")
        return self


class KnowledgeObject(BaseModel):
    """One typed, evidence-backed fact extracted from a paper."""

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    kind: KnowledgeKind
    paper_id: str = Field(min_length=1)

    name: str = Field(min_length=1, description="Canonical label, e.g. 'MIMIC-III'")
    description: str = Field(default="", description="What the paper says about it")
    details: KnowledgeDetails

    # Non-empty by construction. This is the anti-hallucination guarantee.
    evidence: tuple[Evidence, ...]
    confidence: Confidence = Field(default_factory=Confidence.unknown)

    extracted_by: str = Field(min_length=1)
    validated_by: tuple[str, ...] = ()
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _requires_evidence(self) -> KnowledgeObject:
        """No evidence, no knowledge.

        Enforced here rather than in a validator so that an unsupported extraction cannot
        exist even transiently — there is no window in which it could be logged, cached,
        or persisted before something rejects it.
        """
        if not self.evidence:
            raise ValueError(
                f"KnowledgeObject {self.id!r} has no evidence; "
                "knowledge without provenance is not knowledge"
            )
        return self

    @model_validator(mode="after")
    def _details_match_kind(self) -> KnowledgeObject:
        if self.details.kind is not self.kind:
            raise ValueError(
                f"details kind {self.details.kind} does not match object kind {self.kind}"
            )
        return self

    @property
    def primary_location(self) -> SourceLocation:
        """The most precise place this fact was found — what a citation points at."""
        return max(self.evidence, key=lambda item: item.location.precision).location

    @property
    def quotes(self) -> tuple[str, ...]:
        return tuple(item.quote for item in self.evidence if item.quote)

    def cite(self) -> str:
        """Human-readable provenance, e.g. ``manual:01 p.4 §Results ¶2``."""
        return self.primary_location.describe()


class ExtractionStats(BaseModel):
    """How the objects on a ``PaperKnowledge`` came to exist.

    Persisted with the knowledge rather than recomputed, because the counts only exist
    while the extractors are running. Without this, reusing cached knowledge produced
    objects with no record of how many proposals they came from — a numerator with no
    denominator, which is what made the corpus grounding rate exceed 1.0.

    ``None`` on knowledge extracted before this field existed: that is *unknown*, not
    zero, and the aggregate treats it as such.
    """

    model_config = {"frozen": True}

    proposed: int = Field(ge=0, description="Drafts the extractors returned")
    grounded: int = Field(ge=0, description="Drafts whose quote was located in the document")
    accepted: int = Field(ge=0, description="Grounded drafts that also passed validation")
    rejected_ungrounded: int = Field(default=0, ge=0)
    rejected_invalid: int = Field(default=0, ge=0)
    rejection_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _counts_must_reconcile(self) -> ExtractionStats:
        """A rate can only be trusted if its own inputs are consistent."""
        if self.grounded > self.proposed:
            raise ValueError(f"grounded ({self.grounded}) cannot exceed proposed ({self.proposed})")
        if self.accepted > self.grounded:
            raise ValueError(f"accepted ({self.accepted}) cannot exceed grounded ({self.grounded})")
        return self

    @property
    def grounding_rate(self) -> float:
        """Share of proposals the document's own text supported.

        The denominator is proposals, the numerator is *grounded* drafts — the direct
        hallucination measure. Validation rejections are a separate quality signal and
        belong in ``acceptance_rate``, not here.
        """
        return self.grounded / self.proposed if self.proposed else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Share of proposals that became knowledge, after grounding *and* validation."""
        return self.accepted / self.proposed if self.proposed else 0.0


class PaperKnowledge(BaseModel):
    """Everything known about one paper, with the verdict that admitted it."""

    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    document_sha256: str = Field(
        min_length=1, description="Ties this knowledge to the exact bytes it came from"
    )
    objects: tuple[KnowledgeObject, ...] = ()
    relations: tuple[KnowledgeRelation, ...] = ()
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # None means the counters were not recorded, not that they were zero.
    extraction: ExtractionStats | None = None

    def of_kind(self, kind: KnowledgeKind) -> tuple[KnowledgeObject, ...]:
        return tuple(item for item in self.objects if item.kind is kind)

    def by_id(self, object_id: str) -> KnowledgeObject | None:
        return next((item for item in self.objects if item.id == object_id), None)

    @property
    def kinds_present(self) -> tuple[KnowledgeKind, ...]:
        return tuple(sorted({item.kind for item in self.objects}, key=lambda k: k.value))

    @property
    def evidence_count(self) -> int:
        return sum(len(item.evidence) for item in self.objects)

    @property
    def mean_confidence(self) -> float:
        if not self.objects:
            return 0.0
        return round(sum(item.confidence.score for item in self.objects) / len(self.objects), 6)


def make_knowledge_id(paper_id: str, kind: KnowledgeKind, name: str, index: int) -> str:
    """Deterministic id so re-extracting the same paper does not duplicate records."""
    slug = "-".join(name.lower().split())[:48] or "unnamed"
    return f"{paper_id}#{kind.value}:{slug}:{index}"
