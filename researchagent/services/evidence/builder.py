"""EvidenceBundleBuilder — assembles the canonical retrieval unit.

Walks all four retrieval layers for one query and composes the result into a bundle:

    query -> knowledge (L1) -> cross-paper agreement (L4) -> evidence (L2)
          -> contradictions -> coverage -> confidence -> validation

The builder never invents. Every knowledge object comes from the repository, every
evidence item is one the extraction stage already grounded in a document, every relation
was derived from evidence, and every contradiction is mechanically checkable. The bundle
is an assembly of existing facts, which is what makes it safe to hand to a model.
"""

from __future__ import annotations

import hashlib
import time

from researchagent.config.schemas import BundleSettings
from researchagent.core.interfaces.retrieval import (
    CrossPaperRetriever,
    EvidenceRetriever,
    KnowledgeRetriever,
)
from researchagent.core.logging import get_logger
from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationResult,
    aggregate,
)
from researchagent.models.bundle import (
    BundleCoverage,
    BundledEvidence,
    BundleStatistics,
    EvidenceBundle,
    EvidenceStance,
)
from researchagent.models.knowledge import KnowledgeObject, KnowledgeRelation
from researchagent.models.query import ResearchQuery
from researchagent.services.evidence.contradictions import ContradictionDetector
from researchagent.services.validation.bundle import (
    BundleCoverageValidator,
    ProvenanceValidator,
)

logger = get_logger(__name__)


class EvidenceBundleBuilder:
    """Builds one bundle per query."""

    name = "evidence_bundle_builder"

    def __init__(
        self,
        knowledge: KnowledgeRetriever,
        evidence: EvidenceRetriever,
        cross_paper: CrossPaperRetriever,
        contradictions: ContradictionDetector,
        settings: BundleSettings | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._evidence = evidence
        self._cross_paper = cross_paper
        self._contradictions = contradictions
        self._settings = settings or BundleSettings()

    async def build(
        self, query: ResearchQuery, *, relations: tuple[KnowledgeRelation, ...] = ()
    ) -> EvidenceBundle:
        started = time.perf_counter()

        objects, considered = await self._collect_objects(query)
        bundled = await self._collect_evidence(objects)
        contradictions = self._contradictions.detect(objects)
        applicable_relations = _relations_within(relations, objects)

        coverage = self._coverage(query, objects, bundled, considered)
        statistics = _statistics(objects, bundled, applicable_relations, len(contradictions))
        confidence = self._confidence(objects, bundled, coverage, statistics)
        validation = self._validate(query, objects, bundled, coverage, confidence)

        bundle = EvidenceBundle(
            id=_bundle_id(query),
            query=query,
            knowledge_objects=objects,
            evidence=bundled,
            relations=applicable_relations,
            contradictions=contradictions,
            coverage=coverage,
            statistics=statistics,
            confidence=confidence,
            validation=validation,
            built_by=self.name,
        )

        logger.info(
            "bundle_built",
            bundle_id=bundle.id,
            question_id=query.question_id,
            objects=len(objects),
            evidence=len(bundled),
            papers=coverage.paper_count,
            contradictions=len(contradictions),
            confidence=confidence.score,
            trusted=bundle.is_trusted,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return bundle

    async def _collect_objects(
        self, query: ResearchQuery
    ) -> tuple[tuple[KnowledgeObject, ...], int]:
        """Layers 1 and 4: matching facts, re-ranked by cross-paper agreement."""
        result = await self._cross_paper.retrieve(
            query.model_copy(update={"limit": self._settings.max_objects})
        )
        objects = tuple(
            hit.item for hit in result.hits if hit.score >= self._settings.min_object_score
        )
        return objects, result.considered

    async def _collect_evidence(
        self, objects: tuple[KnowledgeObject, ...]
    ) -> tuple[BundledEvidence, ...]:
        """Layer 2: the support for exactly these facts.

        A provenance walk, not a search — the objects are already chosen, and their
        evidence is what makes them checkable.
        """
        if not objects:
            return ()

        records = await self._evidence.for_objects(tuple(item.id for item in objects))
        by_object: dict[str, list[BundledEvidence]] = {}

        for record in records:
            for object_id in record.knowledge_object_ids:
                bundled = BundledEvidence(
                    evidence=record.evidence,
                    paper_id=record.paper_id,
                    knowledge_object_id=object_id,
                    stance=EvidenceStance.SUPPORTING,
                    relevance=Confidence.from_signals(
                        [
                            ConfidenceSignal(
                                name="linked_support",
                                value=1.0,
                                observation=(
                                    f"evidence is linked to {object_id} at "
                                    f"{record.location.describe()}"
                                ),
                            )
                        ]
                    ),
                )
                by_object.setdefault(object_id, []).append(bundled)

        collected: list[BundledEvidence] = []
        for object_id in (item.id for item in objects):
            collected.extend(by_object.get(object_id, [])[: self._settings.max_evidence_per_object])

        # Fall back to the evidence the objects carry when the index has not been built.
        if not collected:
            collected = [
                BundledEvidence(
                    evidence=item,
                    paper_id=obj.paper_id,
                    knowledge_object_id=obj.id,
                    relevance=Confidence.from_signals(
                        [
                            ConfidenceSignal(
                                name="owned_evidence",
                                value=1.0,
                                observation=f"evidence carried by {obj.id} itself",
                            )
                        ]
                    ),
                )
                for obj in objects
                for item in obj.evidence[: self._settings.max_evidence_per_object]
            ]

        known = {item.id for item in objects}
        return tuple(item for item in collected if item.knowledge_object_id in known)

    def _coverage(
        self,
        query: ResearchQuery,
        objects: tuple[KnowledgeObject, ...],
        evidence: tuple[BundledEvidence, ...],
        considered: int,
    ) -> BundleCoverage:
        with_evidence = {item.knowledge_object_id for item in evidence}
        return BundleCoverage(
            papers_represented=tuple(sorted({item.paper_id for item in objects})),
            kinds_covered=tuple(sorted({item.kind for item in objects}, key=lambda k: k.value)),
            objects_with_evidence=sum(1 for item in objects if item.id in with_evidence),
            objects_without_evidence=sum(1 for item in objects if item.id not in with_evidence),
            papers_considered=considered,
            question_id=query.question_id,
        )

    def _confidence(
        self,
        objects: tuple[KnowledgeObject, ...],
        evidence: tuple[BundledEvidence, ...],
        coverage: BundleCoverage,
        statistics: BundleStatistics,
    ) -> Confidence:
        if not objects:
            return Confidence.unknown()

        mean_object_confidence = sum(item.confidence.score for item in objects) / len(objects)
        return Confidence.from_signals(
            [
                ConfidenceSignal(
                    name="object_confidence",
                    value=mean_object_confidence,
                    observation=(
                        f"{len(objects)} knowledge objects with mean validated confidence "
                        f"{mean_object_confidence:.2f}"
                    ),
                ),
                ConfidenceSignal(
                    name="evidence_completeness",
                    value=coverage.evidence_completeness,
                    observation=(
                        f"{coverage.objects_with_evidence} of {len(objects)} objects carry "
                        "retrievable evidence"
                    ),
                ),
                ConfidenceSignal(
                    name="corroboration",
                    value=min(coverage.paper_count / self._settings.corroboration_target, 1.0),
                    observation=(
                        f"{coverage.paper_count} distinct paper(s) contribute: "
                        f"{list(coverage.papers_represented)}"
                    ),
                ),
                ConfidenceSignal(
                    name="evidence_density",
                    value=min(statistics.evidence_density / 2, 1.0),
                    observation=(
                        f"{statistics.evidence_items} evidence items across "
                        f"{statistics.knowledge_objects} objects"
                    ),
                ),
            ]
        )

    def _validate(
        self,
        query: ResearchQuery,
        objects: tuple[KnowledgeObject, ...],
        evidence: tuple[BundledEvidence, ...],
        coverage: BundleCoverage,
        confidence: Confidence,
    ) -> ValidationResult:
        checks = [
            ProvenanceValidator().check_bundle(query, objects, evidence, confidence),
            BundleCoverageValidator(self._settings).check_coverage(query, coverage, confidence),
        ]
        return aggregate(
            checks,
            validator="bundle_validator",
            subject_id=_bundle_id(query),
            subject_type="EvidenceBundle",
        )


def _relations_within(
    relations: tuple[KnowledgeRelation, ...], objects: tuple[KnowledgeObject, ...]
) -> tuple[KnowledgeRelation, ...]:
    """Keep only relations whose endpoints are both present in the bundle."""
    known = {item.id for item in objects}
    return tuple(
        relation
        for relation in relations
        if relation.subject_id in known and relation.object_id in known
    )


def _statistics(
    objects: tuple[KnowledgeObject, ...],
    evidence: tuple[BundledEvidence, ...],
    relations: tuple[KnowledgeRelation, ...],
    contradictions: int,
) -> BundleStatistics:
    return BundleStatistics(
        knowledge_objects=len(objects),
        evidence_items=len(evidence),
        supporting=sum(1 for e in evidence if e.stance is EvidenceStance.SUPPORTING),
        contradicting=sum(1 for e in evidence if e.stance is EvidenceStance.CONTRADICTING),
        unknown=sum(1 for e in evidence if e.stance is EvidenceStance.UNKNOWN),
        relations=len(relations),
        contradictions=contradictions,
        distinct_papers=len({item.paper_id for item in objects}),
        distinct_pages=len(
            {(e.paper_id, e.location.page) for e in evidence if e.location.page is not None}
        ),
    )


def _bundle_id(query: ResearchQuery) -> str:
    """Deterministic: the same question over the same filters yields the same bundle id."""
    parts = (
        query.question_id or "",
        query.text,
        ",".join(sorted(kind.value for kind in query.kinds)),
        ",".join(sorted(query.paper_ids)),
    )
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]  # noqa: S324
    prefix = query.question_id or "query"
    return f"bundle:{prefix}:{digest}"
