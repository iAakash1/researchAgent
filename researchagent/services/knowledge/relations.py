"""Relation building.

Deliberately **not** an LLM step. Asking a model to state relationships between extracted
objects invites it to assert connections the paper never made — and unlike a quote, a
relationship has no verbatim form to check it against.

Instead, relations are derived from evidence that already exists: two objects are related
when the paper says so in the same sentence, or when an extractor recorded the link as a
field (a result naming its metric and dataset). The evidence of the relation is the
evidence of the objects it connects.

The rule this enforces is the v0.7 one, brought forward: only extracted facts become graph
edges, never generated ones.
"""

from __future__ import annotations

from researchagent.core.evidence import Evidence
from researchagent.core.logging import get_logger
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.knowledge import (
    KnowledgeKind,
    KnowledgeObject,
    KnowledgeRelation,
    RelationPredicate,
    ResultDetails,
)
from researchagent.services.knowledge.grounding import normalise

logger = get_logger(__name__)


class RelationBuilder:
    """Derives typed relations from the objects and their evidence."""

    name = "relation_builder"

    def build(self, objects: tuple[KnowledgeObject, ...]) -> tuple[KnowledgeRelation, ...]:
        by_kind = {kind: [o for o in objects if o.kind is kind] for kind in KnowledgeKind}
        relations: list[KnowledgeRelation] = []

        relations.extend(self._from_result_fields(by_kind))
        relations.extend(self._from_co_location(by_kind))

        deduplicated = _deduplicate(relations)
        logger.debug("relations_built", proposed=len(relations), kept=len(deduplicated))
        return deduplicated

    def _from_result_fields(
        self, by_kind: dict[KnowledgeKind, list[KnowledgeObject]]
    ) -> list[KnowledgeRelation]:
        """A result already names its metric and dataset; match those to real objects.

        This is the strongest relation evidence available: the extractor read both sides
        out of the same grounded sentence.
        """
        relations: list[KnowledgeRelation] = []
        metrics = by_kind[KnowledgeKind.METRIC]
        datasets = by_kind[KnowledgeKind.DATASET]

        for result in by_kind[KnowledgeKind.RESULT]:
            if not isinstance(result.details, ResultDetails):
                continue

            metric = _match_by_name(result.details.metric_name, metrics)
            if metric is not None:
                relations.append(
                    _relation(
                        RelationPredicate.MEASURED_BY,
                        result,
                        metric,
                        observation=(
                            f"the result names metric {result.details.metric_name!r} in its "
                            f"own quoted sentence"
                        ),
                    )
                )

            dataset = _match_by_name(result.details.dataset_name, datasets)
            if dataset is not None:
                relations.append(
                    _relation(
                        RelationPredicate.REPORTED_ON,
                        result,
                        dataset,
                        observation=(
                            f"the result names dataset {result.details.dataset_name!r} in its "
                            f"own quoted sentence"
                        ),
                    )
                )

        return relations

    def _from_co_location(
        self, by_kind: dict[KnowledgeKind, list[KnowledgeObject]]
    ) -> list[KnowledgeRelation]:
        """Two objects named in the same paragraph are related by that paragraph.

        Weaker than a field match and scored accordingly — co-occurrence is real evidence
        of association, but it is not proof of the specific predicate, so these relations
        carry visibly lower confidence.
        """
        relations: list[KnowledgeRelation] = []

        for predicate, (domain_kind, range_kind) in (
            (RelationPredicate.EVALUATED_ON, (KnowledgeKind.METHOD, KnowledgeKind.DATASET)),
            (RelationPredicate.LIMITS, (KnowledgeKind.LIMITATION, KnowledgeKind.METHOD)),
        ):
            for subject in by_kind[domain_kind]:
                for candidate in by_kind[range_kind]:
                    shared = _shared_paragraph(subject, candidate)
                    if shared is None:
                        continue
                    relations.append(
                        _relation(
                            predicate,
                            subject,
                            candidate,
                            observation=(f"both are supported by the same paragraph at {shared}"),
                            strength=0.6,
                        )
                    )

        return relations


def _relation(
    predicate: RelationPredicate,
    subject: KnowledgeObject,
    target: KnowledgeObject,
    *,
    observation: str,
    strength: float = 1.0,
) -> KnowledgeRelation:
    evidence: tuple[Evidence, ...] = (*subject.evidence[:1], *target.evidence[:1])
    return KnowledgeRelation(
        id=f"{subject.id}--{predicate.value}--{target.id}",
        predicate=predicate,
        subject_id=subject.id,
        object_id=target.id,
        evidence=evidence,
        confidence=Confidence.from_signals(
            [ConfidenceSignal(name="relation_basis", value=strength, observation=observation)]
        ),
    )


def _match_by_name(name: str | None, candidates: list[KnowledgeObject]) -> KnowledgeObject | None:
    """Find the object a result's field refers to. Substring either way, longest first."""
    if not name:
        return None
    needle = normalise(name)
    if not needle:
        return None

    best: KnowledgeObject | None = None
    for candidate in candidates:
        haystack = normalise(candidate.name)
        if not haystack:
            continue
        matches = needle in haystack or haystack in needle
        if matches and (best is None or len(candidate.name) > len(best.name)):
            best = candidate
    return best


def _shared_paragraph(left: KnowledgeObject, right: KnowledgeObject) -> str | None:
    """The address of a paragraph supporting both objects, if there is one."""
    left_locations = {
        (item.location.section_id, item.location.paragraph_index): item.location
        for item in left.evidence
        if item.location.paragraph_index is not None
    }
    for item in right.evidence:
        key = (item.location.section_id, item.location.paragraph_index)
        if item.location.paragraph_index is not None and key in left_locations:
            return left_locations[key].describe()
    return None


def _deduplicate(relations: list[KnowledgeRelation]) -> tuple[KnowledgeRelation, ...]:
    """Keep the highest-confidence relation per (subject, predicate, object)."""
    best: dict[str, KnowledgeRelation] = {}
    for relation in relations:
        existing = best.get(relation.id)
        if existing is None or relation.confidence.score > existing.confidence.score:
            best[relation.id] = relation
    return tuple(best.values())
