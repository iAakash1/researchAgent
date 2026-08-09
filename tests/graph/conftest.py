"""Fixtures for the knowledge graph tests.

Everything is built from real domain objects — KnowledgeObject, Evidence, SourceLocation —
rather than stubs, because the properties under test (provenance survives mapping, ids stay
stable, untrusted edges get rejected) are properties of those types.
"""

from __future__ import annotations

import pytest

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.knowledge import (
    DatasetDetails,
    KnowledgeKind,
    KnowledgeObject,
    KnowledgeRelation,
    MethodDetails,
    MetricDetails,
    PaperKnowledge,
    RelationPredicate,
    ResultDetails,
)
from researchagent.models.paper import Paper, SourceName


def location(paper_id: str, page: int = 3, paragraph: int = 2) -> SourceLocation:
    return SourceLocation(
        document_id=paper_id,
        page=page,
        section="Experiments",
        paragraph_index=paragraph,
        char_start=100,
        char_end=180,
    )


def evidence(paper_id: str, quote: str, *, page: int = 3) -> Evidence:
    return Evidence.from_text(
        claim=quote,
        quote=quote,
        location=location(paper_id, page=page),
        produced_by="test_grounder",
    )


def knowledge_object(
    paper_id: str,
    kind: KnowledgeKind,
    name: str,
    index: int = 0,
    *,
    with_evidence: bool = True,
    numeric_value: float | None = None,
    metric_name: str | None = None,
    dataset_name: str | None = None,
) -> KnowledgeObject:
    details = {
        KnowledgeKind.METHOD: MethodDetails(),
        KnowledgeKind.DATASET: DatasetDetails(),
        KnowledgeKind.METRIC: MetricDetails(),
        KnowledgeKind.RESULT: ResultDetails(
            numeric_value=numeric_value, metric_name=metric_name, dataset_name=dataset_name
        ),
    }[kind]
    return KnowledgeObject(
        id=f"{paper_id}#{kind.value}:{name.lower().replace(' ', '-')}:{index}",
        kind=kind,
        paper_id=paper_id,
        name=name,
        description=f"{name} as described by {paper_id}",
        details=details,
        # An object with no evidence cannot be constructed, so "ungrounded" is modelled as
        # a *relation* with no evidence rather than an object with none.
        evidence=(evidence(paper_id, f"we use {name} throughout"),) if with_evidence else (),
        confidence=Confidence(
            score=0.8,
            signals=(ConfidenceSignal(name="grounded", value=0.8, observation="quote located"),),
        ),
        extracted_by="test_extractor",
    )


def relation(
    predicate: RelationPredicate,
    subject: KnowledgeObject,
    obj: KnowledgeObject,
    *,
    grounded: bool = True,
) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=f"{subject.id}--{predicate.value}--{obj.id}",
        predicate=predicate,
        subject_id=subject.id,
        object_id=obj.id,
        evidence=(
            (evidence(subject.paper_id, f"{subject.name} evaluated on {obj.name}"),)
            if grounded
            else ()
        ),
        confidence=Confidence(score=0.75) if grounded else Confidence.unknown(),
    )


@pytest.fixture
def paper_a() -> PaperKnowledge:
    """Paper A: RAG evaluated on MIMIC-III, reporting F1 = 0.82."""
    method = knowledge_object("manual:01", KnowledgeKind.METHOD, "RAG")
    dataset = knowledge_object("manual:01", KnowledgeKind.DATASET, "MIMIC-III")
    metric = knowledge_object("manual:01", KnowledgeKind.METRIC, "F1")
    result = knowledge_object(
        "manual:01",
        KnowledgeKind.RESULT,
        "F1 on MIMIC-III",
        numeric_value=0.82,
        metric_name="F1",
        dataset_name="MIMIC-III",
    )
    return PaperKnowledge(
        paper_id="manual:01",
        document_sha256="a" * 64,
        objects=(method, dataset, metric, result),
        relations=(
            relation(RelationPredicate.EVALUATED_ON, method, dataset),
            relation(RelationPredicate.MEASURED_BY, result, metric),
            relation(RelationPredicate.PRODUCED_BY, result, method),
            relation(RelationPredicate.REPORTED_ON, result, dataset),
        ),
    )


@pytest.fixture
def paper_b() -> PaperKnowledge:
    """Paper B: the same method and dataset, a materially different number."""
    method = knowledge_object("manual:02", KnowledgeKind.METHOD, "RAG")
    dataset = knowledge_object("manual:02", KnowledgeKind.DATASET, "MIMIC-III")
    metric = knowledge_object("manual:02", KnowledgeKind.METRIC, "F1")
    result = knowledge_object(
        "manual:02",
        KnowledgeKind.RESULT,
        "F1 on MIMIC-III",
        numeric_value=0.41,
        metric_name="F1",
        dataset_name="MIMIC-III",
    )
    return PaperKnowledge(
        paper_id="manual:02",
        document_sha256="b" * 64,
        objects=(method, dataset, metric, result),
        relations=(
            relation(RelationPredicate.EVALUATED_ON, method, dataset),
            relation(RelationPredicate.MEASURED_BY, result, metric),
            relation(RelationPredicate.PRODUCED_BY, result, method),
        ),
    )


@pytest.fixture
def papers() -> dict[str, Paper]:
    return {
        "manual:01": Paper(
            id="manual:01",
            title="Retrieval-Augmented Clinical QA",
            source=SourceName.MANUAL,
            provider="manual",
            year=2023,
        ),
        "manual:02": Paper(
            id="manual:02",
            title="Revisiting RAG on Clinical Notes",
            source=SourceName.MANUAL,
            provider="manual",
            year=2024,
        ),
    }
