"""Builders for conclusion-maintenance tests.

Paper ids use Crossref's test prefix 10.5555, which never names a real work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.evidence import EvidenceRole
from researchagent.models.integrity import (
    IntegrityObservation,
    IntegrityStatus,
    ObservationBasis,
)
from researchagent.models.knowledge import (
    KnowledgeKind,
    KnowledgeObject,
    MethodDetails,
    ResultDetails,
)
from researchagent.models.maintenance import (
    ConclusionRecord,
    MaintainedTopic,
    SupportLink,
    VerificationSnapshot,
)
from researchagent.models.reasoning import VerificationVerdict
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.repositories.maintenance_repository import (
    JsonConclusionRepository,
    JsonMaintenanceDiffRepository,
)

TOPIC = "T-test"


def paper(name: str) -> str:
    return f"doi:10.5555/{name}"


def location(paper_id: str, paragraph: int = 1) -> SourceLocation:
    return SourceLocation(
        document_id=paper_id,
        page=2,
        section_id="s1",
        section_title="Results",
        paragraph_index=paragraph,
    )


def support(
    paper_id: str, index: int = 0, *, role: EvidenceRole = EvidenceRole.FOUNDING
) -> SupportLink:
    name = paper_id.rsplit("/", 1)[-1]
    return SupportLink(
        evidence_id=f"E-{name}-{index}",
        knowledge_object_id=f"{paper_id}#method:circuit-breaker:{index}",
        paper_id=paper_id,
        location=location(paper_id, index),
        bundle_id="B-1",
        doi=paper_id.removeprefix("doi:"),
        quote=f"{name} reports that circuit breakers shed load ({index})",
        role=role,
    )


def verified(score: float = 0.9) -> VerificationSnapshot:
    return VerificationSnapshot(
        verdict=VerificationVerdict.VERIFIED,
        confidence=Confidence(
            score=score,
            signals=(
                ConfidenceSignal(name="provenance_resolved", value=score, observation="test"),
            ),
        ),
        verified_by="verification",
        run_id="R-0",
    )


def conclusion(
    conclusion_id: str,
    *supports: SupportLink,
    topic_id: str = TOPIC,
    statement: str | None = None,
) -> ConclusionRecord:
    return ConclusionRecord(
        conclusion_id=conclusion_id,
        version=1,
        topic_id=topic_id,
        run_id=topic_id,
        question_id="RQ1",
        statement=statement or f"Conclusion {conclusion_id} holds across the cited papers.",
        supports=supports,
        verification=verified(),
    )


def observation(
    paper_id: str, status: IntegrityStatus, *, basis: ObservationBasis | None = None
) -> IntegrityObservation:
    chosen = basis or (
        ObservationBasis.REGISTRY_RECORD
        if status is not IntegrityStatus.UNKNOWN
        else ObservationBasis.LOOKUP_FAILED
    )
    raw = {
        IntegrityStatus.RETRACTED: "retraction",
        IntegrityStatus.WITHDRAWN: "withdrawal",
        IntegrityStatus.EXPRESSION_OF_CONCERN: "expression_of_concern",
        IntegrityStatus.CORRECTED: "correction",
    }.get(status)
    return IntegrityObservation(
        paper_id=paper_id,
        identifier=paper_id,
        doi=paper_id.removeprefix("doi:"),
        status=status,
        basis=chosen,
        registry="crossref",
        raw_update_type=raw if chosen is ObservationBasis.REGISTRY_RECORD else None,
        notice_doi=f"10.5555/notice-{paper_id.rsplit('/', 1)[-1]}" if raw else None,
        notice_date="2026-09-01" if raw else None,
        observation=f"synthetic {status.value} observation",
    )


def evidence_for(paper_id: str, quote: str) -> Evidence:
    return Evidence.from_text(
        claim=quote, quote=quote, location=location(paper_id, 9), produced_by="test"
    )


def method_object(paper_id: str, name: str, index: int = 0) -> KnowledgeObject:
    return KnowledgeObject(
        id=f"{paper_id}#method:{name.lower().replace(' ', '-')}:{index}",
        kind=KnowledgeKind.METHOD,
        paper_id=paper_id,
        name=name,
        details=MethodDetails(),
        evidence=(evidence_for(paper_id, f"we use {name} to shed load in this system"),),
        extracted_by="test",
    )


def result_object(
    paper_id: str, value: float, *, metric: str = "F1", dataset: str = "MIMIC-III", index: int = 0
) -> KnowledgeObject:
    return KnowledgeObject(
        id=f"{paper_id}#result:{metric.lower()}:{index}",
        kind=KnowledgeKind.RESULT,
        paper_id=paper_id,
        name=f"{metric} on {dataset}",
        details=ResultDetails(
            metric_name=metric, dataset_name=dataset, value_text=str(value), numeric_value=value
        ),
        evidence=(evidence_for(paper_id, f"we report {metric} of {value} on {dataset} here"),),
        extracted_by="test",
    )


def plan() -> ResearchPlan:
    return ResearchPlan(
        topic="Overload mitigation",
        framing="How distributed systems shed load under overload conditions.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which techniques mitigate overload in distributed systems?",
                rationale="Overload drives metastable failure and mitigations differ",
                priority=QuestionPriority.HIGH,
                keywords=["overload"],
            )
        ],
        strategy=SearchStrategy(queries=["overload mitigation"]),
    )


def topic(*known: str, topic_id: str = TOPIC) -> MaintainedTopic:
    return MaintainedTopic(
        topic_id=topic_id, goal="Overload mitigation in systems", plan=plan(), known_paper_ids=known
    )


@pytest.fixture
def conclusions(tmp_path: Path) -> JsonConclusionRepository:
    return JsonConclusionRepository(tmp_path / "maintenance")


@pytest.fixture
def diffs(tmp_path: Path) -> JsonMaintenanceDiffRepository:
    return JsonMaintenanceDiffRepository(tmp_path / "maintenance")
