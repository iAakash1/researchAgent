"""Fixtures for the agentic loop.

Everything is built from real domain objects and driven by a scripted fake LLM, so the
tests exercise the actual citation-resolution and verdict logic rather than a mock of it.
No network, no Ollama, no Groq.
"""

from __future__ import annotations

import pytest

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence, ConfidenceSignal, ValidationResult
from researchagent.models.bundle import BundledEvidence, EvidenceBundle
from researchagent.models.knowledge import (
    KnowledgeKind,
    KnowledgeObject,
    MethodDetails,
)
from researchagent.models.query import QueryIntent, ResearchQuery
from researchagent.models.reasoning import Citation, ResearchFinding
from researchagent.models.research import QuestionPriority, ResearchQuestion


def a_location(paper_id: str, page: int = 4) -> SourceLocation:
    return SourceLocation(
        document_id=paper_id,
        page=page,
        section_title="Experiments",
        paragraph_index=2,
        char_start=100,
        char_end=200,
    )


def an_evidence(paper_id: str, quote: str, *, page: int = 4) -> Evidence:
    return Evidence.from_text(
        claim=quote, quote=quote, location=a_location(paper_id, page), produced_by="test"
    )


def an_object(paper_id: str, name: str, index: int = 0) -> KnowledgeObject:
    return KnowledgeObject(
        id=f"{paper_id}#method:{name.lower().replace(' ', '-')}:{index}",
        kind=KnowledgeKind.METHOD,
        paper_id=paper_id,
        name=name,
        description=f"{name} as described by {paper_id}",
        details=MethodDetails(),
        evidence=(an_evidence(paper_id, f"we use {name} for this"),),
        confidence=Confidence(
            score=0.8,
            signals=(ConfidenceSignal(name="grounded", value=0.8, observation="quote located"),),
        ),
        extracted_by="test",
    )


def a_bundle(
    bundle_id: str, papers: tuple[str, ...] = ("manual:01", "manual:02")
) -> EvidenceBundle:
    objects = tuple(
        an_object(paper, "Circuit Breaker", index) for index, paper in enumerate(papers)
    )
    evidence = tuple(
        BundledEvidence(evidence=obj.evidence[0], paper_id=obj.paper_id, knowledge_object_id=obj.id)
        for obj in objects
    )
    return EvidenceBundle(
        id=bundle_id,
        query=ResearchQuery(text="how is overload mitigated", intent=QueryIntent.ANSWER),
        knowledge_objects=objects,
        evidence=evidence,
        validation=ValidationResult.passed(
            validator="test",
            subject_id=bundle_id,
            subject_type="EvidenceBundle",
            confidence=Confidence(score=0.9),
        ),
        built_by="test",
    )


@pytest.fixture
def bundle() -> EvidenceBundle:
    return a_bundle("B-1")


@pytest.fixture
def question() -> ResearchQuestion:
    return ResearchQuestion(
        id="RQ1",
        question="Which techniques mitigate overload in distributed systems?",
        rationale="Overload drives metastable failure and the mitigations differ",
        priority=QuestionPriority.HIGH,
        keywords=["overload", "circuit breaker"],
    )


@pytest.fixture
def finding(bundle: EvidenceBundle) -> ResearchFinding:
    return ResearchFinding(
        question_id="RQ1",
        statement="Circuit breakers are reported as an overload mitigation by two papers.",
        citations=(
            Citation(
                bundle_id=bundle.id,
                evidence_ids=tuple(item.evidence.id for item in bundle.evidence),
                paper_ids=("manual:01", "manual:02"),
            ),
        ),
        produced_by="reasoning",
    )
