"""Conclusion persistence and the dependency graph: append-only history, resolution of
citations down to papers, and reverse traversal that reaches only dependents."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.validation import Confidence, ValidationResult
from researchagent.integrations.memory_graph import InMemoryGraphRepository
from researchagent.models.bundle import (
    BundledEvidence,
    Contradiction,
    ContradictionKind,
    EvidenceBundle,
)
from researchagent.models.evidence import EvidenceRecord, EvidenceRole, PaperEvidence
from researchagent.models.graph import EdgeKind, NodeKind
from researchagent.models.integrity import IntegrityStatus
from researchagent.models.maintenance import (
    ChangeType,
    ConclusionDiff,
    ConclusionRecord,
    ConclusionState,
    MaintenanceMode,
)
from researchagent.models.query import ResearchQuery
from researchagent.models.reasoning import (
    Citation,
    FindingStatus,
    ResearchFinding,
    VerificationResult,
    VerificationVerdict,
)
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.repositories.graph_repository import JsonGraphRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.repositories.maintenance_repository import (
    JsonConclusionRepository,
    JsonMaintenanceDiffRepository,
)
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.schemas.workflow import ResearchState
from researchagent.services.maintenance.dependency_graph import (
    DependencyGraphMapper,
    DependencyGraphStore,
    dependents_of,
)
from researchagent.services.maintenance.recorder import ConclusionRecorder, conclusion_id_for
from tests.maintenance.conftest import (
    TOPIC,
    conclusion,
    evidence_for,
    method_object,
    observation,
    paper,
    plan,
    support,
    topic,
)

P1, P2, P3 = paper("p1"), paper("p2"), paper("p3")


# --- append-only history ------------------------------------------------------------------


async def test_versions_are_append_only(conclusions: JsonConclusionRepository) -> None:
    first = conclusion("C1", support(P1))
    await conclusions.append(first)
    second = first.model_copy(
        update={"version": 2, "previous_version": 1, "state": ConclusionState.WEAKENED}
    )
    await conclusions.append(second)

    with pytest.raises(RepositoryError):
        await conclusions.append(second)  # rewriting v2
    with pytest.raises(RepositoryError):
        await conclusions.append(first)  # rewriting v1
    with pytest.raises(RepositoryError):
        await conclusions.append(
            first.model_copy(update={"version": 4, "previous_version": 3})
        )  # skipping v3

    history = await conclusions.history("C1")
    assert [item.version for item in history] == [1, 2]
    assert history[0].state is ConclusionState.ACTIVE  # the old version is untouched
    assert (await conclusions.latest("C1")) == second


async def test_for_topic_returns_latest_versions_only(
    conclusions: JsonConclusionRepository,
) -> None:
    await conclusions.append(conclusion("C1", support(P1)))
    await conclusions.append(conclusion("C2", support(P2)))
    await conclusions.append(conclusion("C9", support(P3), topic_id="other"))
    latest_c1 = conclusion("C1", support(P1)).model_copy(
        update={"version": 2, "previous_version": 1}
    )
    await conclusions.append(latest_c1)

    found = await conclusions.for_topic(TOPIC)

    assert [(item.conclusion_id, item.version) for item in found] == [("C1", 2), ("C2", 1)]


def test_version_chain_is_linear() -> None:
    dumped = conclusion("C1", support(P1)).model_dump()
    with pytest.raises(ValidationError):
        ConclusionRecord.model_validate(dumped | {"version": 3, "previous_version": 1})
    with pytest.raises(ValidationError):
        ConclusionRecord.model_validate(dumped | {"version": 1, "previous_version": 0})


def test_only_an_invalidated_conclusion_may_have_no_support() -> None:
    with pytest.raises(ValidationError):
        conclusion("C1")


async def test_topics_and_diffs_round_trip(
    conclusions: JsonConclusionRepository, diffs: JsonMaintenanceDiffRepository
) -> None:
    await conclusions.save_topic(topic(P1, P2))
    stored = await conclusions.get_topic(TOPIC)
    assert stored is not None and stored.known_paper_ids == (P1, P2)

    empty = ConclusionDiff(
        topic_id=TOPIC, maintenance_run_id="M-1", mode=MaintenanceMode.INCREMENTAL
    )
    await diffs.save(empty)
    latest = await diffs.latest(TOPIC)
    assert latest is not None and latest.id == empty.id
    assert latest.is_empty
    assert latest.counts == {kind.value: 0 for kind in ChangeType}
    assert (await diffs.get(empty.id)) == latest


# --- recording a run ----------------------------------------------------------------------


def bundle_with(*objects: tuple[str, str]) -> EvidenceBundle:
    built = [method_object(paper_id, name) for paper_id, name in objects]
    return EvidenceBundle(
        id="B-run",
        query=ResearchQuery(text="overload mitigation"),
        knowledge_objects=tuple(built),
        evidence=tuple(
            BundledEvidence(
                evidence=obj.evidence[0], paper_id=obj.paper_id, knowledge_object_id=obj.id
            )
            for obj in built
        ),
        validation=ValidationResult.passed(
            validator="test",
            subject_id="B-run",
            subject_type="EvidenceBundle",
            confidence=Confidence.unknown(),
        ),
        built_by="test",
    )


def finding(
    bundle: EvidenceBundle, *, status: FindingStatus, extra_ids: tuple[str, ...] = ()
) -> ResearchFinding:
    return ResearchFinding(
        question_id="RQ1",
        statement="Circuit breakers are reported as an overload mitigation by two papers.",
        citations=(
            Citation(
                bundle_id=bundle.id,
                evidence_ids=(*(item.evidence.id for item in bundle.evidence), *extra_ids),
            ),
        ),
        status=status,
        produced_by="reasoning",
    )


async def test_record_run_resolves_accepted_findings_to_papers(tmp_path: Path) -> None:
    bundles = JsonBundleRepository(tmp_path / "bundles")
    evidence = JsonEvidenceRepository(tmp_path / "evidence")
    knowledge = JsonKnowledgeRepository(tmp_path / "knowledge")
    repository = JsonConclusionRepository(tmp_path / "maintenance")
    bundle = bundle_with((P1, "Circuit Breaker"), (P2, "Circuit Breaker"))
    await bundles.save(bundle)

    # One evidence id resolves only through the evidence repository, one not at all.
    stray = method_object(P3, "Load Shedding")
    await evidence.save_paper(
        PaperEvidence(
            paper_id=P3,
            document_sha256="sha",
            records=(
                EvidenceRecord(
                    evidence=stray.evidence[0],
                    paper_id=P3,
                    document_sha256="sha",
                    links=(),
                ),
            ),
        )
    )
    accepted = finding(
        bundle, status=FindingStatus.VERIFIED, extra_ids=(stray.evidence[0].id, "E-missing")
    )
    rejected = finding(bundle, status=FindingStatus.REJECTED)
    session = ReasoningSession(
        findings=(accepted, rejected),
        verifications=(
            VerificationResult(
                finding_id=accepted.id,
                verdict=VerificationVerdict.VERIFIED,
                supporting=accepted.citations,
                verified_by="verification",
            ),
        ),
    )
    state = ResearchState(
        run_id="R-1", goal="Overload mitigation in systems", plan=plan(), reasoning=session
    )
    recorder = ConclusionRecorder(repository, bundles, evidence, knowledge)

    recorded = await recorder.record_run(state)
    again = await recorder.record_run(state)

    assert len(recorded) == 1 and again == ()
    record = recorded[0]
    assert record.conclusion_id == conclusion_id_for("R-1", accepted.id)
    assert record.version == 1 and record.topic_id == "R-1"
    assert record.paper_ids == (P1, P2)
    assert {link.knowledge_object_id for link in record.supports} == {
        f"{P1}#method:circuit-breaker:0",
        f"{P2}#method:circuit-breaker:0",
    }
    assert all(link.doi and link.doi.startswith("10.5555/") for link in record.supports)
    # No knowledge-object link, and not stored at all: recorded, never guessed.
    assert set(record.unresolved_evidence_ids) == {stray.evidence[0].id, "E-missing"}
    assert record.verification is not None
    assert record.verification.verdict is VerificationVerdict.VERIFIED
    assert (await repository.get_topic("R-1")) is not None


# --- dependency graph ---------------------------------------------------------------------


def test_reverse_traversal_reaches_only_dependents() -> None:
    only_p1 = conclusion("C-only-p1", support(P1))
    mixed = conclusion("C-mixed", support(P1, 1), support(P2))
    unrelated = conclusion("C-unrelated", support(P3))
    graph = DependencyGraphMapper(schema_version="racm1").build(TOPIC, (only_p1, mixed, unrelated))

    reached = dependents_of(graph, [P1])

    assert set(reached) == {"C-only-p1", "C-mixed"}
    assert reached["C-mixed"] == frozenset({P1})
    assert dependents_of(graph, [paper("nobody")]) == {}


def test_projection_carries_the_full_chain() -> None:
    record = conclusion("C1", support(P1))
    contradiction = Contradiction(
        id="X1",
        kind=ContradictionKind.VALUE_CONFLICT,
        description="F1 on MIMIC-III differs",
        left_object_id=record.supports[0].knowledge_object_id,
        right_object_id=f"{P2}#result:f1:0",
        left_paper_id=P1,
        right_paper_id=P2,
        left_evidence=(evidence_for(P1, "we report F1 of 0.82 on MIMIC-III here"),),
        right_evidence=(evidence_for(P2, "we report F1 of 0.41 on MIMIC-III here"),),
        detected_by="contradiction_detector",
    )
    record = record.model_copy(update={"contradictions": (contradiction,)})
    graph = DependencyGraphMapper(schema_version="racm1").build(
        TOPIC, (record,), {P1: observation(P1, IntegrityStatus.RETRACTED)}
    )

    kinds = {edge.kind for edge in graph.edges}
    assert {
        EdgeKind.SUPPORTED_BY,
        EdgeKind.GROUNDED_IN,
        EdgeKind.IN,
        EdgeKind.INVOLVES,
        EdgeKind.HAS_STATUS,
    } <= kinds
    node_kinds = {node.kind for node in graph.nodes}
    assert {
        NodeKind.CONCLUSION,
        NodeKind.KNOWLEDGE_OBJECT,
        NodeKind.SOURCE_LOCATION,
        NodeKind.PAPER,
        NodeKind.CONTRADICTION,
        NodeKind.INTEGRITY_OBSERVATION,
    } <= node_kinds
    conclusion_node = next(node for node in graph.nodes if node.kind is NodeKind.CONCLUSION)
    assert conclusion_node.properties.version == 1
    # Dependency nodes are not knowledge-graph entities.
    assert not conclusion_node.is_knowledge_node
    # Every dependency edge is grounded in a located quote.
    assert all(edge.is_trusted for edge in graph.edges if edge.kind is not EdgeKind.HAS_STATUS)


def test_corroborating_support_is_a_dependency() -> None:
    record = conclusion("C1", support(P1), support(P2, role=EvidenceRole.CORROBORATING))
    graph = DependencyGraphMapper(schema_version="racm1").build(TOPIC, (record,))

    assert set(dependents_of(graph, [P2])) == {"C1"}


@pytest.mark.parametrize("backend", ["memory", "json"])
async def test_projection_persists_through_the_graph_port(tmp_path: Path, backend: str) -> None:
    repository = (
        InMemoryGraphRepository() if backend == "memory" else JsonGraphRepository(tmp_path / "g")
    )
    graph = DependencyGraphMapper(schema_version="racm1").build(
        TOPIC, (conclusion("C1", support(P1)),)
    )
    identifier = await DependencyGraphStore(repository).write(graph)

    versions = await repository.versions()
    assert [version.identifier for version in versions] == [identifier]
    node = await repository.get_node("conclusion:C1", versions[0])
    assert node is not None and node.kind is NodeKind.CONCLUSION
    stats = await repository.stats(versions[0])
    assert stats.edges_by_kind.get("SUPPORTED_BY") == 1
