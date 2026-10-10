"""The maintenance workflow end to end, on the real container with a fake LLM and an
offline scripted registry: incremental vs full rerun, cost aggregation, API endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.agents.reviewer.schemas import CritiqueDraft
from researchagent.agents.verification.schemas import VerificationDraft
from researchagent.api.app import create_app
from researchagent.config.schemas import FullRerunSettings
from researchagent.container import Container
from researchagent.core.events import EventType, LLMCallPayload
from researchagent.core.interfaces.integrity import LookupOutcome, RegistryLookup
from researchagent.core.interfaces.llm import TokenUsage
from researchagent.core.validation import Confidence, ValidationResult
from researchagent.models.evidence import EvidenceLink, EvidenceRecord, PaperEvidence
from researchagent.models.integrity import IntegrityStatus, UpdateNotice
from researchagent.models.knowledge import KnowledgeObject, PaperKnowledge
from researchagent.models.maintenance import (
    ChangeType,
    ConclusionState,
    MaintainedTopic,
    MaintenanceMode,
    SupportLink,
)
from researchagent.models.reasoning import Citation, FindingStatus, ResearchFinding
from researchagent.schemas.knowledge import ValidatedKnowledge
from researchagent.schemas.maintenance import MaintenanceRequest
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.workflows.maintenance_runner import MaintenanceRunner
from tests.conftest import FakeLLMProvider, ScriptedIntegrityRegistry
from tests.maintenance.conftest import conclusion, method_object, paper, plan

P1, P2, P3, P4 = paper("p1"), paper("p2"), paper("p3"), paper("p4")
TOPIC = "R-topic"


async def store_paper(container: Container, *objects: KnowledgeObject) -> None:
    paper_id = objects[0].paper_id
    await container.knowledge_repository.save(
        ValidatedKnowledge(
            value=PaperKnowledge(paper_id=paper_id, document_sha256="sha", objects=objects),
            validation=ValidationResult.passed(
                validator="test",
                subject_id=paper_id,
                subject_type="PaperKnowledge",
                confidence=Confidence(score=0.9),
            ),
        )
    )
    await container.evidence_repository.save_paper(
        PaperEvidence(
            paper_id=paper_id,
            document_sha256="sha",
            records=tuple(
                EvidenceRecord(
                    evidence=item,
                    paper_id=paper_id,
                    document_sha256="sha",
                    links=(
                        EvidenceLink(
                            evidence_id=item.id,
                            knowledge_object_id=obj.id,
                            knowledge_kind=obj.kind,
                            linked_by="test",
                        ),
                    ),
                )
                for obj in objects
                for item in obj.evidence
            ),
        )
    )


def link_to(obj: KnowledgeObject) -> SupportLink:
    evidence = obj.evidence[0]
    return SupportLink(
        evidence_id=evidence.id,
        knowledge_object_id=obj.id,
        paper_id=obj.paper_id,
        location=evidence.location,
        bundle_id="B-orig",
        doi=obj.paper_id.removeprefix("doi:"),
        quote=evidence.quote or "",
    )


def answer(doi: str, *types: str) -> RegistryLookup:
    return RegistryLookup(
        registry="crossref",
        doi=doi,
        outcome=LookupOutcome.FOUND,
        detail="scripted",
        notices=tuple(
            UpdateNotice(
                update_type=kind,
                relation="updated-by",
                source="retraction-watch",
                notice_doi=f"10.5555/notice-{doi.rsplit('/', 1)[-1]}",
                notice_date="2026-09-30",
            )
            for kind in types
        ),
    )


@pytest.fixture
async def seeded(container: Container) -> dict[str, KnowledgeObject]:
    objects = {pid: method_object(pid, "Circuit Breaker") for pid in (P1, P2, P3)}
    for obj in objects.values():
        await store_paper(container, obj)
    conclusions = container.maintenance.conclusions
    await conclusions.save_topic(
        MaintainedTopic(
            topic_id=TOPIC,
            goal="Overload mitigation in systems",
            plan=plan(),
            known_paper_ids=(P1, P2, P3),
        )
    )
    for record in (
        conclusion("C-solo", link_to(objects[P1]), topic_id=TOPIC),
        conclusion(
            "C-mixed",
            link_to(objects[P1]),
            link_to(objects[P2]),
            link_to(objects[P3]),
            topic_id=TOPIC,
        ),
        conclusion("C-other", link_to(objects[P2]), link_to(objects[P3]), topic_id=TOPIC),
    ):
        await conclusions.append(record)

    registry = container.maintenance.registry
    assert isinstance(registry, ScriptedIntegrityRegistry)
    registry.answers = {
        "10.5555/p1": answer("10.5555/p1", "retraction"),
        "10.5555/p2": answer("10.5555/p2"),
        "10.5555/p3": answer("10.5555/p3"),
    }
    return objects


def script_llm(fake_provider: FakeLLMProvider, objects: dict[str, KnowledgeObject]) -> None:
    fake_provider.usage = TokenUsage(prompt_tokens=100, completion_tokens=20)
    fake_provider.structured_sequence = [
        VerificationDraft(
            verdict="verified",
            reasoning="P2 and P3 both state it",
            supporting_evidence_ids=[objects[P2].evidence[0].id, objects[P3].evidence[0].id],
        ),
        CritiqueDraft(critique="no further concerns"),
    ]


async def test_incremental_run_propagates_only_to_dependents(
    container: Container, seeded: dict[str, KnowledgeObject], fake_provider: FakeLLMProvider
) -> None:
    script_llm(fake_provider, seeded)
    runner = container.maintenance.runner

    diff = await runner.run(TOPIC)

    assert diff.mode is MaintenanceMode.INCREMENTAL
    assert diff.counts["invalidated"] == 1 and diff.counts["kept"] == 1
    assert sum(diff.counts.values()) == 2
    assert diff.affected_conclusions == 2 and diff.total_conclusions == 3
    assert [item.paper_id for item in diff.integrity_changes] == [P1]
    assert diff.integrity_changes[0].status is IntegrityStatus.RETRACTED

    solo = diff.change_for("C-solo")
    assert solo is not None and solo.change_type is ChangeType.INVALIDATED
    assert solo.triggers[0].notice_doi == "10.5555/notice-p1"
    mixed = diff.change_for("C-mixed")
    assert mixed is not None and mixed.change_type is ChangeType.KEPT
    assert mixed.removed_evidence_ids == (seeded[P1].evidence[0].id,)
    assert diff.change_for("C-other") is None
    # Untouched means untouched: no new version, no verifier call for it.
    assert len(await container.maintenance.conclusions.history("C-other")) == 1
    assert diff.review is not None  # the re-verified conclusion went past the reviewer

    # Tokens and cost come from the existing per-run usage tracking.
    assert diff.usage.llm_calls == 2
    assert diff.usage.total_tokens == 240
    assert diff.usage.estimated_cost_usd is None  # the fake provider has no pricing
    assert diff.dependency_graph_version is not None


async def test_second_run_with_nothing_new_is_an_empty_diff(
    container: Container, seeded: dict[str, KnowledgeObject], fake_provider: FakeLLMProvider
) -> None:
    script_llm(fake_provider, seeded)
    runner = container.maintenance.runner
    registry = container.maintenance.registry
    assert isinstance(registry, ScriptedIntegrityRegistry)

    await runner.run(TOPIC)
    lookups_after_first = len(registry.lookups)
    second = await runner.run(TOPIC)

    assert second.is_empty
    assert second.counts == {kind.value: 0 for kind in ChangeType}
    assert second.usage.llm_calls == 0 and second.usage.estimated_cost_usd == 0.0
    assert len(registry.lookups) == lookups_after_first  # cached observations reused


async def test_lookup_failure_changes_nothing(
    container: Container, seeded: dict[str, KnowledgeObject]
) -> None:
    registry = container.maintenance.registry
    assert isinstance(registry, ScriptedIntegrityRegistry)
    registry.answers = {}  # every lookup fails

    diff = await container.maintenance.runner.run(TOPIC)

    assert diff.is_empty
    assert len(diff.lookup_failures) == 3


def stub_generator(
    container: Container, objects: dict[str, KnowledgeObject], seen: list[frozenset[str]]
) -> Any:
    """Stands in for the reasoning loop: one accepted finding, plus its LLM usage event."""

    async def generate(
        topic: MaintainedTopic, run_id: str, excluded: frozenset[str]
    ) -> ReasoningSession:
        seen.append(excluded)
        await container.event_bus.emit(
            EventType.LLM_CALL_COMPLETED,
            LLMCallPayload(
                alias="reasoning",
                provider="fake",
                model="fake-model",
                latency_ms=5.0,
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                estimated_cost_usd=0.01,
            ),
            run_id=run_id,
        )
        cited = [obj for pid, obj in objects.items() if pid not in excluded]
        bundle_id = "B-regen"
        from tests.maintenance.test_dependency_graph import bundle_with

        bundle = bundle_with(*((obj.paper_id, obj.name) for obj in cited)).model_copy(
            update={"id": bundle_id}
        )
        await container.bundle_repository.save(bundle)
        finding = ResearchFinding(
            question_id="RQ1",
            statement="Circuit breakers shed load across the remaining papers in this corpus.",
            citations=(
                Citation(
                    bundle_id=bundle_id, evidence_ids=tuple(e.evidence.id for e in bundle.evidence)
                ),
            ),
            status=FindingStatus.VERIFIED,
            produced_by="reasoning",
        )
        return ReasoningSession(findings=(finding,))

    return generate


@pytest.mark.parametrize(
    ("exclude", "expected_excluded"),
    [((), frozenset()), ((IntegrityStatus.RETRACTED,), frozenset({P1}))],
    ids=["B1-no-filter", "B2-retrieval-filter"],
)
async def test_full_rerun_mode_regenerates_and_accounts_cost(
    container: Container,
    seeded: dict[str, KnowledgeObject],
    exclude: tuple[IntegrityStatus, ...],
    expected_excluded: frozenset[str],
) -> None:
    service = container.maintenance.service
    service._config = service.config.model_copy(  # per-test config variant
        update={"full_rerun": FullRerunSettings(exclude_statuses=exclude)}
    )
    seen: list[frozenset[str]] = []
    runner = MaintenanceRunner(
        service, container.maintenance.agent_for, stub_generator(container, seeded, seen)
    )

    diff = await runner.run(TOPIC, MaintenanceRequest(mode=MaintenanceMode.FULL_RERUN))

    assert seen == [expected_excluded]
    assert diff.mode is MaintenanceMode.FULL_RERUN
    assert diff.counts["new"] == 1 and sum(diff.counts.values()) == 1
    assert set(diff.superseded_conclusion_ids) == {"C-solo", "C-mixed", "C-other"}
    assert diff.regenerated_topic_id is not None
    regenerated = await service.records(diff.regenerated_topic_id)
    assert len(regenerated) == 1
    # Original conclusions are not rewritten by a regeneration: no rule fired on them.
    for conclusion_id in ("C-solo", "C-mixed", "C-other"):
        assert len(await service.history(conclusion_id)) == 1
    assert diff.usage.total_tokens == 1200
    assert diff.usage.estimated_cost_usd == pytest.approx(0.01)


async def test_new_paper_adds_support_and_new_conclusions(
    container: Container, seeded: dict[str, KnowledgeObject], fake_provider: FakeLLMProvider
) -> None:
    newcomer = method_object(P4, "Circuit Breaker")
    await store_paper(container, newcomer)
    fake_provider.structured_sequence = [
        # C-mixed: re-verified on P2 and P3.
        VerificationDraft(
            verdict="verified",
            supporting_evidence_ids=[seeded[P2].evidence[0].id, seeded[P3].evidence[0].id],
        ),
        # C-solo: its only founding paper is retracted, but the new paper corroborates it,
        # so it is re-verified on that corroboration instead of being invalidated.
        VerificationDraft(verdict="verified", supporting_evidence_ids=[newcomer.evidence[0].id]),
        CritiqueDraft(),
    ]
    seen: list[frozenset[str]] = []
    runner = MaintenanceRunner(
        container.maintenance.service,
        container.maintenance.agent_for,
        stub_generator(container, {**seeded, P4: newcomer}, seen),
    )

    diff = await runner.run(TOPIC)

    assert diff.new_paper_ids == (P4,)
    assert seen == [frozenset({P1})]  # the retracted paper is hidden from discovery
    other = diff.change_for("C-other")
    assert other is not None and other.change_type is ChangeType.KEPT
    assert other.added_evidence_ids == (newcomer.evidence[0].id,)
    assert diff.counts["new"] == 1
    new = next(change for change in diff.changes if change.change_type is ChangeType.NEW)
    assert new.before_version is None
    assert [trigger.paper_id for trigger in new.triggers] == [P4]
    solo = diff.change_for("C-solo")
    assert solo is not None and solo.change_type is ChangeType.WEAKENED
    assert any("support retained 0.00" in reason for reason in solo.reasons)


# --- API ---------------------------------------------------------------------------------


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


async def test_api_trigger_latest_diff_and_history(
    client: AsyncClient, seeded: dict[str, KnowledgeObject], fake_provider: FakeLLMProvider
) -> None:
    script_llm(fake_provider, seeded)

    before = await client.get(f"/maintenance/topics/{TOPIC}/diff")
    assert before.status_code == 404 and before.json()["error"]["code"] == "diff_not_found"

    topics = await client.get("/maintenance/topics")
    assert [item["topic_id"] for item in topics.json()] == [TOPIC]

    triggered = await client.post(f"/maintenance/topics/{TOPIC}/run", json={"mode": "incremental"})
    assert triggered.status_code == 200
    body = triggered.json()
    assert body["counts"]["invalidated"] == 1 and body["mode"] == "incremental"

    latest = await client.get(f"/maintenance/topics/{TOPIC}/diff")
    assert latest.status_code == 200 and latest.json()["id"] == body["id"]

    history = await client.get("/maintenance/conclusions/C-solo/history")
    versions = history.json()["versions"]
    assert [item["version"] for item in versions] == [1, 2]
    assert [item["state"] for item in versions] == [
        ConclusionState.ACTIVE.value,
        ConclusionState.INVALIDATED.value,
    ]


async def test_api_unknown_topic_and_conclusion_are_404(client: AsyncClient) -> None:
    run = await client.post("/maintenance/topics/nope/run")
    diff = await client.get("/maintenance/topics/nope/diff")
    history = await client.get("/maintenance/conclusions/nope/history")

    assert run.status_code == 404 and run.json()["error"]["code"] == "topic_not_found"
    assert diff.status_code == 404 and diff.json()["error"]["code"] == "topic_not_found"
    assert history.status_code == 404
    assert history.json()["error"]["code"] == "conclusion_not_found"
