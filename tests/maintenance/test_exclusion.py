"""Retrieval-time paper exclusion: what a run may not draw on, it cannot see."""

from __future__ import annotations

import pytest

from researchagent.config.schemas import ReasoningConfig
from researchagent.container import Container
from researchagent.core.exceptions import ConfigurationError
from researchagent.schemas.workflow import ResearchState
from researchagent.workflows.reasoning_runner import ReasoningRunner
from tests.maintenance.conftest import method_object, paper, plan
from tests.maintenance.test_workflow import store_paper

P1, P2 = paper("p1"), paper("p2")


@pytest.fixture
async def two_papers(container: Container) -> None:
    for paper_id in (P1, P2):
        await store_paper(container, method_object(paper_id, "Circuit Breaker"))


async def test_excluded_papers_vanish_from_every_tool(
    container: Container, two_papers: None
) -> None:
    open_view = container.toolbox.for_run()
    closed = container.toolbox.for_run(excluded_paper_ids=frozenset({P1}))

    everything = await open_view.search_knowledge("circuit breaker")
    visible = await closed.search_knowledge("circuit breaker")
    assert {obj.paper_id for obj in everything.objects} == {P1, P2}
    assert {obj.paper_id for obj in visible.objects} == {P2}

    ids = tuple(item.id for obj in everything.objects for item in obj.evidence)
    assert len(await open_view.get_provenance(ids)) == 2
    assert len(await closed.get_provenance(ids)) == 1

    assert not (await closed.get_paper_context(P1)).found
    assert closed.for_agent("verification", 0).excluded_paper_ids == frozenset({P1})


async def test_pruned_bundle_gets_its_own_id(container: Container, two_papers: None) -> None:
    full = await container.toolbox.for_run().build_bundle("circuit breaker")
    pruned = await container.toolbox.for_run(excluded_paper_ids=frozenset({P1})).build_bundle(
        "circuit breaker"
    )

    assert {obj.paper_id for obj in full.knowledge_objects} == {P1, P2}
    assert {obj.paper_id for obj in pruned.knowledge_objects} == {P2}
    assert all(item.paper_id == P2 for item in pruned.evidence)
    # The original bundle, which earlier citations point at, is not overwritten.
    assert pruned.id != full.id
    assert await container.bundle_repository.get(full.id) == full
    assert pruned.statistics.distinct_papers == 1


async def test_exclusion_without_a_factory_is_refused(container: Container) -> None:
    runner = ReasoningRunner(
        lambda *_: None,  # type: ignore[arg-type,return-value]
        container.bundle_repository,
        ReasoningConfig(),
    )
    state = ResearchState(goal="Overload mitigation in systems", plan=plan())

    with pytest.raises(ConfigurationError):
        await runner.run(state, excluded_paper_ids=frozenset({P1}))
