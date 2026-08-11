"""The graph must survive the process that built it.

Runs A and B of the v0.9 completion experiment reported `graph_version: null` because the
in-memory backend does not outlive its process — a graph was built, then vanished before
anything could query it. These tests pin the persistent path: build, close, reopen, query.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.core.exceptions import GraphStoreError
from researchagent.models.graph import EdgeKind, KnowledgeGraph, NodeKind
from researchagent.models.knowledge import PaperKnowledge
from researchagent.repositories.graph_repository import JsonGraphRepository
from researchagent.services.graph.queries import GraphQueries
from tests.graph.test_graph import build


@pytest.fixture
def graphs_dir(tmp_path: Path) -> Path:
    return tmp_path / "graphs"


class TestPersistence:
    async def test_a_generation_survives_a_new_repository_instance(
        self, graphs_dir: Path, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        """The exact gap: build here, read there, as a second process would."""
        graph = build([paper_a, paper_b])
        writer = JsonGraphRepository(graphs_dir)
        await writer.write_graph(graph)
        await writer.aclose()

        reader = JsonGraphRepository(graphs_dir)
        versions = await reader.versions()

        assert len(versions) == 1
        assert versions[0].identifier == graph.version.identifier
        stats = await reader.stats(versions[0])
        assert stats.nodes == len(graph.nodes)
        assert stats.edges == len(graph.edges)

    async def test_a_reopened_graph_answers_domain_queries(
        self, graphs_dir: Path, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        graph = build([paper_a, paper_b])
        await JsonGraphRepository(graphs_dir).write_graph(graph)

        reader = JsonGraphRepository(graphs_dir)
        version = (await reader.versions())[0]
        found = await GraphQueries(reader).datasets_for_method("RAG", version)

        assert [entity.node.name for entity in found] == ["MIMIC-III"]
        assert found[0].citations, "provenance survives the round trip to disk"

    async def test_neighbours_traverse_a_reloaded_graph(
        self, graphs_dir: Path, paper_a: PaperKnowledge
    ) -> None:
        graph = build([paper_a])
        await JsonGraphRepository(graphs_dir).write_graph(graph)

        reader = JsonGraphRepository(graphs_dir)
        version = (await reader.versions())[0]
        method = (await reader.find_nodes(kind=NodeKind.METHOD, version=version))[0]
        subgraph = await reader.neighbours(method.id, version, depth=1)

        assert subgraph.centre_id == method.id
        assert subgraph.nodes and subgraph.edges

    async def test_edges_between_survive_the_round_trip(
        self, graphs_dir: Path, paper_a: PaperKnowledge
    ) -> None:
        graph = build([paper_a])
        await JsonGraphRepository(graphs_dir).write_graph(graph)

        reader = JsonGraphRepository(graphs_dir)
        version = (await reader.versions())[0]
        method = (await reader.find_nodes(kind=NodeKind.METHOD, version=version))[0]
        dataset = (await reader.find_nodes(kind=NodeKind.DATASET, version=version))[0]

        edges = await reader.edges_between(method.id, dataset.id, version)

        assert edges
        assert edges[0].kind is EdgeKind.EVALUATED_ON
        assert edges[0].provenance.cite()


class TestIdempotencyOnDisk:
    async def test_rebuilding_replaces_rather_than_accumulates(
        self, graphs_dir: Path, paper_a: PaperKnowledge
    ) -> None:
        repository = JsonGraphRepository(graphs_dir)

        first = await repository.write_graph(build([paper_a]))
        second = await repository.write_graph(build([paper_a]))

        assert (first.nodes, first.edges) == (second.nodes, second.edges)
        assert len(list(graphs_dir.glob("*.json"))) == 1

    async def test_two_corpora_produce_two_files(
        self, graphs_dir: Path, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        repository = JsonGraphRepository(graphs_dir)

        await repository.write_graph(build([paper_a]))
        await repository.write_graph(build([paper_a, paper_b]))

        assert len(await repository.versions()) == 2

    async def test_deleting_one_generation_leaves_the_other(
        self, graphs_dir: Path, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        repository = JsonGraphRepository(graphs_dir)
        small = build([paper_a])
        await repository.write_graph(small)
        await repository.write_graph(build([paper_a, paper_b]))

        assert await repository.delete_version(small.version)
        assert len(await repository.versions()) == 1


class TestFailureModes:
    async def test_an_absent_generation_reads_as_empty_not_as_an_error(
        self, graphs_dir: Path, paper_a: PaperKnowledge
    ) -> None:
        repository = JsonGraphRepository(graphs_dir)
        version = build([paper_a]).version

        assert await repository.versions() == ()
        assert (await repository.stats(version)).nodes == 0
        assert await repository.get_node("anything", version) is None

    async def test_an_unreadable_generation_raises_rather_than_looking_empty(
        self, graphs_dir: Path, paper_a: PaperKnowledge
    ) -> None:
        """ "Corrupt" and "never built" must not be the same answer."""
        graph = build([paper_a])
        repository = JsonGraphRepository(graphs_dir)
        await repository.write_graph(graph)
        (graphs_dir / f"{graph.version.identifier}.json").write_text("{ not json")

        fresh = JsonGraphRepository(graphs_dir)
        with pytest.raises(GraphStoreError) as caught:
            await fresh.stats(graph.version)

        assert "Rebuild" in (caught.value.remedy or "")

    async def test_health_reports_a_writable_directory(self, graphs_dir: Path) -> None:
        assert await JsonGraphRepository(graphs_dir).health() is True


class TestBackendParity:
    async def test_the_json_backend_satisfies_the_same_port(self, graphs_dir: Path) -> None:
        """No second graph abstraction: this is one more adapter behind one port."""
        from researchagent.core.interfaces.graph_repository import GraphRepository
        from researchagent.integrations.memory_graph import InMemoryGraphRepository

        assert issubclass(JsonGraphRepository, GraphRepository)
        assert issubclass(InMemoryGraphRepository, GraphRepository)

    async def test_both_backends_agree_on_a_generation(
        self, graphs_dir: Path, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        from researchagent.integrations.memory_graph import InMemoryGraphRepository

        graph: KnowledgeGraph = build([paper_a, paper_b])
        memory = InMemoryGraphRepository()
        disk = JsonGraphRepository(graphs_dir)

        in_memory = await memory.write_graph(graph)
        on_disk = await disk.write_graph(graph)

        assert (in_memory.nodes, in_memory.edges) == (on_disk.nodes, on_disk.edges)
        assert in_memory.nodes_by_kind == on_disk.nodes_by_kind
