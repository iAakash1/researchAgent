"""Knowledge graph tests.

The graph is a *derived index*. Every property tested here follows from that: it must be
reconstructible, idempotent, provenance-carrying, and incapable of asserting anything the
source repositories do not already support.

No test needs a running Neo4j — the in-memory repository implements the same port.
"""

from __future__ import annotations

import pytest

from researchagent.core.exceptions import GraphNotBuiltError
from researchagent.integrations.memory_graph import InMemoryGraphRepository
from researchagent.models.graph import (
    EdgeKind,
    GraphEdge,
    GraphProvenance,
    KnowledgeGraph,
    NodeKind,
    edge_id_for,
    node_id_for_knowledge,
    node_id_for_paper,
)
from researchagent.models.knowledge import KnowledgeKind, PaperKnowledge
from researchagent.models.paper import Paper
from researchagent.services.evidence import ContradictionDetector
from researchagent.services.graph.builder import GraphBuilder
from researchagent.services.graph.mapper import GraphMapper
from researchagent.services.graph.queries import GraphQueries
from researchagent.services.graph.validator import GraphValidator


def build(
    knowledge: list[PaperKnowledge], papers: dict[str, Paper] | None = None
) -> KnowledgeGraph:
    detector = ContradictionDetector()
    every_object = tuple(obj for item in knowledge for obj in item.objects)
    return GraphMapper().build(
        knowledge, papers=papers or {}, contradictions=detector.detect(every_object)
    )


class TestMapping:
    """Knowledge in, graph out. Nothing model-generated enters here."""

    def test_every_knowledge_object_becomes_a_node(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])

        assert len(graph.nodes_of(NodeKind.METHOD)) == 1
        assert len(graph.nodes_of(NodeKind.DATASET)) == 1
        assert len(graph.nodes_of(NodeKind.METRIC)) == 1
        assert len(graph.nodes_of(NodeKind.RESULT)) == 1
        assert len(graph.nodes_of(NodeKind.PAPER)) == 1

    def test_paper_node_carries_catalogue_metadata(
        self, paper_a: PaperKnowledge, papers: dict[str, Paper]
    ) -> None:
        graph = build([paper_a], papers)

        paper_node = graph.node(node_id_for_paper("manual:01"))
        assert paper_node is not None
        assert paper_node.name == "Retrieval-Augmented Clinical QA"
        assert paper_node.properties.year == 2023

    def test_relations_become_typed_edges(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])

        assert len(graph.edges_of(EdgeKind.EVALUATED_ON)) == 1
        assert len(graph.edges_of(EdgeKind.MEASURED_BY)) == 1
        assert len(graph.edges_of(EdgeKind.PRODUCED_BY)) == 1

    def test_paper_mentions_every_fact_it_states(self, paper_a: PaperKnowledge) -> None:
        mentions = build([paper_a]).edges_of(EdgeKind.MENTIONS)

        assert len(mentions) == len(paper_a.objects)
        assert all(edge.source_id == node_id_for_paper("manual:01") for edge in mentions)

    def test_the_graph_adds_no_facts_of_its_own(self, paper_a: PaperKnowledge) -> None:
        """Node count is bounded by the knowledge: mapping derives, it does not invent."""
        graph = build([paper_a])

        knowledge_nodes = [node for node in graph.nodes if node.kind is not NodeKind.PAPER]
        assert len(knowledge_nodes) <= len(paper_a.objects)


class TestStableIdentity:
    """Ids are domain-derived, never storage-derived."""

    def test_the_same_entity_in_two_papers_becomes_one_node(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        """The whole point of name-derived ids: cross-paper questions become answerable."""
        graph = build([paper_a, paper_b])

        datasets = graph.nodes_of(NodeKind.DATASET)
        assert len(datasets) == 1, "MIMIC-III is one dataset, not one per paper"
        assert set(datasets[0].paper_ids) == {"manual:01", "manual:02"}

    def test_merged_node_keeps_every_originating_object(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        method = build([paper_a, paper_b]).nodes_of(NodeKind.METHOD)[0]

        assert len(method.evidence_ids) >= 2, "merging must not discard either side's evidence"

    def test_results_stay_scoped_to_their_paper(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        """Merging results would turn a disagreement into a self-loop and delete it."""
        results = build([paper_a, paper_b]).nodes_of(NodeKind.RESULT)

        assert len(results) == 2, "two papers measuring the same thing are two claims"
        assert {node.paper_ids[0] for node in results} == {"manual:01", "manual:02"}

    def test_shared_entities_still_merge(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        graph = build([paper_a, paper_b])

        assert len(graph.nodes_of(NodeKind.METHOD)) == 1
        assert len(graph.nodes_of(NodeKind.METRIC)) == 1

    def test_node_ids_are_deterministic_across_processes(self) -> None:
        first = node_id_for_knowledge(KnowledgeKind.DATASET, "MIMIC-III")
        second = node_id_for_knowledge(KnowledgeKind.DATASET, "mimic-iii")

        assert first == second, "identity is the normalised name, not its casing"

    def test_edge_ids_are_derived_from_their_endpoints(self) -> None:
        edge_id = edge_id_for(EdgeKind.EVALUATED_ON, "method:rag:x", "dataset:mimic:y")

        assert edge_id == "method:rag:x--EVALUATED_ON--dataset:mimic:y"


class TestValidation:
    """Invalid data is rejected with a reason, never silently dropped."""

    def test_an_edge_with_no_provenance_is_rejected(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])
        method = graph.nodes_of(NodeKind.METHOD)[0]
        dataset = graph.nodes_of(NodeKind.DATASET)[0]
        unprovenanced = GraphEdge(
            id=edge_id_for(EdgeKind.EVALUATED_ON, method.id, dataset.id) + "-x",
            kind=EdgeKind.EVALUATED_ON,
            source_id=method.id,
            target_id=dataset.id,
            provenance=GraphProvenance(derived_by="a_model_that_was_asked_nicely"),
        )
        polluted = KnowledgeGraph(
            version=graph.version, nodes=graph.nodes, edges=(*graph.edges, unprovenanced)
        )

        report = GraphValidator().validate(polluted)

        assert unprovenanced.id not in {edge.id for edge in report.accepted_edges}
        assert report.rejection_reasons["no_provenance"] == 1

    def test_an_edge_pointing_at_a_missing_node_is_rejected(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])
        dangling = GraphEdge(
            id="ghost--EVALUATED_ON--also-ghost",
            kind=EdgeKind.EVALUATED_ON,
            source_id="method:ghost:0000000000",
            target_id=graph.nodes_of(NodeKind.DATASET)[0].id,
            provenance=GraphProvenance(
                derived_by="test",
                evidence_ids=("e1",),
                locations=(paper_a.objects[0].evidence[0].location,),
            ),
        )
        polluted = KnowledgeGraph(
            version=graph.version, nodes=graph.nodes, edges=(*graph.edges, dangling)
        )

        report = GraphValidator().validate(polluted)

        assert dangling.id not in {edge.id for edge in report.accepted_edges}
        assert sum(report.rejection_reasons.values()) >= 1

    def test_an_edge_between_the_wrong_kinds_is_rejected(self, paper_a: PaperKnowledge) -> None:
        """EVALUATED_ON runs Method -> Dataset. Metric -> Dataset is meaningless."""
        graph = build([paper_a])
        metric = graph.nodes_of(NodeKind.METRIC)[0]
        dataset = graph.nodes_of(NodeKind.DATASET)[0]
        mistyped = GraphEdge(
            id=edge_id_for(EdgeKind.EVALUATED_ON, metric.id, dataset.id),
            kind=EdgeKind.EVALUATED_ON,
            source_id=metric.id,
            target_id=dataset.id,
            provenance=GraphProvenance(
                derived_by="test",
                evidence_ids=("e1",),
                locations=(paper_a.objects[0].evidence[0].location,),
            ),
        )
        polluted = KnowledgeGraph(
            version=graph.version, nodes=graph.nodes, edges=(*graph.edges, mistyped)
        )

        report = GraphValidator().validate(polluted)

        assert mistyped.id not in {edge.id for edge in report.accepted_edges}

    def test_a_clean_graph_is_accepted_whole(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])

        report = GraphValidator().validate(graph)

        assert report.result.success
        assert len(report.accepted_edges) == len(graph.edges)
        assert not report.rejected_edges

    def test_rejections_are_reported_with_reasons_not_discarded(
        self, paper_a: PaperKnowledge
    ) -> None:
        graph = build([paper_a])
        bad = GraphEdge(
            id="x--LIMITS--y",
            kind=EdgeKind.LIMITS,
            source_id=graph.nodes_of(NodeKind.METHOD)[0].id,
            target_id=graph.nodes_of(NodeKind.DATASET)[0].id,
            provenance=GraphProvenance(derived_by="test"),
        )
        report = GraphValidator().validate(
            KnowledgeGraph(version=graph.version, nodes=graph.nodes, edges=(bad,))
        )

        assert report.rejected_edges
        assert all(rejected.reason for rejected in report.rejected_edges)


class TestProvenance:
    """A relationship nobody can check is not knowledge."""

    def test_every_persisted_edge_is_traceable_to_a_quote(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])
        report = GraphValidator().validate(graph)

        assert report.accepted_edges
        for edge in report.accepted_edges:
            assert edge.is_trusted
            assert edge.provenance.evidence_ids
            assert edge.provenance.locations

    def test_citations_address_a_page_and_paragraph(self, paper_a: PaperKnowledge) -> None:
        edge = build([paper_a]).edges_of(EdgeKind.EVALUATED_ON)[0]

        citation = edge.provenance.cite()[0]
        assert "manual:01" in citation

    def test_provenance_coverage_is_a_share_not_a_count(self, paper_a: PaperKnowledge) -> None:
        graph = build([paper_a])

        assert 0.0 <= graph.provenance_coverage <= 1.0

    async def test_provenance_survives_the_round_trip_to_storage(
        self, paper_a: PaperKnowledge
    ) -> None:
        graph = build([paper_a])
        repository = InMemoryGraphRepository()
        await repository.write_graph(graph)
        queries = GraphQueries(repository)

        method = graph.nodes_of(NodeKind.METHOD)[0]
        dataset = graph.nodes_of(NodeKind.DATASET)[0]
        citations = await queries.provenance_for(method.id, dataset.id, graph.version)

        assert citations, "the edge must remain citable after being stored"


class TestContradictions:
    """Both sides stay queryable. The graph reports disagreement, it does not settle it."""

    def test_conflicting_results_produce_a_contradicts_edge(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        graph = build([paper_a, paper_b])

        assert graph.edges_of(EdgeKind.CONTRADICTS), "0.82 vs 0.41 on the same metric+dataset"

    async def test_both_sides_are_returned_and_neither_is_dropped(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        graph = build([paper_a, paper_b])
        repository = InMemoryGraphRepository()
        await repository.write_graph(graph)

        pairs = await GraphQueries(repository).contradictions(graph.version)

        assert pairs
        assert pairs[0].left.id != pairs[0].right.id
        assert len(set(pairs[0].papers)) == 2, "a cross-paper disagreement names both papers"

    def test_a_contradiction_edge_carries_its_own_provenance(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        edge = build([paper_a, paper_b]).edges_of(EdgeKind.CONTRADICTS)[0]

        assert edge.is_trusted, "disagreement is a claim, and claims need evidence"


class TestVersioning:
    """Generations are identified, never mixed."""

    def test_the_same_corpus_yields_the_same_version(self, paper_a: PaperKnowledge) -> None:
        assert build([paper_a]).version.identifier == build([paper_a]).version.identifier

    def test_a_different_corpus_yields_a_different_version(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        assert build([paper_a]).version.identifier != build([paper_a, paper_b]).version.identifier

    def test_a_schema_bump_yields_a_different_version(self, paper_a: PaperKnowledge) -> None:
        v1 = GraphMapper(schema_version="1").build([paper_a])
        v2 = GraphMapper(schema_version="2").build([paper_a])

        assert v1.version.identifier != v2.version.identifier

    async def test_generations_do_not_leak_into_each_other(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        repository = InMemoryGraphRepository()
        small = build([paper_a])
        large = build([paper_a, paper_b])
        await repository.write_graph(small)
        await repository.write_graph(large)

        assert (await repository.stats(small.version)).nodes < (
            await repository.stats(large.version)
        ).nodes
        assert len(await repository.versions()) == 2

    async def test_deleting_a_generation_leaves_the_other_intact(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        repository = InMemoryGraphRepository()
        small, large = build([paper_a]), build([paper_a, paper_b])
        await repository.write_graph(small)
        await repository.write_graph(large)

        assert await repository.delete_version(small.version)
        assert (await repository.stats(large.version)).nodes > 0


class TestIdempotency:
    """Rebuilding is safe: the graph is derived, so it must be re-derivable."""

    async def test_building_twice_does_not_duplicate(self, paper_a: PaperKnowledge) -> None:
        repository = InMemoryGraphRepository()
        graph = build([paper_a])

        first = await repository.write_graph(graph)
        second = await repository.write_graph(build([paper_a]))

        assert (first.nodes, first.edges) == (second.nodes, second.edges)
        assert len(await repository.versions()) == 1

    def test_mapping_is_deterministic(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> None:
        first, second = build([paper_a, paper_b]), build([paper_a, paper_b])

        assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
        assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


class TestQueries:
    """Domain questions, answered with citations attached."""

    @pytest.fixture
    async def repository(
        self, paper_a: PaperKnowledge, paper_b: PaperKnowledge
    ) -> InMemoryGraphRepository:
        repository = InMemoryGraphRepository()
        await repository.write_graph(build([paper_a, paper_b]))
        return repository

    @pytest.fixture
    def graph(self, paper_a: PaperKnowledge, paper_b: PaperKnowledge) -> KnowledgeGraph:
        return build([paper_a, paper_b])

    async def test_datasets_for_a_method(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        found = await GraphQueries(repository).datasets_for_method("RAG", graph.version)

        assert [entity.node.name for entity in found] == ["MIMIC-III"]
        assert found[0].citations, "an answer without a citation is an assertion"

    async def test_methods_for_a_dataset(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        found = await GraphQueries(repository).methods_for_dataset("MIMIC-III", graph.version)

        assert [entity.node.name for entity in found] == ["RAG"]

    async def test_metrics_for_a_method_traverses_two_hops(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        """Method <- PRODUCED_BY - Result - MEASURED_BY -> Metric. A flat index cannot do this."""
        found = await GraphQueries(repository).metrics_for_method("RAG", graph.version)

        assert [entity.node.name for entity in found] == ["F1"]

    async def test_papers_using_an_entity(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        found = await GraphQueries(repository).papers_using("MIMIC-III", graph.version)

        assert {node.id for node in found} == {
            node_id_for_paper("manual:01"),
            node_id_for_paper("manual:02"),
        }

    async def test_papers_sharing_a_dataset(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        shared = await GraphQueries(repository).entities_across_papers(
            graph.version, kind=NodeKind.DATASET
        )

        assert shared
        assert shared[0].node.name == "MIMIC-III"
        assert shared[0].share_count == 2, "both papers must appear on the shared dataset"

    async def test_neighbours_returns_a_subgraph_not_a_node_list(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        method = graph.nodes_of(NodeKind.METHOD)[0]

        subgraph = await repository.neighbours(method.id, graph.version, depth=1)

        assert subgraph.centre_id == method.id
        assert subgraph.nodes and subgraph.edges

    async def test_an_unknown_entity_returns_empty_rather_than_raising(
        self, repository: InMemoryGraphRepository, graph: KnowledgeGraph
    ) -> None:
        found = await GraphQueries(repository).datasets_for_method("NotAMethod", graph.version)

        assert found == ()


class TestBuilder:
    """End to end, through the repository port."""

    async def test_build_reports_what_it_did(
        self, knowledge_repository: object, paper_a: PaperKnowledge
    ) -> None:
        from researchagent.core.validation import Confidence, ValidationResult
        from researchagent.schemas.knowledge import ValidatedKnowledge

        repo = knowledge_repository
        await repo.save(  # type: ignore[attr-defined]
            ValidatedKnowledge(
                value=paper_a,
                validation=ValidationResult.passed(
                    validator="test",
                    subject_id=paper_a.paper_id,
                    subject_type="PaperKnowledge",
                    confidence=Confidence(score=0.9),
                ),
            )
        )
        graph_repository = InMemoryGraphRepository()
        builder = GraphBuilder(repo, graph_repository)  # type: ignore[arg-type]

        report = await builder.build()

        assert report.succeeded
        assert report.papers == 1
        assert report.nodes > 0
        assert report.edges_accepted > 0
        assert report.provenance_coverage == 1.0

    async def test_building_with_no_knowledge_reports_failure_rather_than_an_empty_graph(
        self, knowledge_repository: object
    ) -> None:
        builder = GraphBuilder(knowledge_repository, InMemoryGraphRepository())  # type: ignore[arg-type]

        report = await builder.build()

        assert not report.succeeded
        assert report.error


class TestGraphNotBuilt:
    def test_the_error_names_how_to_fix_it(self) -> None:
        error = GraphNotBuiltError("nothing yet")

        assert error.remedy is not None
        assert "/graph/build" in error.remedy
        assert not error.retryable, "retrying a query cannot build the graph"
