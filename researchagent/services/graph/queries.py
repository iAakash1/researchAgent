"""Domain-level graph queries.

The questions retrieval cannot answer. Retrieval finds what is *relevant*; the graph
finds how things are *connected* — which methods share a dataset, which papers report
conflicting numbers, which entities recur across the corpus.

Every result carries provenance. A graph answer without the evidence behind it is an
assertion, and the architecture does not deal in those.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from researchagent.core.interfaces.graph_repository import GraphRepository
from researchagent.core.logging import get_logger
from researchagent.models.graph import EdgeKind, GraphEdge, GraphNode, GraphVersion, NodeKind

logger = get_logger(__name__)


class ConnectedEntity(BaseModel):
    """One entity reached from another, with the evidence for the connection."""

    model_config = {"frozen": True}

    node: GraphNode
    via: EdgeKind
    citations: tuple[str, ...] = ()
    confidence: float = 0.0


class SharedEntity(BaseModel):
    """An entity several papers or methods have in common."""

    model_config = {"frozen": True}

    node: GraphNode
    shared_by: tuple[GraphNode, ...] = ()
    citations: tuple[str, ...] = ()

    @property
    def share_count(self) -> int:
        return len(self.shared_by)


class ContradictionPair(BaseModel):
    model_config = {"frozen": True}

    left: GraphNode
    right: GraphNode
    papers: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    confidence: float = 0.0


class GraphQueries:
    """Provenance-aware domain queries over a graph generation."""

    name = "graph_queries"

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    async def datasets_for_method(
        self, method_name: str, version: GraphVersion
    ) -> tuple[ConnectedEntity, ...]:
        """Which datasets was this method evaluated on?"""
        return await self._connected(
            method_name, NodeKind.METHOD, EdgeKind.EVALUATED_ON, NodeKind.DATASET, version
        )

    async def methods_for_dataset(
        self, dataset_name: str, version: GraphVersion
    ) -> tuple[ConnectedEntity, ...]:
        """Which methods were evaluated on this dataset?"""
        return await self._connected(
            dataset_name, NodeKind.DATASET, EdgeKind.EVALUATED_ON, NodeKind.METHOD, version
        )

    async def metrics_for_method(
        self, method_name: str, version: GraphVersion
    ) -> tuple[ConnectedEntity, ...]:
        """Which metrics were used to evaluate this method?

        Two hops: Method <- PRODUCED_BY - Result - MEASURED_BY -> Metric. The kind of
        question a flat index cannot answer at all.
        """
        nodes = await self._repository.find_nodes(
            kind=NodeKind.METHOD, name=method_name, version=version, limit=5
        )
        found: dict[str, ConnectedEntity] = {}

        for method in nodes:
            neighbourhood = await self._repository.neighbours(
                method.id, version, edge_kinds=(EdgeKind.PRODUCED_BY,), depth=1
            )
            for result_node in neighbourhood.nodes:
                if result_node.kind is not NodeKind.RESULT:
                    continue
                metrics = await self._repository.neighbours(
                    result_node.id, version, edge_kinds=(EdgeKind.MEASURED_BY,), depth=1
                )
                for candidate in metrics.nodes:
                    if candidate.kind is NodeKind.METRIC and candidate.id not in found:
                        edges = await self._repository.edges_between(
                            result_node.id, candidate.id, version
                        )
                        found[candidate.id] = ConnectedEntity(
                            node=candidate,
                            via=EdgeKind.MEASURED_BY,
                            citations=_citations(edges),
                            confidence=max((e.confidence.score for e in edges), default=0.0),
                        )
        return tuple(found.values())

    async def papers_using(self, entity_name: str, version: GraphVersion) -> tuple[GraphNode, ...]:
        """Which papers mention this entity?"""
        matches = await self._repository.find_nodes(name=entity_name, version=version, limit=5)
        papers: dict[str, GraphNode] = {}
        for match in matches:
            if match.kind is NodeKind.PAPER:
                continue
            neighbourhood = await self._repository.neighbours(
                match.id, version, edge_kinds=(EdgeKind.MENTIONS,), depth=1
            )
            for node in neighbourhood.nodes:
                if node.kind is NodeKind.PAPER:
                    papers[node.id] = node
        return tuple(papers.values())

    async def entities_across_papers(
        self, version: GraphVersion, *, kind: NodeKind | None = None, minimum_papers: int = 2
    ) -> tuple[SharedEntity, ...]:
        """Which entities recur across several papers?

        The cross-paper question the whole merged-node design exists to serve.
        """
        nodes = await self._repository.find_nodes(kind=kind, version=version, limit=1000)
        shared = []
        for node in nodes:
            if node.kind is NodeKind.PAPER or len(node.paper_ids) < minimum_papers:
                continue
            papers = await self._repository.neighbours(
                node.id, version, edge_kinds=(EdgeKind.MENTIONS,), depth=1
            )
            shared.append(
                SharedEntity(
                    node=node,
                    shared_by=tuple(n for n in papers.nodes if n.kind is NodeKind.PAPER),
                    citations=_citations(papers.edges),
                )
            )
        shared.sort(key=lambda item: (-len(item.node.paper_ids), item.node.name))
        return tuple(shared)

    async def shared_datasets(self, version: GraphVersion) -> tuple[SharedEntity, ...]:
        """Which datasets are used by more than one method? The gap-analysis query."""
        datasets = await self._repository.find_nodes(
            kind=NodeKind.DATASET, version=version, limit=1000
        )
        shared = []
        for dataset in datasets:
            neighbourhood = await self._repository.neighbours(
                dataset.id, version, edge_kinds=(EdgeKind.EVALUATED_ON,), depth=1
            )
            methods = tuple(n for n in neighbourhood.nodes if n.kind is NodeKind.METHOD)
            if len(methods) >= 2:
                shared.append(
                    SharedEntity(
                        node=dataset, shared_by=methods, citations=_citations(neighbourhood.edges)
                    )
                )
        shared.sort(key=lambda item: (-item.share_count, item.node.name))
        return tuple(shared)

    async def contradictions(self, version: GraphVersion) -> tuple[ContradictionPair, ...]:
        """Where does the literature disagree? Both sides, never resolved."""
        nodes = {
            node.id: node for node in await self._repository.find_nodes(version=version, limit=2000)
        }
        pairs = []
        for node in nodes.values():
            neighbourhood = await self._repository.neighbours(
                node.id, version, edge_kinds=(EdgeKind.CONTRADICTS,), depth=1
            )
            for edge in neighbourhood.edges:
                if edge.source_id != node.id:
                    continue
                other = nodes.get(edge.target_id)
                if other is None:
                    continue
                pairs.append(
                    ContradictionPair(
                        left=node,
                        right=other,
                        papers=edge.provenance.paper_ids,
                        citations=edge.provenance.cite(),
                        confidence=edge.confidence.score,
                    )
                )
        return tuple(pairs)

    async def provenance_for(
        self, source_id: str, target_id: str, version: GraphVersion
    ) -> tuple[str, ...]:
        """Which evidence supports the relationship between these two nodes?"""
        edges = await self._repository.edges_between(source_id, target_id, version)
        return _citations(edges)

    async def _connected(
        self,
        name: str,
        from_kind: NodeKind,
        via: EdgeKind,
        to_kind: NodeKind,
        version: GraphVersion,
    ) -> tuple[ConnectedEntity, ...]:
        matches = await self._repository.find_nodes(
            kind=from_kind, name=name, version=version, limit=5
        )
        found: dict[str, ConnectedEntity] = {}

        for match in matches:
            neighbourhood = await self._repository.neighbours(
                match.id, version, edge_kinds=(via,), depth=1
            )
            for node in neighbourhood.nodes:
                if node.kind is not to_kind or node.id in found:
                    continue
                edges = await self._repository.edges_between(match.id, node.id, version)
                if not edges:
                    edges = await self._repository.edges_between(node.id, match.id, version)
                found[node.id] = ConnectedEntity(
                    node=node,
                    via=via,
                    citations=_citations(edges),
                    confidence=max((edge.confidence.score for edge in edges), default=0.0),
                )
        return tuple(found.values())


def _citations(edges: Sequence[GraphEdge]) -> tuple[str, ...]:
    """Distinct provenance addresses across a set of edges, in first-seen order."""
    seen: dict[str, None] = {}
    for edge in edges:
        for citation in edge.provenance.cite():
            seen.setdefault(citation, None)
    return tuple(seen)
