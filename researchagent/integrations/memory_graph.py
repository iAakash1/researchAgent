"""In-memory graph repository.

A full implementation of the port, so graph mapping, validation, idempotency and every
domain query are tested without a Neo4j server. The Neo4j adapter is exercised by
integration tests that skip when it is absent.
"""

from __future__ import annotations

from typing import ClassVar

from researchagent.core.interfaces.graph_repository import (
    GraphRepository,
    GraphStats,
    Subgraph,
)
from researchagent.core.logging import get_logger
from researchagent.models.graph import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphVersion,
    KnowledgeGraph,
    NodeKind,
)
from researchagent.utils.text import normalise

logger = get_logger(__name__)


class InMemoryGraphRepository(GraphRepository):
    name: ClassVar[str] = "memory"

    def __init__(self) -> None:
        self._generations: dict[str, KnowledgeGraph] = {}

    async def write_graph(self, graph: KnowledgeGraph) -> GraphStats:
        # Whole-generation replace: idempotent by construction, because the same corpus
        # produces the same version identifier and the same deterministic ids.
        self._generations[graph.version.identifier] = graph
        return await self.stats(graph.version)

    async def get_node(self, node_id: str, version: GraphVersion) -> GraphNode | None:
        graph = self._generations.get(version.identifier)
        return graph.node(node_id) if graph else None

    async def find_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        name: str | None = None,
        version: GraphVersion,
        limit: int = 50,
    ) -> tuple[GraphNode, ...]:
        graph = self._generations.get(version.identifier)
        if graph is None:
            return ()
        needle = normalise(name) if name else None
        matches = [
            node
            for node in graph.nodes
            if (kind is None or node.kind is kind)
            and (needle is None or needle in normalise(node.name))
        ]
        return tuple(matches[:limit])

    async def neighbours(
        self,
        node_id: str,
        version: GraphVersion,
        *,
        edge_kinds: tuple[EdgeKind, ...] = (),
        depth: int = 1,
        limit: int = 50,
    ) -> Subgraph:
        graph = self._generations.get(version.identifier)
        if graph is None or graph.node(node_id) is None:
            return Subgraph(centre_id=node_id, depth=depth)

        frontier = {node_id}
        seen = {node_id}
        collected: list[GraphEdge] = []

        for _ in range(max(depth, 0)):
            next_frontier: set[str] = set()
            for edge in graph.edges:
                if edge_kinds and edge.kind not in edge_kinds:
                    continue
                if edge.source_id in frontier:
                    collected.append(edge)
                    next_frontier.add(edge.target_id)
                elif edge.target_id in frontier:
                    collected.append(edge)
                    next_frontier.add(edge.source_id)
            frontier = next_frontier - seen
            seen |= next_frontier
            if not frontier:
                break

        nodes = tuple(node for node in graph.nodes if node.id in seen)[:limit]
        return Subgraph(nodes=nodes, edges=tuple(collected[:limit]), centre_id=node_id, depth=depth)

    async def edges_between(
        self, source_id: str, target_id: str, version: GraphVersion
    ) -> tuple[GraphEdge, ...]:
        graph = self._generations.get(version.identifier)
        if graph is None:
            return ()
        pair = {source_id, target_id}
        return tuple(edge for edge in graph.edges if {edge.source_id, edge.target_id} == pair)

    async def versions(self) -> tuple[GraphVersion, ...]:
        return tuple(
            sorted(
                (graph.version for graph in self._generations.values()),
                key=lambda version: version.created_at,
                reverse=True,
            )
        )

    async def stats(self, version: GraphVersion) -> GraphStats:
        graph = self._generations.get(version.identifier)
        if graph is None:
            return GraphStats(version=version.identifier)

        nodes_by_kind: dict[str, int] = {}
        for node in graph.nodes:
            nodes_by_kind[node.kind.value] = nodes_by_kind.get(node.kind.value, 0) + 1
        edges_by_kind: dict[str, int] = {}
        for edge in graph.edges:
            edges_by_kind[edge.kind.value] = edges_by_kind.get(edge.kind.value, 0) + 1

        return GraphStats(
            version=version.identifier,
            nodes=len(graph.nodes),
            edges=len(graph.edges),
            nodes_by_kind=nodes_by_kind,
            edges_by_kind=edges_by_kind,
        )

    async def delete_version(self, version: GraphVersion) -> bool:
        return self._generations.pop(version.identifier, None) is not None

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
