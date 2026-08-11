"""File-backed graph storage.

A third adapter behind the existing :class:`GraphRepository` port, alongside the in-memory
one (fast, per-process) and Neo4j (a server). This one exists for the case both miss: a
graph that must survive the process that built it, on a machine with no Neo4j running.

Same shape as the other JSON repositories in this package — one file per generation, so a
version can be written, read and deleted without touching the others. The graph remains a
derived index: losing this directory costs a rebuild, never a fact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from researchagent.core.exceptions import GraphStoreError
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

logger = get_logger(__name__)


class JsonGraphRepository(GraphRepository):
    """One JSON document per graph generation."""

    name: ClassVar[str] = "json"

    def __init__(self, graphs_dir: Path) -> None:
        self._graphs_dir = graphs_dir
        self._lock = asyncio.Lock()
        self._cache: dict[str, KnowledgeGraph] = {}

    @property
    def graphs_dir(self) -> Path:
        return self._graphs_dir

    async def write_graph(self, graph: KnowledgeGraph) -> GraphStats:
        """Whole-generation replace.

        Idempotent by construction: the same corpus yields the same version identifier and
        the same deterministic node and edge ids, so a rebuild overwrites one file rather
        than accumulating duplicates.
        """
        async with self._lock:
            self._graphs_dir.mkdir(parents=True, exist_ok=True)
            path = self._path_for(graph.version.identifier)
            try:
                path.write_text(graph.model_dump_json(indent=2))
            except OSError as exc:
                raise GraphStoreError(
                    "Could not write the graph generation",
                    path=str(path),
                    reason=str(exc),
                ) from exc
            self._cache[graph.version.identifier] = graph
        logger.info(
            "graph_persisted",
            version=graph.version.identifier,
            nodes=len(graph.nodes),
            edges=len(graph.edges),
        )
        return await self.stats(graph.version)

    async def get_node(self, node_id: str, version: GraphVersion) -> GraphNode | None:
        graph = await self._load(version.identifier)
        return graph.node(node_id) if graph else None

    async def find_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        name: str | None = None,
        version: GraphVersion,
        limit: int = 50,
    ) -> tuple[GraphNode, ...]:
        graph = await self._load(version.identifier)
        if graph is None:
            return ()
        needle = name.lower() if name else None
        matches = [
            node
            for node in graph.nodes
            if (kind is None or node.kind is kind)
            and (needle is None or needle in node.name.lower())
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
        graph = await self._load(version.identifier)
        if graph is None:
            return Subgraph(centre_id=node_id, depth=depth)

        wanted = set(edge_kinds)
        reached = {node_id}
        edges: dict[str, GraphEdge] = {}
        frontier = {node_id}

        for _ in range(max(1, depth)):
            nxt: set[str] = set()
            for edge in graph.edges:
                if wanted and edge.kind not in wanted:
                    continue
                if edge.source_id in frontier:
                    edges[edge.id] = edge
                    nxt.add(edge.target_id)
                elif edge.target_id in frontier:
                    edges[edge.id] = edge
                    nxt.add(edge.source_id)
            frontier = nxt - reached
            reached |= nxt
            if not frontier:
                break

        nodes = tuple(node for node in graph.nodes if node.id in reached and node.id != node_id)
        return Subgraph(
            nodes=nodes[:limit],
            edges=tuple(edges.values()),
            centre_id=node_id,
            depth=depth,
        )

    async def edges_between(
        self, source_id: str, target_id: str, version: GraphVersion
    ) -> tuple[GraphEdge, ...]:
        graph = await self._load(version.identifier)
        if graph is None:
            return ()
        pair = {source_id, target_id}
        return tuple(edge for edge in graph.edges if {edge.source_id, edge.target_id} == pair)

    async def versions(self) -> tuple[GraphVersion, ...]:
        versions = []
        for path in sorted(self._graphs_dir.glob("*.json")) if self._graphs_dir.is_dir() else []:
            graph = await self._load(path.stem)
            if graph is not None:
                versions.append(graph.version)
        return tuple(sorted(versions, key=lambda v: v.created_at, reverse=True))

    async def stats(self, version: GraphVersion) -> GraphStats:
        graph = await self._load(version.identifier)
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
        async with self._lock:
            path = self._path_for(version.identifier)
            self._cache.pop(version.identifier, None)
            if not path.is_file():
                return False
            path.unlink()
        return True

    async def health(self) -> bool:
        try:
            self._graphs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("graph_store_unwritable", path=str(self._graphs_dir), reason=str(exc))
            return False
        return True

    async def aclose(self) -> None:
        self._cache.clear()

    async def _load(self, identifier: str) -> KnowledgeGraph | None:
        cached = self._cache.get(identifier)
        if cached is not None:
            return cached

        path = self._path_for(identifier)
        if not path.is_file():
            return None
        try:
            graph = KnowledgeGraph.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            # A generation written by an incompatible schema is unreadable, not silently
            # empty: reporting it as absent would look like "no graph has been built".
            raise GraphStoreError(
                "Stored graph generation could not be read",
                path=str(path),
                reason=str(exc),
                remedy="Rebuild the graph; it is derived from the repositories",
            ) from exc
        self._cache[identifier] = graph
        return graph

    def _path_for(self, identifier: str) -> Path:
        return self._graphs_dir / f"{identifier}.json"
