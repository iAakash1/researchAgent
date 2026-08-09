"""Knowledge graph persistence port.

Neo4j is a derived index behind this interface. No domain logic imports a driver, and
the graph can be dropped and rebuilt from the knowledge and evidence repositories at any
time — which is what keeps the source of truth where it belongs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.models.graph import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphVersion,
    KnowledgeGraph,
    NodeKind,
)


class Subgraph(BaseModel):
    """A neighbourhood of the graph, with the edges that connect it."""

    model_config = {"frozen": True}

    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    centre_id: str | None = None
    depth: int = Field(default=1, ge=0)


class GraphStats(BaseModel):
    model_config = {"frozen": True}

    version: str = ""
    nodes: int = 0
    edges: int = 0
    nodes_by_kind: dict[str, int] = Field(default_factory=dict)
    edges_by_kind: dict[str, int] = Field(default_factory=dict)


class GraphRepository(ABC):
    """Stores and queries a versioned knowledge graph."""

    name: ClassVar[str]

    @abstractmethod
    async def write_graph(self, graph: KnowledgeGraph) -> GraphStats:
        """Persist a whole generation.

        Must be idempotent: writing the same graph twice produces the same graph, never
        duplicates. Domain ids are the identity; database-generated ids are not.
        """

    @abstractmethod
    async def get_node(self, node_id: str, version: GraphVersion) -> GraphNode | None: ...

    @abstractmethod
    async def find_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        name: str | None = None,
        version: GraphVersion,
        limit: int = 50,
    ) -> tuple[GraphNode, ...]: ...

    @abstractmethod
    async def neighbours(
        self,
        node_id: str,
        version: GraphVersion,
        *,
        edge_kinds: tuple[EdgeKind, ...] = (),
        depth: int = 1,
        limit: int = 50,
    ) -> Subgraph: ...

    @abstractmethod
    async def edges_between(
        self, source_id: str, target_id: str, version: GraphVersion
    ) -> tuple[GraphEdge, ...]:
        """Every relationship connecting two nodes — the provenance lookup."""

    @abstractmethod
    @abstractmethod
    async def versions(self) -> tuple[GraphVersion, ...]:
        """Every generation held, newest first.

        Lets callers address "the current graph" without knowing a version identifier,
        and makes the presence of older generations visible rather than implicit.
        """

    @abstractmethod
    async def stats(self, version: GraphVersion) -> GraphStats: ...

    @abstractmethod
    async def delete_version(self, version: GraphVersion) -> bool: ...

    @abstractmethod
    async def health(self) -> bool:
        """Cheap probe; must not raise. An unreachable graph degrades, never crashes."""

    @abstractmethod
    async def aclose(self) -> None: ...
