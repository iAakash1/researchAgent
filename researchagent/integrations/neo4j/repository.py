"""Neo4j graph repository adapter.

The only module that imports the Neo4j driver. Neo4j is a *derived index*: everything it
holds is reconstructible from the knowledge and evidence repositories, so losing it costs
a rebuild and never a fact.

Domain ids are the identity. Nodes and edges are written with `MERGE` on the deterministic
id produced by the mapper, which makes construction idempotent — running it twice over the
same corpus produces the same graph, not a duplicated one.
"""

from __future__ import annotations

from typing import Any, ClassVar

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
    GraphNodeProperties,
    GraphProvenance,
    GraphVersion,
    KnowledgeGraph,
    NodeKind,
)

logger = get_logger(__name__)

_BATCH = 500


class Neo4jGraphRepository(GraphRepository):
    name: ClassVar[str] = "neo4j"

    def __init__(
        self,
        *,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._auth = (user, password)
        self._database = database
        self._driver: Any | None = None

    def _connect(self) -> Any:
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise GraphStoreError("neo4j driver is not installed") from exc
            self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
        return self._driver

    async def write_graph(self, graph: KnowledgeGraph) -> GraphStats:
        driver = self._connect()
        version = graph.version.identifier

        try:
            async with driver.session(database=self._database) as session:
                # Uniqueness on (id, graph_version) is what makes MERGE idempotent and
                # keeps generations from mixing.
                await session.run(
                    "CREATE CONSTRAINT ra_node_identity IF NOT EXISTS "
                    "FOR (n:RAEntity) REQUIRE (n.id, n.graph_version) IS UNIQUE"
                )
                for start in range(0, len(graph.nodes), _BATCH):
                    await session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (n:RAEntity {id: row.id, graph_version: $version})
                        SET n += row.props, n.kind = row.kind, n.name = row.name,
                            n.built_at = $built_at
                        """,
                        rows=[_node_row(node) for node in graph.nodes[start : start + _BATCH]],
                        version=version,
                        built_at=graph.version.created_at.isoformat(),
                    )
                for start in range(0, len(graph.edges), _BATCH):
                    await session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (a:RAEntity {id: row.source, graph_version: $version})
                        MATCH (b:RAEntity {id: row.target, graph_version: $version})
                        MERGE (a)-[r:RELATES {id: row.id, graph_version: $version}]->(b)
                        SET r += row.props
                        """,
                        rows=[_edge_row(edge) for edge in graph.edges[start : start + _BATCH]],
                        version=version,
                    )
        except GraphStoreError:
            raise
        except Exception as exc:
            raise GraphStoreError("Could not write the graph", reason=str(exc)) from exc

        return await self.stats(graph.version)

    async def get_node(self, node_id: str, version: GraphVersion) -> GraphNode | None:
        rows = await self._read(
            "MATCH (n:RAEntity {id: $id, graph_version: $version}) RETURN n LIMIT 1",
            id=node_id,
            version=version.identifier,
        )
        return _to_node(rows[0]["n"]) if rows else None

    async def find_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        name: str | None = None,
        version: GraphVersion,
        limit: int = 50,
    ) -> tuple[GraphNode, ...]:
        rows = await self._read(
            """
            MATCH (n:RAEntity {graph_version: $version})
            WHERE ($kind IS NULL OR n.kind = $kind)
              AND ($name IS NULL OR toLower(n.name) CONTAINS toLower($name))
            RETURN n LIMIT $limit
            """,
            version=version.identifier,
            kind=kind.value if kind else None,
            name=name,
            limit=limit,
        )
        return tuple(_to_node(row["n"]) for row in rows)

    async def neighbours(
        self,
        node_id: str,
        version: GraphVersion,
        *,
        edge_kinds: tuple[EdgeKind, ...] = (),
        depth: int = 1,
        limit: int = 50,
    ) -> Subgraph:
        # Depth is interpolated because Cypher does not parameterise path length; it is an
        # int from a typed field, never user text.
        hops = max(1, min(depth, 5))
        rows = await self._read(
            f"""
            MATCH (c:RAEntity {{id: $id, graph_version: $version}})
            MATCH path = (c)-[r:RELATES*1..{hops}]-(n:RAEntity)
            WHERE ($kinds IS NULL OR ALL(e IN relationships(path) WHERE e.kind IN $kinds))
            RETURN DISTINCT n, relationships(path) AS rels LIMIT $limit
            """,
            id=node_id,
            version=version.identifier,
            kinds=[kind.value for kind in edge_kinds] or None,
            limit=limit,
        )

        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for row in rows:
            node = _to_node(row["n"])
            nodes[node.id] = node
            for relationship in row["rels"]:
                edge = _to_edge(relationship)
                edges[edge.id] = edge

        return Subgraph(
            nodes=tuple(nodes.values()),
            edges=tuple(edges.values()),
            centre_id=node_id,
            depth=depth,
        )

    async def edges_between(
        self, source_id: str, target_id: str, version: GraphVersion
    ) -> tuple[GraphEdge, ...]:
        rows = await self._read(
            """
            MATCH (a:RAEntity {graph_version: $version})-[r:RELATES]-(b:RAEntity)
            WHERE a.id IN [$source, $target] AND b.id IN [$source, $target] AND a.id <> b.id
            RETURN DISTINCT r
            """,
            version=version.identifier,
            source=source_id,
            target=target_id,
        )
        return tuple(_to_edge(row["r"]) for row in rows)

    async def versions(self) -> tuple[GraphVersion, ...]:
        rows = await self._read(
            """
            MATCH (n:RAEntity)
            RETURN DISTINCT n.graph_version AS version, max(n.built_at) AS built_at
            ORDER BY built_at DESC
            """
        )
        return tuple(_to_version(row) for row in rows if row.get("version"))

    async def stats(self, version: GraphVersion) -> GraphStats:
        node_rows = await self._read(
            "MATCH (n:RAEntity {graph_version: $version}) RETURN n.kind AS kind, count(*) AS total",
            version=version.identifier,
        )
        edge_rows = await self._read(
            "MATCH (:RAEntity {graph_version: $version})-[r:RELATES]->() "
            "RETURN r.kind AS kind, count(*) AS total",
            version=version.identifier,
        )
        nodes_by_kind = {row["kind"]: row["total"] for row in node_rows}
        edges_by_kind = {row["kind"]: row["total"] for row in edge_rows}
        return GraphStats(
            version=version.identifier,
            nodes=sum(nodes_by_kind.values()),
            edges=sum(edges_by_kind.values()),
            nodes_by_kind=nodes_by_kind,
            edges_by_kind=edges_by_kind,
        )

    async def delete_version(self, version: GraphVersion) -> bool:
        rows = await self._read(
            "MATCH (n:RAEntity {graph_version: $version}) DETACH DELETE n RETURN count(*) AS n",
            version=version.identifier,
        )
        return bool(rows and rows[0]["n"])

    async def health(self) -> bool:
        try:
            await self._read("RETURN 1 AS ok")
        except GraphStoreError as exc:
            logger.warning("neo4j_unhealthy", uri=self._uri, reason=exc.message)
            return False
        return True

    async def aclose(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        driver = self._connect()
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(cypher, **params)
                return [record.data() async for record in result]
        except Exception as exc:
            raise GraphStoreError("Neo4j query failed", reason=str(exc)) from exc


def _node_row(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind.value,
        "name": node.name,
        "props": {
            "knowledge_object_id": node.knowledge_object_id,
            "paper_ids": list(node.paper_ids),
            "evidence_ids": list(node.evidence_ids),
            "confidence": node.confidence.score,
            **{
                key: value
                for key, value in node.properties.model_dump().items()
                if value is not None
            },
        },
    }


def _edge_row(edge: GraphEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source": edge.source_id,
        "target": edge.target_id,
        "props": {
            "kind": edge.kind.value,
            "evidence_ids": list(edge.provenance.evidence_ids),
            "paper_ids": list(edge.provenance.paper_ids),
            "citations": list(edge.provenance.cite()),
            "derived_by": edge.provenance.derived_by,
            "confidence": edge.confidence.score,
        },
    }


def _to_node(raw: dict[str, Any]) -> GraphNode:
    from researchagent.core.validation import Confidence

    return GraphNode(
        id=raw["id"],
        kind=NodeKind(raw["kind"]),
        name=raw.get("name") or raw["id"],
        knowledge_object_id=raw.get("knowledge_object_id"),
        paper_ids=tuple(raw.get("paper_ids") or ()),
        evidence_ids=tuple(raw.get("evidence_ids") or ()),
        confidence=Confidence(score=float(raw.get("confidence") or 0.0)),
        properties=GraphNodeProperties(
            description=raw.get("description") or "",
            year=raw.get("year"),
            venue=raw.get("venue"),
            doi=raw.get("doi"),
            metric_name=raw.get("metric_name"),
            dataset_name=raw.get("dataset_name"),
            numeric_value=raw.get("numeric_value"),
            unit=raw.get("unit"),
        ),
    )


def _to_edge(raw: dict[str, Any]) -> GraphEdge:
    from researchagent.core.validation import Confidence

    source, target = raw["id"].split(f"--{raw['kind']}--", 1)
    return GraphEdge(
        id=raw["id"],
        kind=EdgeKind(raw["kind"]),
        source_id=source,
        target_id=target,
        provenance=GraphProvenance(
            evidence_ids=tuple(raw.get("evidence_ids") or ()),
            paper_ids=tuple(raw.get("paper_ids") or ()),
            derived_by=raw.get("derived_by") or "unknown",
        ),
        confidence=Confidence(score=float(raw.get("confidence") or 0.0)),
    )


def _to_version(row: dict[str, Any]) -> GraphVersion:
    """Reconstruct a version from its identifier.

    The identifier is built by ``GraphVersion.identifier`` and is the only thing stored on
    a node, so this parses it back rather than keeping a parallel table that could drift.
    """
    from datetime import UTC, datetime

    identifier = str(row["version"])
    schema, extraction, relation, corpus = (part[1:] for part in identifier.split("-", 3))
    built_at = row.get("built_at")
    return GraphVersion(
        schema_version=schema,
        extraction_version=extraction,
        relation_version=relation,
        corpus_fingerprint=corpus,
        created_at=(datetime.fromisoformat(str(built_at)) if built_at else datetime.now(UTC)),
    )
