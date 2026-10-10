"""The conclusion-dependency graph.

    Conclusion -SUPPORTED_BY-> KnowledgeObject -GROUNDED_IN-> SourceLocation -IN-> Paper
    Contradiction -INVOLVES-> KnowledgeObject
    Paper -HAS_STATUS-> IntegrityObservation

Built as an ordinary :class:`KnowledgeGraph` and written through the existing
``GraphRepository`` port, so memory, JSON and Neo4j all store it unchanged. Like the
knowledge graph it is a *derived* index: the append-only conclusion repository is the
source of truth, and this projection can be rebuilt from it at any time.

Reverse traversal (paper → location → knowledge object → conclusion) is what turns "this
paper was retracted" into "these conclusions depend on it" — and, as importantly, into
"these do not", which is what keeps unaffected conclusions untouched.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.interfaces.graph_repository import GraphRepository
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
    edge_id_for,
    node_id_for_paper,
)
from researchagent.models.integrity import IntegrityObservation
from researchagent.models.maintenance import ConclusionRecord, SupportLink

logger = get_logger(__name__)

_DEPENDENCY_EDGES = frozenset({EdgeKind.IN, EdgeKind.GROUNDED_IN, EdgeKind.SUPPORTED_BY})


def conclusion_node_id(conclusion_id: str) -> str:
    return f"conclusion:{conclusion_id}"


def knowledge_node_id(knowledge_object_id: str) -> str:
    return f"ko:{knowledge_object_id}"


def location_node_id(location: SourceLocation) -> str:
    parts = (
        location.document_id,
        location.page,
        location.section_id,
        location.paragraph_index,
        location.char_start,
        location.char_end,
    )
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]  # noqa: S324
    return f"loc:{digest}"


class DependencyGraphMapper:
    derived_by = "dependency_graph_mapper"

    def __init__(self, *, schema_version: str) -> None:
        self._schema_version = schema_version

    def build(
        self,
        topic_id: str,
        records: Iterable[ConclusionRecord],
        observations: Mapping[str, IntegrityObservation] | None = None,
    ) -> KnowledgeGraph:
        """Project the latest conclusion versions of one topic."""
        latest = tuple(records)
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        known = dict(observations or {})

        for record in latest:
            node_id = conclusion_node_id(record.conclusion_id)
            nodes[node_id] = GraphNode(
                id=node_id,
                kind=NodeKind.CONCLUSION,
                name=record.statement[:160],
                paper_ids=record.paper_ids,
                evidence_ids=record.evidence_ids,
                properties=GraphNodeProperties(
                    description=record.statement,
                    version=record.version,
                    status=record.state.value,
                ),
            )
            for link in record.supports:
                self._add_support(nodes, edges, node_id, link)
            for contradiction in record.contradictions:
                self._add_contradiction_side(
                    nodes,
                    edges,
                    contradiction.id,
                    contradiction.description,
                    contradiction.left_object_id,
                    contradiction.left_paper_id,
                    contradiction.left_evidence,
                )
                self._add_contradiction_side(
                    nodes,
                    edges,
                    contradiction.id,
                    contradiction.description,
                    contradiction.right_object_id,
                    contradiction.right_paper_id,
                    contradiction.right_evidence,
                )

        for paper_id, observation in known.items():
            self._add_paper(nodes, paper_id, None)
            self._add_observation(nodes, edges, paper_id, observation)

        version = GraphVersion(
            schema_version=self._schema_version,
            corpus_fingerprint=_fingerprint(topic_id, latest, known),
        )
        return KnowledgeGraph(
            version=version, nodes=tuple(nodes.values()), edges=tuple(edges.values())
        )

    def _add_support(
        self,
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
        conclusion_id: str,
        link: SupportLink,
    ) -> None:
        ko = self._add_knowledge(nodes, link.knowledge_object_id, link.paper_id)
        location = self._add_location(nodes, link.location, link.paper_id)
        paper = self._add_paper(nodes, link.paper_id, link.doi)
        provenance = GraphProvenance(
            evidence_ids=(link.evidence_id,),
            paper_ids=(link.paper_id,),
            locations=(link.location,),
            derived_by=self.derived_by,
        )
        _merge_edge(edges, EdgeKind.SUPPORTED_BY, conclusion_id, ko, provenance)
        _merge_edge(edges, EdgeKind.GROUNDED_IN, ko, location, provenance)
        _merge_edge(edges, EdgeKind.IN, location, paper, provenance)

    def _add_contradiction_side(
        self,
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
        contradiction_id: str,
        description: str,
        object_id: str,
        paper_id: str,
        evidence: tuple[Evidence, ...],
    ) -> None:
        node_id = f"contradiction:{contradiction_id}"
        nodes.setdefault(
            node_id,
            GraphNode(
                id=node_id, kind=NodeKind.CONTRADICTION, name=description[:160] or contradiction_id
            ),
        )
        ko = self._add_knowledge(nodes, object_id, paper_id)
        paper = self._add_paper(nodes, paper_id, None)
        for item in evidence:
            location = self._add_location(nodes, item.location, paper_id)
            provenance = GraphProvenance(
                evidence_ids=(item.id,),
                paper_ids=(paper_id,),
                locations=(item.location,),
                derived_by=self.derived_by,
            )
            _merge_edge(edges, EdgeKind.GROUNDED_IN, ko, location, provenance)
            _merge_edge(edges, EdgeKind.IN, location, paper, provenance)
        _merge_edge(
            edges,
            EdgeKind.INVOLVES,
            node_id,
            ko,
            GraphProvenance(
                evidence_ids=tuple(item.id for item in evidence),
                paper_ids=(paper_id,),
                locations=tuple(item.location for item in evidence),
                derived_by=self.derived_by,
            ),
        )

    def _add_observation(
        self,
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
        paper_id: str,
        observation: IntegrityObservation,
    ) -> None:
        node_id = f"integrity:{observation.id}"
        nodes[node_id] = GraphNode(
            id=node_id,
            kind=NodeKind.INTEGRITY_OBSERVATION,
            name=f"{observation.status.value} ({observation.registry})",
            paper_ids=(paper_id,),
            properties=GraphNodeProperties(
                description=observation.observation,
                doi=observation.notice_doi,
                status=observation.status.value,
            ),
        )
        # No quoted sentence backs a registry record, so the edge carries no evidence and
        # is honestly untrusted by the knowledge-graph rule; the observation node holds
        # the registry fact instead.
        _merge_edge(
            edges,
            EdgeKind.HAS_STATUS,
            node_id_for_paper(paper_id),
            node_id,
            GraphProvenance(paper_ids=(paper_id,), derived_by=f"integrity:{observation.registry}"),
        )

    def _add_knowledge(self, nodes: dict[str, GraphNode], object_id: str, paper_id: str) -> str:
        node_id = knowledge_node_id(object_id)
        nodes.setdefault(
            node_id,
            GraphNode(
                id=node_id,
                kind=NodeKind.KNOWLEDGE_OBJECT,
                name=object_id,
                knowledge_object_id=object_id,
                paper_ids=(paper_id,),
            ),
        )
        return node_id

    def _add_location(
        self, nodes: dict[str, GraphNode], location: SourceLocation, paper_id: str
    ) -> str:
        node_id = location_node_id(location)
        nodes.setdefault(
            node_id,
            GraphNode(
                id=node_id,
                kind=NodeKind.SOURCE_LOCATION,
                name=location.describe(),
                paper_ids=(paper_id,),
            ),
        )
        return node_id

    def _add_paper(self, nodes: dict[str, GraphNode], paper_id: str, doi: str | None) -> str:
        node_id = node_id_for_paper(paper_id)
        existing = nodes.get(node_id)
        if existing is None or (doi and not existing.properties.doi):
            nodes[node_id] = GraphNode(
                id=node_id,
                kind=NodeKind.PAPER,
                name=paper_id,
                paper_ids=(paper_id,),
                properties=GraphNodeProperties(doi=doi),
            )
        return node_id


def dependents_of(graph: KnowledgeGraph, paper_ids: Iterable[str]) -> dict[str, frozenset[str]]:
    """Conclusions that depend on any of ``paper_ids``, each with the papers that reach it.

    Directed reverse traversal over IN, GROUNDED_IN and SUPPORTED_BY only: a conclusion is
    reached exactly when one of its support links ends in one of these papers.
    """
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind in _DEPENDENCY_EDGES:
            reverse[edge.target_id].append(edge.source_id)
    kinds = {node.id: node.kind for node in graph.nodes}

    reached: dict[str, set[str]] = defaultdict(set)
    for paper_id in dict.fromkeys(paper_ids):
        start = node_id_for_paper(paper_id)
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for source in reverse.get(current, ()):
                if source in seen:
                    continue
                seen.add(source)
                if kinds.get(source) is NodeKind.CONCLUSION:
                    reached[source.removeprefix("conclusion:")].add(paper_id)
                else:
                    queue.append(source)
    return {conclusion: frozenset(papers) for conclusion, papers in reached.items()}


class DependencyGraphStore:
    """Writes projections through the shared ``GraphRepository`` port."""

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> GraphRepository:
        return self._repository

    async def write(self, graph: KnowledgeGraph) -> str:
        stats = await self._repository.write_graph(graph)
        logger.info(
            "dependency_graph_written",
            version=graph.version.identifier,
            nodes=stats.nodes,
            edges=stats.edges,
        )
        return graph.version.identifier


def _merge_edge(
    edges: dict[str, GraphEdge],
    kind: EdgeKind,
    source: str,
    target: str,
    provenance: GraphProvenance,
) -> None:
    edge_id = edge_id_for(kind, source, target)
    existing = edges.get(edge_id)
    if existing is not None:
        provenance = GraphProvenance(
            evidence_ids=tuple(
                dict.fromkeys((*existing.provenance.evidence_ids, *provenance.evidence_ids))
            ),
            paper_ids=tuple(dict.fromkeys((*existing.provenance.paper_ids, *provenance.paper_ids))),
            locations=tuple(dict.fromkeys((*existing.provenance.locations, *provenance.locations))),
            derived_by=provenance.derived_by,
        )
    edges[edge_id] = GraphEdge(
        id=edge_id, kind=kind, source_id=source, target_id=target, provenance=provenance
    )


def _fingerprint(
    topic_id: str,
    records: tuple[ConclusionRecord, ...],
    observations: Mapping[str, IntegrityObservation],
) -> str:
    payload = "|".join(
        [
            topic_id,
            *sorted(f"{record.conclusion_id}@{record.version}" for record in records),
            *sorted(observation.id for observation in observations.values()),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]  # noqa: S324
