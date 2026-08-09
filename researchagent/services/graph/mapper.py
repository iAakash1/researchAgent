"""Knowledge to graph mapping.

Turns validated knowledge into nodes and edges. Nothing is generated: every node mirrors
a KnowledgeObject or a Paper, and every edge mirrors either a v0.5 KnowledgeRelation
(evidence-derived) or a v0.6 Contradiction (mechanically detected).

Entities are merged across papers by name. Two papers describing "MIMIC-III" become one
node carrying both papers and both sets of evidence — which is precisely what makes
"which methods share a dataset?" answerable, and what a per-object node could not do.
"""

from __future__ import annotations

from researchagent.core.evidence import SourceLocation
from researchagent.core.logging import get_logger
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import Contradiction
from researchagent.models.graph import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphNodeProperties,
    GraphProvenance,
    GraphVersion,
    KnowledgeGraph,
    NodeKind,
    corpus_fingerprint,
    edge_id_for,
    node_id_for_knowledge,
    node_id_for_paper,
)
from researchagent.models.knowledge import (
    DatasetDetails,
    KnowledgeObject,
    MetricDetails,
    PaperKnowledge,
    ResultDetails,
)
from researchagent.models.paper import Paper

logger = get_logger(__name__)

MAPPER_NAME = "graph_mapper"


class GraphMapper:
    """Builds a graph generation from validated knowledge."""

    name = MAPPER_NAME

    def __init__(
        self,
        schema_version: str = "1",
        extraction_version: str = "1",
        relation_version: str = "1",
    ) -> None:
        # Version identity is configuration, not a per-call argument: two graphs built by
        # the same mapper must be comparable.
        self._schema_version = schema_version
        self._extraction_version = extraction_version
        self._relation_version = relation_version

    def build(
        self,
        knowledge: list[PaperKnowledge],
        *,
        papers: dict[str, Paper] | None = None,
        contradictions: tuple[Contradiction, ...] = (),
    ) -> KnowledgeGraph:
        papers = papers or {}
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        object_to_node: dict[str, str] = {}

        for paper_knowledge in knowledge:
            self._add_paper(nodes, paper_knowledge.paper_id, papers.get(paper_knowledge.paper_id))
            for obj in paper_knowledge.objects:
                node_id = self._add_knowledge_node(nodes, obj)
                object_to_node[obj.id] = node_id
                self._add_mentions(edges, paper_knowledge.paper_id, node_id, obj)

        for paper_knowledge in knowledge:
            for relation in paper_knowledge.relations:
                self._add_relation(edges, relation, object_to_node, paper_knowledge)

        for contradiction in contradictions:
            self._add_contradiction(edges, contradiction, object_to_node)

        version = GraphVersion(
            schema_version=self._schema_version,
            extraction_version=self._extraction_version,
            relation_version=self._relation_version,
            corpus_fingerprint=corpus_fingerprint(
                tuple(item.paper_id for item in knowledge),
                sum(len(item.objects) for item in knowledge),
            ),
        )
        graph = KnowledgeGraph(
            version=version, nodes=tuple(nodes.values()), edges=tuple(edges.values())
        )
        logger.info(
            "graph_mapped",
            version=version.identifier,
            nodes=len(graph.nodes),
            edges=len(graph.edges),
            provenance_coverage=round(graph.provenance_coverage, 4),
        )
        return graph

    def _add_paper(self, nodes: dict[str, GraphNode], paper_id: str, paper: Paper | None) -> str:
        node_id = node_id_for_paper(paper_id)
        if node_id in nodes:
            return node_id
        nodes[node_id] = GraphNode(
            id=node_id,
            kind=NodeKind.PAPER,
            name=(paper.title if paper else paper_id)[:200],
            paper_ids=(paper_id,),
            confidence=Confidence.certain(f"paper {paper_id} exists in the paper repository"),
            properties=GraphNodeProperties(
                year=paper.year if paper else None,
                venue=paper.venue if paper else None,
                doi=paper.doi if paper else None,
            ),
        )
        return node_id

    def _add_knowledge_node(self, nodes: dict[str, GraphNode], obj: KnowledgeObject) -> str:
        node_id = node_id_for_knowledge(obj.kind, obj.name, obj.paper_id)
        evidence_ids = tuple(item.id for item in obj.evidence)
        existing = nodes.get(node_id)

        if existing is None:
            nodes[node_id] = GraphNode(
                id=node_id,
                kind=NodeKind.for_knowledge(obj.kind),
                name=obj.name[:200],
                knowledge_object_id=obj.id,
                paper_ids=(obj.paper_id,),
                confidence=obj.confidence,
                evidence_ids=evidence_ids,
                properties=_properties_for(obj),
            )
            return node_id

        # Same entity, another paper: merge rather than duplicate, and record the extra
        # corroboration as part of the node's confidence.
        merged_papers = tuple(sorted({*existing.paper_ids, obj.paper_id}))
        nodes[node_id] = existing.model_copy(
            update={
                "paper_ids": merged_papers,
                "evidence_ids": tuple({*existing.evidence_ids, *evidence_ids}),
                "confidence": existing.confidence.combined_with(
                    Confidence.from_signals(
                        [
                            ConfidenceSignal(
                                name="cross_paper_agreement",
                                value=min(len(merged_papers) / 3, 1.0),
                                observation=(
                                    f"{len(merged_papers)} paper(s) describe {obj.name!r}: "
                                    f"{list(merged_papers)}"
                                ),
                            )
                        ]
                    )
                ),
            }
        )
        return node_id

    def _add_mentions(
        self,
        edges: dict[str, GraphEdge],
        paper_id: str,
        node_id: str,
        obj: KnowledgeObject,
    ) -> None:
        source = node_id_for_paper(paper_id)
        edge_id = edge_id_for(EdgeKind.MENTIONS, source, node_id)
        if edge_id in edges:
            return
        edges[edge_id] = GraphEdge(
            id=edge_id,
            kind=EdgeKind.MENTIONS,
            source_id=source,
            target_id=node_id,
            provenance=GraphProvenance(
                evidence_ids=tuple(item.id for item in obj.evidence),
                paper_ids=(paper_id,),
                locations=tuple(item.location for item in obj.evidence),
                derived_by=self.name,
            ),
            confidence=obj.confidence,
        )

    def _add_relation(
        self,
        edges: dict[str, GraphEdge],
        relation: object,
        object_to_node: dict[str, str],
        paper_knowledge: PaperKnowledge,
    ) -> None:
        subject_id = getattr(relation, "subject_id", "")
        object_id = getattr(relation, "object_id", "")
        source = object_to_node.get(subject_id)
        target = object_to_node.get(object_id)
        if source is None or target is None or source == target:
            return

        kind = EdgeKind.for_predicate(getattr(relation, "predicate"))  # noqa: B009
        edge_id = edge_id_for(kind, source, target)
        evidence = tuple(getattr(relation, "evidence", ()))

        candidate = GraphEdge(
            id=edge_id,
            kind=kind,
            source_id=source,
            target_id=target,
            provenance=GraphProvenance(
                evidence_ids=tuple(item.id for item in evidence),
                paper_ids=(paper_knowledge.paper_id,),
                locations=tuple(item.location for item in evidence),
                derived_by=self.name,
            ),
            confidence=getattr(relation, "confidence", Confidence.unknown()),
        )
        existing = edges.get(edge_id)
        if existing is None or candidate.confidence.score > existing.confidence.score:
            edges[edge_id] = candidate

    def _add_contradiction(
        self,
        edges: dict[str, GraphEdge],
        contradiction: Contradiction,
        object_to_node: dict[str, str],
    ) -> None:
        """Disagreement becomes an edge, never a deletion.

        A graph that silently dropped one side of a conflict would manufacture consensus,
        which is the opposite of what a literature review is for.
        """
        source = object_to_node.get(contradiction.left_object_id)
        target = object_to_node.get(contradiction.right_object_id)
        if source is None or target is None or source == target:
            return

        edge_id = edge_id_for(EdgeKind.CONTRADICTS, source, target)
        locations: list[SourceLocation] = [
            item.location for item in (*contradiction.left_evidence, *contradiction.right_evidence)
        ]
        edges[edge_id] = GraphEdge(
            id=edge_id,
            kind=EdgeKind.CONTRADICTS,
            source_id=source,
            target_id=target,
            provenance=GraphProvenance(
                evidence_ids=tuple(
                    item.id
                    for item in (*contradiction.left_evidence, *contradiction.right_evidence)
                ),
                paper_ids=(contradiction.left_paper_id, contradiction.right_paper_id),
                locations=tuple(locations),
                derived_by=contradiction.detected_by,
            ),
            confidence=contradiction.confidence,
        )


def _properties_for(obj: KnowledgeObject) -> GraphNodeProperties:
    details = obj.details
    if isinstance(details, ResultDetails):
        return GraphNodeProperties(
            description=obj.description[:500],
            metric_name=details.metric_name,
            dataset_name=details.dataset_name,
            numeric_value=details.numeric_value,
            unit=details.unit,
        )
    if isinstance(details, MetricDetails):
        return GraphNodeProperties(description=obj.description[:500], unit=details.unit)
    if isinstance(details, DatasetDetails):
        return GraphNodeProperties(description=obj.description[:500])
    return GraphNodeProperties(description=obj.description[:500])
