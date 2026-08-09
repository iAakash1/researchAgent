"""The knowledge graph model.

A derived index over validated knowledge, never a second source of truth. Every node
corresponds to a KnowledgeObject or a Paper that already exists in a repository, and
every trusted edge corresponds to a KnowledgeRelation that was derived from evidence in
v0.5. Delete the graph and it rebuilds; delete the repositories and nothing can.

Identity is domain-derived and stable, so constructing the graph twice over the same
corpus produces the same graph. Database-generated ids are never domain identity.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.core.evidence import SourceLocation
from researchagent.core.validation import Confidence
from researchagent.models.knowledge import KnowledgeKind, RelationPredicate


class NodeKind(StrEnum):
    """Node types. One per validated domain object — no speculative types.

    ``PAPER`` is the only node not derived from a KnowledgeObject; it exists because
    every fact needs something to hang provenance from, and papers are already validated
    artefacts in the paper repository.
    """

    PAPER = "Paper"
    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"
    RESULT = "Result"
    LIMITATION = "Limitation"
    FUTURE_WORK = "FutureWork"

    @classmethod
    def for_knowledge(cls, kind: KnowledgeKind) -> NodeKind:
        return {
            KnowledgeKind.METHOD: cls.METHOD,
            KnowledgeKind.DATASET: cls.DATASET,
            KnowledgeKind.METRIC: cls.METRIC,
            KnowledgeKind.RESULT: cls.RESULT,
            KnowledgeKind.LIMITATION: cls.LIMITATION,
            KnowledgeKind.FUTURE_WORK: cls.FUTURE_WORK,
        }[kind]


class EdgeKind(StrEnum):
    """Edge types.

    The first group mirrors the evidence-derived KnowledgeRelations from v0.5. The
    ``MENTIONS`` and ``CONTRADICTS`` edges are structural: the first attaches every fact
    to the paper that stated it, the second records a disagreement the v0.6 detector
    found mechanically. Nothing here is invented by a model.
    """

    MENTIONS = "MENTIONS"  # Paper -> any knowledge node
    EVALUATED_ON = "EVALUATED_ON"  # Method -> Dataset
    MEASURED_BY = "MEASURED_BY"  # Result -> Metric
    REPORTED_ON = "REPORTED_ON"  # Result -> Dataset
    PRODUCED_BY = "PRODUCED_BY"  # Result -> Method
    LIMITS = "LIMITS"  # Limitation -> Method
    EXTENDS = "EXTENDS"  # FutureWork -> Limitation
    CONTRADICTS = "CONTRADICTS"  # knowledge node <-> knowledge node

    @classmethod
    def for_predicate(cls, predicate: RelationPredicate) -> EdgeKind:
        return cls(predicate.value.upper())


class GraphProvenance(BaseModel):
    """Where an edge or node came from. An edge without this is never trusted."""

    model_config = {"frozen": True}

    evidence_ids: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()
    locations: tuple[SourceLocation, ...] = ()
    derived_by: str = Field(min_length=1)

    @property
    def is_grounded(self) -> bool:
        """Whether this can be traced back to a quoted sentence in a document."""
        return bool(self.evidence_ids) and bool(self.locations)

    def cite(self) -> tuple[str, ...]:
        return tuple(location.describe() for location in self.locations)


class GraphNode(BaseModel):
    """One entity in the graph, mirroring a validated domain object."""

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    kind: NodeKind
    name: str = Field(min_length=1)
    # The object this node mirrors. Following it recovers the full validated record,
    # because the graph deliberately stores a projection rather than a copy.
    knowledge_object_id: str | None = None
    paper_ids: tuple[str, ...] = ()
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    evidence_ids: tuple[str, ...] = ()
    properties: GraphNodeProperties = Field(default_factory=lambda: GraphNodeProperties())

    @property
    def is_knowledge_node(self) -> bool:
        return self.kind is not NodeKind.PAPER


class GraphNodeProperties(BaseModel):
    """Denormalised attributes for querying. Typed, never a free dict."""

    model_config = {"frozen": True}

    description: str = ""
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    metric_name: str | None = None
    dataset_name: str | None = None
    numeric_value: float | None = None
    unit: str | None = None


class GraphEdge(BaseModel):
    """One relationship, with the evidence that justifies it."""

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    kind: EdgeKind
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    provenance: GraphProvenance
    confidence: Confidence = Field(default_factory=Confidence.unknown)

    @model_validator(mode="after")
    def _no_self_loop(self) -> GraphEdge:
        if self.source_id == self.target_id:
            raise ValueError("a graph edge cannot point at itself")
        return self

    @property
    def is_trusted(self) -> bool:
        """No provenance, no trust. The rule the whole architecture rests on."""
        return self.provenance.is_grounded


class GraphVersion(BaseModel):
    """Identifies a graph generation.

    Any change to the corpus, extraction, relation derivation or schema produces a new
    version, so two generations can never silently mix.
    """

    model_config = {"frozen": True}

    schema_version: str = "1"
    corpus_fingerprint: str = Field(min_length=1)
    extraction_version: str = "1"
    relation_version: str = "1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def identifier(self) -> str:
        return (
            f"s{self.schema_version}-x{self.extraction_version}"
            f"-r{self.relation_version}-c{self.corpus_fingerprint}"
        )


class KnowledgeGraph(BaseModel):
    """A complete, versioned graph generation."""

    model_config = {"frozen": True}

    version: GraphVersion
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    def node(self, node_id: str) -> GraphNode | None:
        return next((node for node in self.nodes if node.id == node_id), None)

    def nodes_of(self, kind: NodeKind) -> tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if node.kind is kind)

    def edges_of(self, kind: EdgeKind) -> tuple[GraphEdge, ...]:
        return tuple(edge for edge in self.edges if edge.kind is kind)

    @property
    def trusted_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(edge for edge in self.edges if edge.is_trusted)

    @property
    def provenance_coverage(self) -> float:
        """Share of edges that can be traced to a quoted sentence."""
        return len(self.trusted_edges) / len(self.edges) if self.edges else 0.0


def node_id_for_paper(paper_id: str) -> str:
    return f"paper:{paper_id}"


def node_id_for_knowledge(kind: KnowledgeKind, name: str, paper_id: str | None = None) -> str:
    """Stable, name-derived identity.

    Deliberately *not* the KnowledgeObject id: two papers describing "MIMIC-III" must
    become one node, which is what makes cross-paper questions answerable. The objects
    behind the node are kept on it, so nothing is lost by merging.

    Claim-like kinds are the exception and stay scoped to their paper. A *result* is a
    measurement one paper reports, not a shared entity: merging "F1 = 0.82" from one paper
    with "F1 = 0.41" from another would collapse a disagreement into a single node and
    turn the CONTRADICTS edge between them into a self-loop — silently deleting exactly
    the finding the graph exists to surface. Same for limitations and future work, which
    are the authors' own claims rather than named artefacts.
    """
    slug = "-".join(name.lower().split())[:60] or "unnamed"
    scope = f"{paper_id}|{slug}" if kind.is_claim_like and paper_id else slug
    digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:10]  # noqa: S324
    prefix = f"{kind.value}:{paper_id}:" if kind.is_claim_like and paper_id else f"{kind.value}:"
    return f"{prefix}{slug}:{digest}"


def edge_id_for(kind: EdgeKind, source_id: str, target_id: str) -> str:
    """Deterministic: re-running construction cannot duplicate an edge."""
    return f"{source_id}--{kind.value}--{target_id}"


def corpus_fingerprint(paper_ids: tuple[str, ...], object_count: int) -> str:
    """Identifies the corpus a graph was built from."""
    payload = "|".join(sorted(paper_ids)) + f"#{object_count}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]  # noqa: S324
