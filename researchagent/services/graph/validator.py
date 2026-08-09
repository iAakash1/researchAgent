"""Graph validation.

The last gate before knowledge becomes a queryable index. Invalid nodes and edges are
rejected *and the reason is recorded* — a graph that silently dropped bad data would be
indistinguishable from a graph built from a smaller corpus.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.core.logging import get_logger
from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationIssue,
    ValidationResult,
)
from researchagent.models.graph import EdgeKind, GraphEdge, KnowledgeGraph, NodeKind

logger = get_logger(__name__)

# Edge type -> (allowed source kinds, allowed target kinds). MENTIONS and CONTRADICTS are
# checked separately because their endpoints are structural rather than typed.
_EDGE_TYPES: dict[EdgeKind, tuple[tuple[NodeKind, ...], tuple[NodeKind, ...]]] = {
    EdgeKind.EVALUATED_ON: ((NodeKind.METHOD,), (NodeKind.DATASET,)),
    EdgeKind.MEASURED_BY: ((NodeKind.RESULT,), (NodeKind.METRIC,)),
    EdgeKind.REPORTED_ON: ((NodeKind.RESULT,), (NodeKind.DATASET,)),
    EdgeKind.PRODUCED_BY: ((NodeKind.RESULT,), (NodeKind.METHOD,)),
    EdgeKind.LIMITS: ((NodeKind.LIMITATION,), (NodeKind.METHOD,)),
    EdgeKind.EXTENDS: ((NodeKind.FUTURE_WORK,), (NodeKind.LIMITATION,)),
}


class RejectedEdge(BaseModel):
    """An edge that did not make it, and why."""

    model_config = {"frozen": True}

    edge_id: str
    kind: EdgeKind
    reason: str


class GraphValidationReport(BaseModel):
    model_config = {"frozen": True}

    result: ValidationResult
    accepted_edges: tuple[GraphEdge, ...] = ()
    rejected_edges: tuple[RejectedEdge, ...] = ()
    untrusted_edges: int = Field(default=0, ge=0)

    @property
    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejected in self.rejected_edges:
            counts[rejected.reason] = counts.get(rejected.reason, 0) + 1
        return counts


class GraphValidator:
    """Checks node identity, edge endpoints, edge typing and provenance."""

    name = "graph_validator"

    def __init__(self, *, require_provenance: bool = True) -> None:
        # Configurable, but the default is the zero-trust one. Turning it off produces a
        # graph whose edges cannot be cited, which is why nothing does.
        self._require_provenance = require_provenance

    def validate(self, graph: KnowledgeGraph) -> GraphValidationReport:
        issues: list[ValidationIssue] = []
        node_kinds = {node.id: node.kind for node in graph.nodes}

        duplicates = len(graph.nodes) - len(node_kinds)
        if duplicates:
            issues.append(
                ValidationIssue.error(
                    "duplicate_node_ids",
                    f"{duplicates} node id(s) appear more than once",
                    field="nodes",
                    remedy="Node identity must be derived deterministically from the domain",
                )
            )

        accepted: list[GraphEdge] = []
        rejected: list[RejectedEdge] = []
        untrusted = 0

        for edge in graph.edges:
            reason = _edge_problem(edge, node_kinds)
            if reason is not None:
                rejected.append(RejectedEdge(edge_id=edge.id, kind=edge.kind, reason=reason))
                continue
            if self._require_provenance and not edge.is_trusted:
                # Kept out of the graph: an edge with no traceable evidence is exactly what
                # the zero-trust rule exists to exclude.
                untrusted += 1
                rejected.append(
                    RejectedEdge(edge_id=edge.id, kind=edge.kind, reason="no_provenance")
                )
                continue
            accepted.append(edge)

        if rejected:
            logger.info(
                "graph_edges_rejected",
                rejected=len(rejected),
                accepted=len(accepted),
                untrusted=untrusted,
            )

        signals = [
            ConfidenceSignal(
                name="edge_acceptance",
                value=(len(accepted) / len(graph.edges)) if graph.edges else 0.0,
                observation=f"{len(accepted)} of {len(graph.edges)} edges passed validation",
            ),
            ConfidenceSignal(
                name="provenance_coverage",
                value=graph.provenance_coverage,
                observation=(
                    f"{len(graph.trusted_edges)} of {len(graph.edges)} edges trace to evidence"
                ),
            ),
        ]

        if not graph.nodes:
            issues.append(
                ValidationIssue.error(
                    "empty_graph",
                    "No nodes were produced",
                    field="nodes",
                    remedy="Check that validated knowledge exists in the repository",
                )
            )

        return GraphValidationReport(
            result=ValidationResult.decide(
                validator=self.name,
                subject_id=graph.version.identifier,
                subject_type="KnowledgeGraph",
                confidence=Confidence.from_signals(signals),
                issues=issues,
            ),
            accepted_edges=tuple(accepted),
            rejected_edges=tuple(rejected),
            untrusted_edges=untrusted,
        )


def _edge_problem(edge: GraphEdge, node_kinds: dict[str, NodeKind]) -> str | None:
    source = node_kinds.get(edge.source_id)
    target = node_kinds.get(edge.target_id)

    if source is None or target is None:
        return "dangling_endpoint"

    if edge.kind is EdgeKind.MENTIONS:
        if source is not NodeKind.PAPER or target is NodeKind.PAPER:
            return "mentions_must_run_paper_to_knowledge"
        return None

    if edge.kind is EdgeKind.CONTRADICTS:
        if source is NodeKind.PAPER or target is NodeKind.PAPER:
            return "contradiction_between_papers_not_knowledge"
        if source is not target:
            return "contradiction_between_different_kinds"
        return None

    allowed = _EDGE_TYPES.get(edge.kind)
    if allowed is None:
        return "unknown_edge_kind"
    if source not in allowed[0]:
        return "source_kind_mismatch"
    if target not in allowed[1]:
        return "target_kind_mismatch"
    return None
