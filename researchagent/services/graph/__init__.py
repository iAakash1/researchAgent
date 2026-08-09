"""Knowledge graph: a derived, provenance-aware index over validated knowledge.

Neo4j is an adapter. The knowledge and evidence repositories remain the source of truth,
and the graph is reconstructible from them at any time.
"""

from researchagent.services.graph.builder import GraphBuilder, GraphBuildReport
from researchagent.services.graph.mapper import GraphMapper
from researchagent.services.graph.queries import (
    ConnectedEntity,
    ContradictionPair,
    GraphQueries,
    SharedEntity,
)
from researchagent.services.graph.validator import GraphValidationReport, GraphValidator

__all__ = [
    "ConnectedEntity",
    "ContradictionPair",
    "GraphBuildReport",
    "GraphBuilder",
    "GraphMapper",
    "GraphQueries",
    "GraphValidationReport",
    "GraphValidator",
    "SharedEntity",
]
