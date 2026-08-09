"""Knowledge graph endpoints.

Domain-level operations only. There is deliberately no endpoint that accepts a Cypher
string: the graph is a derived index over validated knowledge, and an arbitrary query
surface would let a caller read it in ways the provenance model cannot describe.

Every relationship returned carries its citations, so an answer can be checked against the
paragraph that produced it rather than believed.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from researchagent.api.dependencies import GraphBuilderDep, GraphQueriesDep, GraphRepositoryDep
from researchagent.core.exceptions import GraphNotBuiltError
from researchagent.core.interfaces.graph_repository import GraphStats, Subgraph
from researchagent.core.logging import get_logger
from researchagent.models.graph import GraphNode, GraphVersion, NodeKind
from researchagent.services.graph.builder import GraphBuildReport
from researchagent.services.graph.queries import ConnectedEntity, ContradictionPair, SharedEntity

router = APIRouter(prefix="/graph", tags=["graph"])
logger = get_logger(__name__)


class BuildRequest(BaseModel):
    """Rebuild the derived index. Empty ``paper_ids`` means the whole corpus."""

    paper_ids: tuple[str, ...] = ()


class VersionsResponse(BaseModel):
    current: GraphVersion | None = None
    versions: tuple[GraphVersion, ...] = ()


class RelationshipEvidence(BaseModel):
    """The provenance chain for one relationship, flattened for transport."""

    source_id: str
    target_id: str
    citations: tuple[str, ...] = ()
    grounded: bool = Field(description="False means the edge cannot be traced to a quote")


async def _current(repository: GraphRepositoryDep) -> GraphVersion:
    """Resolve "the graph" to a concrete generation.

    Raising rather than silently returning empty results: "no graph has been built" and
    "the graph contains nothing matching your query" are different answers, and a caller
    that cannot tell them apart will misreport both.
    """
    versions = await repository.versions()
    if not versions:
        raise GraphNotBuiltError("No knowledge graph generation exists yet")
    return versions[0]


@router.post("/build", response_model=GraphBuildReport)
async def build_graph(
    request: BuildRequest, builder: GraphBuilderDep, run_id: str | None = None
) -> GraphBuildReport:
    """Rebuild the graph from the knowledge repository.

    Idempotent: the same corpus yields the same version identifier and the same
    deterministic node and edge ids, so running this twice replaces a generation rather
    than duplicating it.
    """
    return await builder.build(request.paper_ids, run_id=run_id)


@router.get("/versions", response_model=VersionsResponse)
async def list_versions(repository: GraphRepositoryDep) -> VersionsResponse:
    versions = await repository.versions()
    return VersionsResponse(current=versions[0] if versions else None, versions=versions)


@router.get("/stats", response_model=GraphStats)
async def graph_stats(repository: GraphRepositoryDep) -> GraphStats:
    return await repository.stats(await _current(repository))


@router.get("/methods/{method_name}/datasets", response_model=tuple[ConnectedEntity, ...])
async def datasets_for_method(
    method_name: str, queries: GraphQueriesDep, repository: GraphRepositoryDep
) -> tuple[ConnectedEntity, ...]:
    """Which datasets was this method evaluated on?"""
    return await queries.datasets_for_method(method_name, await _current(repository))


@router.get("/datasets/{dataset_name}/methods", response_model=tuple[ConnectedEntity, ...])
async def methods_for_dataset(
    dataset_name: str, queries: GraphQueriesDep, repository: GraphRepositoryDep
) -> tuple[ConnectedEntity, ...]:
    """Which methods were evaluated on this dataset?"""
    return await queries.methods_for_dataset(dataset_name, await _current(repository))


@router.get("/methods/{method_name}/metrics", response_model=tuple[ConnectedEntity, ...])
async def metrics_for_method(
    method_name: str, queries: GraphQueriesDep, repository: GraphRepositoryDep
) -> tuple[ConnectedEntity, ...]:
    """Which metrics were used to evaluate this method? Two hops through Result."""
    return await queries.metrics_for_method(method_name, await _current(repository))


@router.get("/entities/{entity_name}/papers", response_model=tuple[GraphNode, ...])
async def papers_using(
    entity_name: str, queries: GraphQueriesDep, repository: GraphRepositoryDep
) -> tuple[GraphNode, ...]:
    """Which papers mention this entity?"""
    return await queries.papers_using(entity_name, await _current(repository))


@router.get("/shared", response_model=tuple[SharedEntity, ...])
async def shared_entities(
    queries: GraphQueriesDep,
    repository: GraphRepositoryDep,
    kind: NodeKind | None = None,
    minimum_papers: int = Query(default=2, ge=2, le=50),
) -> tuple[SharedEntity, ...]:
    """Entities several papers have in common — the cross-paper view."""
    return await queries.entities_across_papers(
        await _current(repository), kind=kind, minimum_papers=minimum_papers
    )


@router.get("/contradictions", response_model=tuple[ContradictionPair, ...])
async def contradictions(
    queries: GraphQueriesDep, repository: GraphRepositoryDep
) -> tuple[ContradictionPair, ...]:
    """Where the literature disagrees. Both sides are returned; neither is resolved."""
    return await queries.contradictions(await _current(repository))


@router.get("/nodes/{node_id:path}/neighbours", response_model=Subgraph)
async def neighbours(
    node_id: str,
    repository: GraphRepositoryDep,
    depth: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=50, ge=1, le=200),
) -> Subgraph:
    """The subgraph around one entity."""
    return await repository.neighbours(
        node_id, await _current(repository), depth=depth, limit=limit
    )


@router.get("/provenance", response_model=RelationshipEvidence)
async def provenance(
    source_id: str,
    target_id: str,
    queries: GraphQueriesDep,
    repository: GraphRepositoryDep,
) -> RelationshipEvidence:
    """The evidence supporting a relationship, addressed down to page and paragraph."""
    citations = await queries.provenance_for(source_id, target_id, await _current(repository))
    return RelationshipEvidence(
        source_id=source_id,
        target_id=target_id,
        citations=citations,
        grounded=bool(citations),
    )
