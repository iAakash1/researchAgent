"""Evidence and retrieval endpoints.

Exposes the four retrieval layers and the bundle store. Deliberately read-only: this
release retrieves and assembles evidence, it does not reason over it.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from researchagent.api.dependencies import ContainerDep
from researchagent.core.exceptions import PaperNotFoundError
from researchagent.core.interfaces.retrieval import (
    BundleHits,
    DocumentHits,
    EvidenceHits,
    KnowledgeHits,
)
from researchagent.core.logging import get_logger
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.knowledge import KnowledgeKind
from researchagent.models.query import QueryIntent, ResearchQuery

router = APIRouter(prefix="/evidence", tags=["evidence"])
logger = get_logger(__name__)


class RetrieveRequest(BaseModel):
    """A retrieval request. The same shape reaches every layer."""

    text: str = Field(min_length=1, examples=["What triggers metastable failures?"])
    intent: QueryIntent = QueryIntent.ANSWER
    kinds: tuple[KnowledgeKind, ...] = ()
    paper_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=25, ge=1, le=200)

    def to_query(self) -> ResearchQuery:
        return ResearchQuery(
            text=self.text,
            intent=self.intent,
            kinds=self.kinds,
            paper_ids=self.paper_ids,
            terms=self.terms,
            min_confidence=self.min_confidence,
            limit=self.limit,
        )


@router.post("/retrieve/knowledge", response_model=KnowledgeHits)
async def retrieve_knowledge(request: RetrieveRequest, container: ContainerDep) -> KnowledgeHits:
    """Layer 1 — structured facts."""
    return await container.knowledge_retriever.retrieve(request.to_query())


@router.post("/retrieve/evidence", response_model=EvidenceHits)
async def retrieve_evidence(request: RetrieveRequest, container: ContainerDep) -> EvidenceHits:
    """Layer 2 — the quotes supporting them, with page and paragraph."""
    return await container.evidence_retriever.retrieve(request.to_query())


@router.post("/retrieve/documents", response_model=DocumentHits)
async def retrieve_documents(request: RetrieveRequest, container: ContainerDep) -> DocumentHits:
    """Layer 3 — the canonical documents behind the quotes."""
    return await container.document_retriever.retrieve(request.to_query())


@router.post("/retrieve/cross-paper", response_model=KnowledgeHits)
async def retrieve_cross_paper(request: RetrieveRequest, container: ContainerDep) -> KnowledgeHits:
    """Layer 4 — the same entity as several papers describe it."""
    return await container.cross_paper_retriever.retrieve(request.to_query())


@router.post("/bundles/search", response_model=BundleHits)
async def search_bundles(request: RetrieveRequest, container: ContainerDep) -> BundleHits:
    return await container.bundle_retriever.retrieve(request.to_query())


@router.post("/bundles/build", response_model=EvidenceBundle)
async def build_bundle(request: RetrieveRequest, container: ContainerDep) -> EvidenceBundle:
    """Assemble a bundle for an ad-hoc query, without running the whole workflow."""
    bundle = await container.evidence_service.build_bundle(request.to_query())
    await container.bundle_repository.save(bundle)
    return bundle


@router.get("/bundles/{bundle_id:path}", response_model=EvidenceBundle)
async def get_bundle(bundle_id: str, container: ContainerDep) -> EvidenceBundle:
    bundle = await container.bundle_repository.get(bundle_id)
    if bundle is None:
        raise PaperNotFoundError("No such bundle", bundle_id=bundle_id)
    return bundle


@router.get("/papers/{paper_id:path}/evidence")
async def paper_evidence(
    paper_id: str, container: ContainerDep, limit: int = Query(default=200, ge=1, le=1000)
) -> dict[str, object]:
    """Every indexed evidence item for one paper, with its provenance."""
    stored = await container.evidence_repository.get_paper(paper_id)
    if stored is None:
        raise PaperNotFoundError("No evidence indexed for this paper", paper_id=paper_id)
    return {
        "paper_id": stored.paper_id,
        "document_sha256": stored.document_sha256,
        "sections_covered": list(stored.sections_covered),
        "records": [record.model_dump(mode="json") for record in stored.records[:limit]],
    }
