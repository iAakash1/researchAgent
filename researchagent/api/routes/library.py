"""Paper library endpoints: what has been discovered, and fetching the PDFs."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from researchagent.api.dependencies import (
    ContainerDep,
    PaperRepositoryDep,
    RetrievalServiceDep,
)
from researchagent.core.exceptions import PaperNotFoundError
from researchagent.core.interfaces.paper_source import SourceHealth
from researchagent.core.logging import get_logger
from researchagent.models.library import PaperRecord
from researchagent.models.paper import SourceName
from researchagent.services.retrieval_service import RetrievalResult

router = APIRouter(prefix="/library", tags=["library"])
logger = get_logger(__name__)


class LibrarySummary(BaseModel):
    total: int
    by_source: dict[str, int]
    downloaded: int
    pending_parse: int


class RetrieveRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1, max_length=200)


@router.get("/papers", response_model=list[PaperRecord])
async def list_papers(
    repository: PaperRepositoryDep,
    source: SourceName | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[PaperRecord]:
    records = await repository.list_all()
    if source is not None:
        records = [record for record in records if record.paper.provider is source]
    return records[:limit]


@router.get("/papers/{paper_id:path}", response_model=PaperRecord)
async def get_paper(paper_id: str, repository: PaperRepositoryDep) -> PaperRecord:
    record = await repository.get(paper_id)
    if record is None:
        raise PaperNotFoundError("No record for this paper", paper_id=paper_id)
    return record


@router.get("/summary", response_model=LibrarySummary)
async def summary(repository: PaperRepositoryDep) -> LibrarySummary:
    records = await repository.list_all()
    by_source: dict[str, int] = {}
    for record in records:
        key = record.paper.provider.value
        by_source[key] = by_source.get(key, 0) + 1

    return LibrarySummary(
        total=len(records),
        by_source=by_source,
        downloaded=sum(1 for r in records if r.processing.downloaded),
        pending_parse=sum(1 for r in records if not r.processing.parsed),
    )


@router.get("/sources", response_model=list[SourceHealth])
async def source_health(container: ContainerDep) -> list[SourceHealth]:
    """Live probe of every enabled provider."""
    return [await source.health() for source in container.paper_sources]


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(
    request: RetrieveRequest,
    repository: PaperRepositoryDep,
    retrieval: RetrievalServiceDep,
) -> RetrievalResult:
    """Download PDFs for known papers.

    Deliberately explicit and separate from discovery: discovery is cheap and repeatable,
    downloading is neither.
    """
    records = [record for pid in request.paper_ids if (record := await repository.get(pid))]
    if not records:
        raise PaperNotFoundError(
            "None of the requested papers are in the library",
            requested=len(request.paper_ids),
        )
    return await retrieval.retrieve([record.paper for record in records])
