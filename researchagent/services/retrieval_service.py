"""PDF retrieval.

Separate from discovery on purpose: discovery is cheap and repeatable, downloading is
neither. A run decides *what* to read before it spends bandwidth reading it.

Writes to ``storage/papers/raw/downloaded/`` and records the path on the paper's library
record. The manual collection is never written to.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from researchagent.config.schemas import RetrievalSettings
from researchagent.core.exceptions import PaperNotFoundError, PaperSourceError
from researchagent.core.interfaces.paper_source import PaperSource
from researchagent.core.interfaces.repositories import PaperRepository
from researchagent.core.logging import get_logger
from researchagent.models.library import PaperRecord, storage_key_for
from researchagent.models.paper import Paper, SourceName

logger = get_logger(__name__)


class RetrievalOutcome(BaseModel):
    paper_id: str
    downloaded: bool
    path: Path | None = None
    reason: str | None = None


class RetrievalResult(BaseModel):
    outcomes: list[RetrievalOutcome] = Field(default_factory=list)

    @property
    def downloaded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.downloaded)

    @property
    def skipped(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.downloaded)


class RetrievalService:
    def __init__(
        self,
        sources: dict[SourceName, PaperSource],
        repository: PaperRepository,
        download_dir: Path,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._sources = sources
        self._repository = repository
        self._download_dir = download_dir
        self._settings = settings or RetrievalSettings()

    async def retrieve(self, papers: list[Paper]) -> RetrievalResult:
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_downloads)

        async def guarded(paper: Paper) -> RetrievalOutcome:
            async with semaphore:
                return await self._retrieve_one(paper)

        outcomes = await asyncio.gather(*(guarded(paper) for paper in papers))
        result = RetrievalResult(outcomes=list(outcomes))
        logger.info(
            "retrieval_complete",
            requested=len(papers),
            downloaded=result.downloaded,
            skipped=result.skipped,
        )
        return result

    async def _retrieve_one(self, paper: Paper) -> RetrievalOutcome:
        # Already on disk (manual library, or a previous run): nothing to fetch.
        if paper.local_path is not None and paper.local_path.is_file():
            await self._record(paper, paper.local_path)
            return RetrievalOutcome(
                paper_id=paper.id, downloaded=False, path=paper.local_path, reason="already_local"
            )

        destination = self._download_dir / f"{storage_key_for(paper.id)}.pdf"
        if self._settings.skip_existing and destination.is_file():
            await self._record(paper, destination)
            return RetrievalOutcome(
                paper_id=paper.id, downloaded=False, path=destination, reason="already_downloaded"
            )

        source = self._sources.get(paper.provider)
        if source is None or not source.supports_download:
            return RetrievalOutcome(
                paper_id=paper.id, downloaded=False, reason="source_cannot_download"
            )
        if not paper.pdf_url:
            return RetrievalOutcome(paper_id=paper.id, downloaded=False, reason="no_pdf_url")

        try:
            path = await source.download_pdf(paper, destination)
        except (PaperSourceError, PaperNotFoundError) as exc:
            logger.warning(
                "pdf_download_failed",
                paper_id=paper.id,
                source=paper.provider.value,
                error_code=exc.code,
            )
            await self._record_error(paper, exc.message)
            return RetrievalOutcome(paper_id=paper.id, downloaded=False, reason=exc.code)

        await self._record(paper, path)
        return RetrievalOutcome(paper_id=paper.id, downloaded=True, path=path)

    async def _record(self, paper: Paper, path: Path) -> None:
        record = await self._repository.get(paper.id) or PaperRecord(paper=paper)
        await self._repository.save(
            record.model_copy(
                update={
                    "pdf_path": path,
                    "processing": record.processing.mark(downloaded=True, last_error=None),
                }
            )
        )

    async def _record_error(self, paper: Paper, message: str) -> None:
        record = await self._repository.get(paper.id)
        if record is None:
            return
        await self._repository.save(
            record.model_copy(update={"processing": record.processing.mark(last_error=message)})
        )
