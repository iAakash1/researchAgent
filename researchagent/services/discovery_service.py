"""Literature discovery.

Fans a research plan out across every enabled provider, normalises, deduplicates and
ranks the results, and persists what it found.

Two properties matter more than throughput:

* **Partial failure is normal.** Five public APIs will not all be up. One provider being
  down must degrade the result set, never fail the run — so provider errors are collected
  and reported, not raised.
* **Providers run concurrently, queries within a provider do not.** Each provider owns a
  rate limiter; firing its queries in parallel would only queue behind that limiter while
  looking like progress.

Does not download PDFs. Retrieval is a separate service and a separate decision.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from researchagent.config.schemas import DiscoverySettings
from researchagent.core.events import DiscoveryPayload, EventBus, EventType, PaperPayload
from researchagent.core.exceptions import PaperSourceError
from researchagent.core.interfaces.paper_source import PaperSource, SearchQuery
from researchagent.core.interfaces.repositories import PaperRepository
from researchagent.core.logging import get_logger
from researchagent.models.library import PaperRecord
from researchagent.models.paper import Paper, SourceName
from researchagent.models.research import ResearchPlan
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.ranking import PaperScorer, ScoredPaper

logger = get_logger(__name__)


class SourceReport(BaseModel):
    """What one provider contributed, and whether it worked."""

    source: SourceName
    papers_returned: int = 0
    queries_run: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class DiscoveryResult(BaseModel):
    candidates: list[ScoredPaper] = Field(default_factory=list)
    reports: list[SourceReport] = Field(default_factory=list)
    total_returned: int = 0
    duplicates_removed: int = 0

    @property
    def sources_succeeded(self) -> list[SourceName]:
        return [r.source for r in self.reports if r.succeeded]

    @property
    def sources_failed(self) -> list[SourceName]:
        return [r.source for r in self.reports if not r.succeeded]


class DiscoveryService:
    def __init__(
        self,
        sources: list[PaperSource],
        deduplicator: PaperDeduplicator,
        scorer: PaperScorer,
        settings: DiscoverySettings,
        *,
        repository: PaperRepository | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._sources = sources
        self._deduplicator = deduplicator
        self._scorer = scorer
        self._settings = settings
        self._repository = repository
        self._event_bus = event_bus

    @property
    def source_names(self) -> list[SourceName]:
        return [source.name for source in self._sources]

    async def discover(self, plan: ResearchPlan, *, run_id: str | None = None) -> DiscoveryResult:
        if not self._sources:
            logger.warning("discovery_no_sources_enabled")
            return DiscoveryResult()

        queries = self._build_queries(plan)
        logger.info(
            "discovery_started",
            sources=[s.value for s in self.source_names],
            queries=len(queries),
        )

        outcomes = await asyncio.gather(
            *(self._search_source(source, queries) for source in self._sources)
        )

        papers = [paper for _, source_papers in outcomes for paper in source_papers]
        reports = [report for report, _ in outcomes]

        if self._settings.require_retrievable:
            papers = [paper for paper in papers if paper.is_retrievable]

        deduplicated = self._deduplicator.deduplicate(papers)
        candidates = self._scorer.rank(
            deduplicated.papers, plan, limit=self._settings.max_candidates
        )
        if self._settings.require_retrievable is False:
            candidates = [c for c in candidates if c.score >= 0.0]

        if self._repository is not None:
            await self._persist(candidates, run_id)

        logger.info(
            "discovery_complete",
            returned=len(papers),
            unique=deduplicated.total,
            candidates=len(candidates),
            failed_sources=[s.value for s in _failed(reports)],
        )
        result = DiscoveryResult(
            candidates=candidates,
            reports=reports,
            total_returned=len(papers),
            duplicates_removed=deduplicated.duplicates_removed,
        )
        await self._emit(result, run_id)
        return result

    async def _emit(self, result: DiscoveryResult, run_id: str | None) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.DISCOVERY_COMPLETED,
            DiscoveryPayload(
                sources_queried=tuple(r.source.value for r in result.reports),
                sources_failed=tuple(s.value for s in result.sources_failed),
                papers_returned=result.total_returned,
                duplicates_removed=result.duplicates_removed,
                candidates=len(result.candidates),
            ),
            run_id=run_id,
            source="discovery_service",
        )
        for candidate in result.candidates:
            paper = candidate.paper
            await self._event_bus.emit(
                EventType.PAPER_MERGED if paper.also_seen_in else EventType.PAPER_DISCOVERED,
                PaperPayload(
                    paper_id=paper.id,
                    provider=paper.provider.value,
                    title=paper.title[:200],
                    merged_from=tuple(s.value for s in paper.also_seen_in),
                ),
                run_id=run_id,
                source="discovery_service",
            )

    def _build_queries(self, plan: ResearchPlan) -> list[SearchQuery]:
        keywords = [kw for question in plan.research_questions for kw in question.keywords]
        return [
            SearchQuery(
                text=query,
                limit=self._settings.results_per_query,
                year_from=plan.strategy.year_from,
                terms=keywords[:8],
            )
            for query in plan.strategy.queries[: self._settings.max_queries]
        ]

    async def _search_source(
        self, source: PaperSource, queries: list[SearchQuery]
    ) -> tuple[SourceReport, list[Paper]]:
        started = datetime.now(UTC)
        papers: list[Paper] = []
        queries_run = 0
        error: str | None = None

        for query in queries:
            try:
                papers.extend(await source.search(query))
                queries_run += 1
            except PaperSourceError as exc:
                # Record and stop querying this provider; continuing would just collect
                # more failures against a rate limit or an outage.
                error = f"{exc.code}: {exc.message}"
                logger.warning(
                    "source_search_failed",
                    source=source.name.value,
                    query=query.text,
                    error_code=exc.code,
                    error=exc.message,
                )
                break

        latency_ms = (datetime.now(UTC) - started).total_seconds() * 1000
        return (
            SourceReport(
                source=source.name,
                papers_returned=len(papers),
                queries_run=queries_run,
                latency_ms=latency_ms,
                error=error,
            ),
            papers,
        )

    async def _persist(self, candidates: list[ScoredPaper], run_id: str | None) -> None:
        records = [
            PaperRecord(
                paper=candidate.paper,
                pdf_path=candidate.paper.local_path,
                run_ids=[run_id] if run_id else [],
            )
            for candidate in candidates
        ]
        # Manual papers are already on disk, so their record starts life downloaded.
        for record in records:
            if record.pdf_path is not None:
                record.processing = record.processing.mark(downloaded=True)

        saved = await self._repository.save_many(records)  # type: ignore[union-attr]
        logger.info("papers_persisted", count=len(saved))


def _failed(reports: list[SourceReport]) -> list[SourceName]:
    return [report.source for report in reports if not report.succeeded]
