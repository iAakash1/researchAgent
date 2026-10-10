"""Integrity refresh: registry lookup, classification, secondary PDF signal, caching."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from researchagent.config.schemas import IntegrityCacheSettings
from researchagent.core.events import EventBus, EventType, IntegrityPayload
from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.integrity import IntegrityRegistry
from researchagent.core.interfaces.repositories import (
    DocumentRepository,
    IntegrityObservationRepository,
    PaperRepository,
)
from researchagent.core.logging import get_logger
from researchagent.models.integrity import (
    IntegrityObservation,
    IntegrityStatus,
    ObservationBasis,
)
from researchagent.models.paper import PaperIdentifiers
from researchagent.services.integrity.classifier import UpdateClassifier
from researchagent.services.integrity.notices import RetractionNoticeScanner

logger = get_logger(__name__)


class IntegrityRefresh(BaseModel):
    """What one refresh learned about one paper."""

    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    # This refresh's observation, or the cached one it reused. May be uninformative.
    observed: IntegrityObservation
    # The newest informative observation now held; None if the paper was never resolved.
    effective: IntegrityObservation | None = None
    cached: bool = False

    @property
    def lookup_failed(self) -> bool:
        return not self.observed.is_informative


class IntegrityService:
    """Keeps a cached, timestamped integrity record for each paper."""

    name = "integrity_service"

    def __init__(
        self,
        registry: IntegrityRegistry,
        repository: IntegrityObservationRepository,
        classifier: UpdateClassifier,
        cache: IntegrityCacheSettings,
        *,
        papers: PaperRepository | None = None,
        documents: DocumentRepository | None = None,
        scanner: RetractionNoticeScanner | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._registry = registry
        self._repository = repository
        self._classifier = classifier
        self._cache = cache
        self._papers = papers
        self._documents = documents
        self._scanner = scanner
        self._event_bus = event_bus

    @property
    def registry_name(self) -> str:
        return self._registry.name

    async def refresh(
        self,
        paper_ids: Sequence[str],
        *,
        now: datetime | None = None,
        force: bool = False,
        run_id: str | None = None,
    ) -> tuple[IntegrityRefresh, ...]:
        """Observe each paper, reusing a cached observation while it is still fresh."""
        moment = now or datetime.now(UTC)
        refreshed: list[IntegrityRefresh] = []
        for paper_id in dict.fromkeys(paper_ids):
            latest = await self._repository.latest(paper_id)
            cached = latest is not None and not force and self._is_fresh(latest, moment)
            observed = (
                latest if cached and latest is not None else await self._observe(paper_id, moment)
            )
            refreshed.append(
                IntegrityRefresh(
                    paper_id=paper_id,
                    observed=observed,
                    effective=await self._repository.latest_informative(paper_id),
                    cached=cached,
                )
            )
            await self._emit(observed, cached=cached, run_id=run_id)
        return tuple(refreshed)

    async def history(self, paper_id: str) -> tuple[IntegrityObservation, ...]:
        return await self._repository.history(paper_id)

    async def _observe(self, paper_id: str, moment: datetime) -> IntegrityObservation:
        identifiers = await self._identifiers(paper_id)
        if identifiers.doi:
            lookup = await self._registry.lookup(identifiers.doi)
            observation = self._classifier.classify(lookup, paper_id=paper_id, retrieved_at=moment)
        else:
            observation = IntegrityObservation(
                paper_id=paper_id,
                identifier=_fallback_identifier(paper_id, identifiers),
                status=IntegrityStatus.UNKNOWN,
                basis=ObservationBasis.NO_LOOKUP_IDENTIFIER,
                registry=self._registry.name,
                observation=f"{self._registry.name} resolves DOIs; this paper has none",
                retrieved_at=moment,
            )
        await self._repository.append(observation)

        notice = await self._pdf_notice(paper_id)
        if notice is None or observation.status not in (
            IntegrityStatus.CURRENT,
            IntegrityStatus.UNKNOWN,
        ):
            return observation

        flagged = IntegrityObservation(
            paper_id=paper_id,
            identifier=observation.identifier,
            doi=observation.doi,
            status=IntegrityStatus.UNKNOWN,
            basis=ObservationBasis.PDF_NOTICE,
            registry=RetractionNoticeScanner.name,
            observation=(
                f"parsed PDF text contains {notice!r}; {observation.registry} reported "
                f"{observation.status.value} ({observation.basis.value})"
            ),
            retrieved_at=moment,
        )
        await self._repository.append(flagged)
        return flagged

    async def _pdf_notice(self, paper_id: str) -> str | None:
        if self._scanner is None or self._documents is None or not self._scanner.enabled:
            return None
        try:
            stored = await self._documents.get(paper_id)
        except RepositoryError as exc:
            logger.warning("pdf_notice_document_unreadable", paper_id=paper_id, error=exc.message)
            return None
        return self._scanner.scan(stored.value) if stored is not None else None

    async def _identifiers(self, paper_id: str) -> PaperIdentifiers:
        if self._papers is not None:
            try:
                record = await self._papers.get(paper_id)
            except RepositoryError as exc:
                logger.warning(
                    "integrity_paper_record_unreadable", paper_id=paper_id, error=exc.message
                )
                record = None
            if record is not None and record.paper.identifiers.any_present():
                return record.paper.identifiers
        return identifiers_from_paper_id(paper_id)

    def _is_fresh(self, observation: IntegrityObservation, moment: datetime) -> bool:
        hours = (
            self._cache.recheck_after_hours
            if observation.is_informative
            else self._cache.recheck_unknown_after_hours
        )
        return moment - observation.retrieved_at < timedelta(hours=hours)

    async def _emit(
        self, observation: IntegrityObservation, *, cached: bool, run_id: str | None
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.INTEGRITY_CHECKED,
            IntegrityPayload(
                paper_id=observation.paper_id,
                status=observation.status.value,
                basis=observation.basis.value,
                registry=observation.registry,
                cached=cached,
            ),
            run_id=run_id,
            source=self.name,
        )


_PREFIXES = {
    "doi": "doi",
    "arxiv": "arxiv_id",
    "pmid": "pubmed_id",
    "openalex": "openalex_id",
    "s2": "semantic_scholar_id",
}


def identifiers_from_paper_id(paper_id: str) -> PaperIdentifiers:
    """Recover identifiers from an id built by ``make_paper_id`` (``doi:10.1/x`` etc.)."""
    prefix, _, value = paper_id.partition(":")
    field = _PREFIXES.get(prefix)
    if field is None or not value:
        return PaperIdentifiers()
    return PaperIdentifiers.model_validate({field: value})


def _fallback_identifier(paper_id: str, identifiers: PaperIdentifiers) -> str:
    for prefix, field in _PREFIXES.items():
        value = getattr(identifiers, field)
        if value:
            return f"{prefix}:{value}"
    return paper_id
