"""Knowledge indexing pipeline.

    KnowledgeObject -> validate -> retrieval representation -> embed -> vector + metadata

The original object is never mutated: the representation is derived and the vector is
stored beside it, so the knowledge repository remains the source of truth and the index
is a rebuildable projection of it.

Untrusted knowledge is never indexed. Zero trust across stages means a fact the previous
stage rejected must not become retrievable by a different route.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from researchagent.config.schemas import IndexSettings
from researchagent.core.events import EventBus, EventType, IndexPayload
from researchagent.core.exceptions import EmbeddingError, VectorStoreError
from researchagent.core.interfaces.embeddings import EmbeddingModel, ModelIdentity
from researchagent.core.interfaces.knowledge_repository import KnowledgeRepository
from researchagent.core.interfaces.vector_store import (
    VectorMetadata,
    VectorRecord,
    VectorStore,
)
from researchagent.core.logging import get_logger
from researchagent.models.knowledge import KnowledgeObject
from researchagent.services.retrieval.representation import REPRESENTATION_VERSION, represent

logger = get_logger(__name__)


class IndexReport(BaseModel):
    """What indexing did, including what it could not do."""

    model_config = {"frozen": True}

    index_version: str = ""
    model_fingerprint: str = ""
    objects_indexed: int = Field(default=0, ge=0)
    papers_indexed: int = Field(default=0, ge=0)
    papers_skipped: tuple[str, ...] = ()
    embedding_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)
    succeeded: bool = True
    error: str | None = None

    @property
    def embedding_ms_per_object(self) -> float:
        return self.embedding_ms / self.objects_indexed if self.objects_indexed else 0.0


class KnowledgeIndexer:
    """Builds the semantic index from validated knowledge."""

    name = "knowledge_indexer"

    def __init__(
        self,
        embeddings: EmbeddingModel,
        store: VectorStore,
        knowledge: KnowledgeRepository,
        settings: IndexSettings | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._knowledge = knowledge
        self._settings = settings or IndexSettings()
        self._event_bus = event_bus

    def index_version(self, identity: ModelIdentity) -> str:
        """Any change that makes vectors incomparable produces a new version.

        Derived rather than configured, so it cannot drift from what actually produced
        the vectors.
        """
        return (
            f"{self._settings.schema_version}-"
            f"{identity.model_name.replace(':', '_').replace('/', '_')}-"
            f"d{identity.dimension}-r{REPRESENTATION_VERSION}"
        )

    async def build(
        self, paper_ids: tuple[str, ...] = (), *, run_id: str | None = None
    ) -> IndexReport:
        started = time.perf_counter()

        try:
            identity = await self._identity()
        except EmbeddingError as exc:
            logger.warning("index_build_unavailable", error_code=exc.code, reason=exc.message)
            return IndexReport(succeeded=False, error=f"{exc.code}: {exc.message}")

        version = self.index_version(identity)
        try:
            await self._store.ensure_collection(identity, version)
        except VectorStoreError as exc:
            return IndexReport(
                index_version=version,
                model_fingerprint=identity.fingerprint,
                succeeded=False,
                error=f"{exc.code}: {exc.message}",
            )

        objects, papers, skipped = await self._collect(paper_ids)
        if not objects:
            return IndexReport(
                index_version=version,
                model_fingerprint=identity.fingerprint,
                papers_skipped=skipped,
                total_ms=_ms(started),
            )

        embed_started = time.perf_counter()
        try:
            vectors = await self._embeddings.embed_batch([represent(obj).text for obj in objects])
        except EmbeddingError as exc:
            return IndexReport(
                index_version=version,
                model_fingerprint=identity.fingerprint,
                succeeded=False,
                error=f"{exc.code}: {exc.message}",
                total_ms=_ms(started),
            )
        embedding_ms = _ms(embed_started)

        records = [
            VectorRecord(
                id=obj.id,
                vector=vector,
                metadata=self._metadata(obj, identity, version),
            )
            for obj, vector in zip(objects, vectors, strict=True)
        ]
        try:
            await self._store.upsert(records)
        except VectorStoreError as exc:
            return IndexReport(
                index_version=version,
                model_fingerprint=identity.fingerprint,
                succeeded=False,
                error=f"{exc.code}: {exc.message}",
                total_ms=_ms(started),
            )

        report = IndexReport(
            index_version=version,
            model_fingerprint=identity.fingerprint,
            objects_indexed=len(records),
            papers_indexed=papers,
            papers_skipped=skipped,
            embedding_ms=embedding_ms,
            total_ms=_ms(started),
        )
        logger.info(
            "semantic_index_built",
            index_version=version,
            model=identity.fingerprint,
            objects=report.objects_indexed,
            papers=report.papers_indexed,
            embedding_ms=round(embedding_ms),
        )
        if self._event_bus is not None:
            await self._event_bus.emit(
                EventType.INDEX_BUILT,
                IndexPayload(
                    index_version=version,
                    model_fingerprint=identity.fingerprint,
                    objects=report.objects_indexed,
                    embedding_ms=embedding_ms,
                ),
                run_id=run_id,
                source=self.name,
            )
        return report

    async def _identity(self) -> ModelIdentity:
        try:
            return self._embeddings.model_identity()
        except EmbeddingError:
            # Dimension is discovered from the model rather than declared; one call fills it.
            await self._embeddings.embed_text("index warm up")
            return self._embeddings.model_identity()

    async def _collect(
        self, paper_ids: tuple[str, ...]
    ) -> tuple[list[KnowledgeObject], int, tuple[str, ...]]:
        wanted = paper_ids or tuple(
            key.replace("-", ":", 1) for key in await self._knowledge.list_ids()
        )
        objects: list[KnowledgeObject] = []
        indexed = 0
        skipped: list[str] = []

        for paper_id in wanted:
            stored = await self._knowledge.get(paper_id)
            if stored is None or not stored.is_trusted:
                skipped.append(paper_id)
                continue
            objects.extend(stored.value.objects)
            indexed += 1

        return objects, indexed, tuple(skipped)

    def _metadata(
        self, obj: KnowledgeObject, identity: ModelIdentity, version: str
    ) -> VectorMetadata:
        location = obj.primary_location
        return VectorMetadata(
            knowledge_object_id=obj.id,
            paper_id=obj.paper_id,
            kind=obj.kind,
            name=obj.name[:200],
            evidence_ids=tuple(item.id for item in obj.evidence),
            section_title=location.section_title,
            page=location.page,
            confidence=obj.confidence.score,
            model_identity=identity,
            index_version=version,
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
