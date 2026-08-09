"""Semantic retrieval over embedded knowledge objects.

The canonical semantic unit is a :class:`KnowledgeObject` with its provenance attached —
not a chunk of PDF text. Chunking discards the structure the previous four releases were
spent building, and a chunk cannot tell you which validated fact it supports. A vector
hit here resolves to a knowledge object, which resolves to its evidence, which resolves
to a page.

Degrades rather than fails. If the embedding backend or the vector store is unreachable,
this returns a result marked ``degraded`` so hybrid fusion falls back to lexical instead
of mistaking an outage for an empty corpus.
"""

from __future__ import annotations

import time
from typing import ClassVar

from researchagent.core.exceptions import EmbeddingError, VectorStoreError
from researchagent.core.interfaces.embeddings import EmbeddingModel
from researchagent.core.interfaces.knowledge_repository import KnowledgeRepository
from researchagent.core.interfaces.retrieval import (
    KnowledgeRetriever,
    RetrievalHit,
    RetrievalLayer,
    RetrievalResult,
)
from researchagent.core.interfaces.vector_store import VectorFilter, VectorStore
from researchagent.core.logging import get_logger
from researchagent.core.validation import ConfidenceSignal
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery

logger = get_logger(__name__)


class SemanticKnowledgeRetriever(KnowledgeRetriever):
    """Layer 1, dense."""

    name: ClassVar[str] = "semantic_knowledge_retriever"

    def __init__(
        self,
        embeddings: EmbeddingModel,
        store: VectorStore,
        knowledge: KnowledgeRepository,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._knowledge = knowledge

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        started = time.perf_counter()

        try:
            vector = await self._embeddings.embed_text(query.text)
            raw_hits = await self._store.search(
                vector,
                limit=query.limit * 2,
                filters=VectorFilter(
                    paper_ids=query.paper_ids,
                    kinds=query.kinds,
                    min_confidence=query.min_confidence,
                ),
            )
        except (EmbeddingError, VectorStoreError) as exc:
            logger.warning(
                "semantic_retrieval_unavailable", error_code=exc.code, reason=exc.message
            )
            return RetrievalResult[KnowledgeObject].unavailable(
                layer=RetrievalLayer.KNOWLEDGE,
                query=query,
                retrieved_by=self.name,
                reason=f"{exc.code}: {exc.message}",
            )

        hits = []
        for raw in raw_hits:
            # The vector store is not the source of truth: the object is loaded from the
            # knowledge repository, and a hit that cannot be resolved is dropped.
            obj = await self._resolve(raw.metadata.knowledge_object_id, raw.metadata.paper_id)
            if obj is None:
                logger.debug("vector_hit_unresolved", object_id=raw.metadata.knowledge_object_id)
                continue
            # Cosine runs [-1, 1]; map to [0, 1] so it composes with bounded signals.
            similarity = max(0.0, min(1.0, (raw.score + 1.0) / 2.0))
            hits.append(
                RetrievalHit[KnowledgeObject](
                    item=obj,
                    score=round(similarity, 6),
                    signals=(
                        ConfidenceSignal(
                            name="semantic_similarity",
                            value=round(similarity, 6),
                            observation=(
                                f"cosine {raw.score:.3f} against "
                                f"{raw.metadata.model_identity.fingerprint}"
                            ),
                        ),
                    ),
                    retrieved_by=self.name,
                )
            )

        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.KNOWLEDGE,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=len(raw_hits),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def health(self) -> bool:
        embedding_health = await self._embeddings.health()
        return embedding_health.healthy and await self._store.health()

    async def _resolve(self, object_id: str, paper_id: str) -> KnowledgeObject | None:
        stored = await self._knowledge.get(paper_id)
        if stored is None or not stored.is_trusted:
            return None
        return stored.value.by_id(object_id)
