"""BM25 retrieval over the canonical retrieval representation.

Implemented directly rather than pulled in as a dependency: the algorithm is twenty
lines, the corpus lives in memory anyway, and an external library would need the same
representation plumbing while hiding the scoring we want to explain.

Operates on the same text semantic retrieval embeds, so a BM25 result and a semantic
result for one query are genuinely comparable rather than differing because they saw
different documents.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import ClassVar

from researchagent.config.schemas import BM25Settings
from researchagent.core.interfaces.repositories import KnowledgeRepository
from researchagent.core.interfaces.retrieval import (
    KnowledgeRetriever,
    RetrievalHit,
    RetrievalLayer,
    RetrievalResult,
)
from researchagent.core.logging import get_logger
from researchagent.core.validation import ConfidenceSignal
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery
from researchagent.services.retrieval.representation import represent
from researchagent.utils.text import normalise

logger = get_logger(__name__)


def tokenise(text: str) -> list[str]:
    return [token for token in normalise(text).split() if len(token) > 1]


class BM25Index:
    """A BM25 index over knowledge objects. Rebuilt from the repository, never persisted."""

    def __init__(self, settings: BM25Settings | None = None) -> None:
        self._settings = settings or BM25Settings()
        self._documents: list[tuple[KnowledgeObject, list[str]]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def build(self, objects: list[KnowledgeObject]) -> None:
        self._documents = [(obj, tokenise(represent(obj).text)) for obj in objects]
        self._document_frequency = Counter()
        for _, tokens in self._documents:
            self._document_frequency.update(set(tokens))
        lengths = [len(tokens) for _, tokens in self._documents]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0

    @property
    def size(self) -> int:
        return len(self._documents)

    def score(self, query_tokens: list[str]) -> list[tuple[KnowledgeObject, float]]:
        if not self._documents or not query_tokens:
            return []

        k1, b = self._settings.k1, self._settings.b
        total = len(self._documents)
        scored: list[tuple[KnowledgeObject, float]] = []

        for obj, tokens in self._documents:
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[token]
                # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.
                idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + k1 * (
                    1 - b + b * (length / self._average_length if self._average_length else 1)
                )
                score += idf * (frequency * (k1 + 1)) / denominator
            if score > 0:
                scored.append((obj, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored


class BM25KnowledgeRetriever(KnowledgeRetriever):
    """Layer 1, lexical-statistical."""

    name: ClassVar[str] = "bm25_knowledge_retriever"

    def __init__(
        self, knowledge: KnowledgeRepository, settings: BM25Settings | None = None
    ) -> None:
        self._knowledge = knowledge
        self._settings = settings or BM25Settings()
        self._index = BM25Index(self._settings)

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        started = time.perf_counter()
        candidates = await self._candidates(query)
        # Rebuilt per query: the corpus is small, and a stale index is a subtler bug than
        # a slow one. v0.8 can cache this behind the same interface.
        self._index.build(candidates)

        scored = self._index.score(tokenise(" ".join(query.search_terms())))
        best = scored[0][1] if scored else 0.0

        hits = []
        for obj, raw in scored[: query.limit]:
            # BM25 is unbounded; normalise against the best score so it can be fused with
            # bounded signals. Rank order is unaffected.
            normalised = raw / best if best > 0 else 0.0
            hits.append(
                RetrievalHit[KnowledgeObject](
                    item=obj,
                    score=round(min(normalised, 1.0), 6),
                    signals=(
                        ConfidenceSignal(
                            name="bm25",
                            value=round(min(normalised, 1.0), 6),
                            observation=(
                                f"BM25 score {raw:.3f} (best in this result set {best:.3f})"
                            ),
                        ),
                    ),
                    retrieved_by=self.name,
                )
            )

        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.KNOWLEDGE,
            query=query,
            hits=tuple(hits),
            considered=len(candidates),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def health(self) -> bool:
        return True

    async def _candidates(self, query: ResearchQuery) -> list[KnowledgeObject]:
        paper_ids = query.paper_ids or tuple(
            key.replace("-", ":", 1) for key in await self._knowledge.list_ids()
        )
        candidates: list[KnowledgeObject] = []
        for paper_id in paper_ids:
            stored = await self._knowledge.get(paper_id)
            if stored is None or not stored.is_trusted:
                continue
            candidates.extend(
                item
                for item in stored.value.objects
                if query.matches_kind(item.kind)
                and query.matches_paper(item.paper_id)
                and item.confidence.score >= query.min_confidence
            )
        return candidates
