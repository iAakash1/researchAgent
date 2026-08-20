"""Deterministic lexical retrieval — the v0.6 baseline.

Term overlap and field weighting over stored knowledge objects. No index to build, no
model to call, and the same query always returns the same ranking, which is what makes it
the control arm every other retriever is measured against.
"""

from __future__ import annotations

import time
from typing import ClassVar

from researchagent.config.schemas import RetrievalWeights
from researchagent.core.interfaces.repositories import KnowledgeRepository
from researchagent.core.interfaces.retrieval import (
    KnowledgeRetriever,
    RetrievalHit,
    RetrievalLayer,
    RetrievalResult,
)
from researchagent.core.logging import get_logger
from researchagent.core.validation import ConfidenceSignal, weighted_score
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery
from researchagent.utils.text import overlap, tokenise

logger = get_logger(__name__)


class LexicalKnowledgeRetriever(KnowledgeRetriever):
    """Layer 1 — structured facts matching a query."""

    name: ClassVar[str] = "lexical_knowledge_retriever"

    def __init__(
        self, knowledge: KnowledgeRepository, weights: RetrievalWeights | None = None
    ) -> None:
        self._knowledge = knowledge
        self._weights = weights or RetrievalWeights()

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        started = time.perf_counter()
        terms = tokenise(" ".join(query.search_terms()))
        candidates = await self._candidates(query)

        hits = []
        for candidate in candidates:
            signals = self._signals(candidate, terms)
            score = weighted_score(signals)
            if score <= 0.0:
                continue
            hits.append(
                RetrievalHit[KnowledgeObject](
                    item=candidate, score=score, signals=tuple(signals), retrieved_by=self.name
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.item.id))
        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.KNOWLEDGE,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=len(candidates),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def health(self) -> bool:
        return bool(await self._knowledge.list_ids())

    async def _candidates(self, query: ResearchQuery) -> list[KnowledgeObject]:
        """Every knowledge object passing the query's structural filters."""
        paper_ids = query.paper_ids or tuple(
            key.replace("-", ":", 1) for key in await self._knowledge.list_ids()
        )
        candidates: list[KnowledgeObject] = []
        for paper_id in paper_ids:
            stored = await self._knowledge.get(paper_id)
            if stored is None or not stored.is_trusted:
                # Zero trust across stages: knowledge the previous stage rejected is not
                # retrievable, no matter how well it matches.
                continue
            candidates.extend(
                item
                for item in stored.value.objects
                if query.matches_kind(item.kind)
                and query.matches_paper(item.paper_id)
                and item.confidence.score >= query.min_confidence
            )
        return candidates

    def _signals(self, item: KnowledgeObject, terms: set[str]) -> list[ConfidenceSignal]:
        name_match = overlap(tokenise(item.name), terms)
        text_match = overlap(tokenise(f"{item.description} {' '.join(item.quotes)}"), terms)

        return [
            ConfidenceSignal(
                name="name_match",
                value=name_match,
                weight=self._weights.name_match,
                observation=f"{name_match:.2f} of query terms appear in {item.name!r}",
            ),
            ConfidenceSignal(
                name="text_match",
                value=text_match,
                weight=self._weights.text_match,
                observation=f"{text_match:.2f} of query terms appear in its description or quote",
            ),
            ConfidenceSignal(
                name="validation_confidence",
                value=item.confidence.score,
                weight=self._weights.validation_confidence,
                observation=f"the object was validated at confidence {item.confidence.score:.2f}",
            ),
            ConfidenceSignal(
                name="evidence_density",
                value=min(len(item.evidence) / 3, 1.0),
                weight=self._weights.evidence_density,
                observation=f"{len(item.evidence)} evidence items support it",
            ),
        ]
