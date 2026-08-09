"""Hybrid retrieval by composition.

A hybrid retriever is itself a :class:`KnowledgeRetriever`, built from other retrievers.
That is what lets v0.8 add a graph traversal retriever to the mix by naming it in
configuration, and what lets the benchmark run each component in isolation against the
composition.

Two fusion strategies, both configurable, neither hardcoded:

* **Reciprocal rank fusion** — combines ranks, not scores. Immune to the fact that a
  BM25 score and a cosine similarity are not on the same scale, which is the usual way
  weighted fusion goes quietly wrong.
* **Weighted score fusion** — combines normalised scores with explicit weights. More
  expressive when the components are calibrated, and directly comparable to RRF in the
  benchmark.

A degraded component is excluded rather than counted as zero: an unreachable vector store
must not push every semantically-strong result down the ranking.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import ClassVar

from researchagent.config.schemas import FusionSettings, FusionStrategy
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

logger = get_logger(__name__)


class ComponentRole(StrEnum):
    LEXICAL = "lexical"
    SPARSE = "sparse"
    DENSE = "dense"


class RetrieverComponent:
    """One named, weighted input to fusion."""

    def __init__(self, retriever: KnowledgeRetriever, role: ComponentRole, weight: float) -> None:
        self.retriever = retriever
        self.role = role
        self.weight = weight


class HybridKnowledgeRetriever(KnowledgeRetriever):
    """Layer 1, composed."""

    name: ClassVar[str] = "hybrid_knowledge_retriever"

    def __init__(
        self, components: list[RetrieverComponent], settings: FusionSettings | None = None
    ) -> None:
        if not components:
            raise ValueError("hybrid retrieval requires at least one component")
        self._components = components
        self._settings = settings or FusionSettings()

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        started = time.perf_counter()
        # Each component sees a wider window than the caller asked for: fusion needs depth
        # to reorder, and truncating first would discard the disagreement it exists to use.
        widened = query.model_copy(
            update={"limit": min(query.limit * self._settings.candidate_multiplier, 500)}
        )

        results = [
            (component, await component.retriever.retrieve(widened))
            for component in self._components
        ]
        usable = [(component, result) for component, result in results if result.is_usable]
        degraded = [result for _, result in results if not result.is_usable]

        for result in degraded:
            logger.warning(
                "fusion_component_degraded",
                retriever=result.retrieved_by,
                reason=result.unavailable_reason,
            )

        if not usable:
            return RetrievalResult[KnowledgeObject].unavailable(
                layer=RetrievalLayer.KNOWLEDGE,
                query=query,
                retrieved_by=self.name,
                reason="every fusion component is unavailable",
            )

        fused = (
            self._reciprocal_rank(usable)
            if self._settings.strategy is FusionStrategy.RECIPROCAL_RANK
            else self._weighted_score(usable)
        )
        fused.sort(key=lambda entry: (-entry[1], entry[0].id))

        hits = tuple(
            RetrievalHit[KnowledgeObject](
                item=obj,
                score=round(min(score, 1.0), 6),
                signals=tuple(signals),
                retrieved_by=self.name,
            )
            for obj, score, signals in fused[: query.limit]
        )

        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.KNOWLEDGE,
            query=query,
            hits=hits,
            considered=max((result.considered for _, result in usable), default=0),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            # Fusion that lost a component is a real result, but a diminished one, and the
            # caller is told so rather than left to infer it.
            degraded=bool(degraded),
            unavailable_reason=(
                "; ".join(f"{r.retrieved_by}: {r.unavailable_reason}" for r in degraded) or None
            ),
        )

    async def health(self) -> bool:
        """Healthy while any component can serve."""
        for component in self._components:
            if await component.retriever.health():
                return True
        return False

    def _reciprocal_rank(
        self, results: list[tuple[RetrieverComponent, RetrievalResult[KnowledgeObject]]]
    ) -> list[tuple[KnowledgeObject, float, list[ConfidenceSignal]]]:
        k = self._settings.rrf_k
        accumulated: dict[str, tuple[KnowledgeObject, float, list[ConfidenceSignal]]] = {}
        total_weight = sum(component.weight for component, _ in results) or 1.0

        for component, result in results:
            for rank, hit in enumerate(result.hits, start=1):
                contribution = component.weight / (k + rank)
                obj, score, signals = accumulated.get(hit.item.id, (hit.item, 0.0, []))
                signals.append(
                    ConfidenceSignal(
                        name=f"{component.role.value}_rank",
                        value=round(1.0 / rank, 6),
                        weight=component.weight,
                        observation=(
                            f"ranked #{rank} by {result.retrieved_by} (score {hit.score:.3f})"
                        ),
                    )
                )
                accumulated[hit.item.id] = (obj, score + contribution, signals)

        # Normalise into [0, 1] against the best achievable RRF score for this k.
        ceiling = total_weight / (k + 1)
        return [
            (obj, score / ceiling if ceiling else 0.0, signals)
            for obj, score, signals in accumulated.values()
        ]

    def _weighted_score(
        self, results: list[tuple[RetrieverComponent, RetrievalResult[KnowledgeObject]]]
    ) -> list[tuple[KnowledgeObject, float, list[ConfidenceSignal]]]:
        accumulated: dict[str, tuple[KnowledgeObject, float, list[ConfidenceSignal]]] = {}
        total_weight = sum(component.weight for component, _ in results) or 1.0

        for component, result in results:
            # Min-max within the component so an absolute scale mismatch between BM25 and
            # cosine cannot decide the ranking.
            scores = [hit.score for hit in result.hits]
            lowest, highest = (min(scores), max(scores)) if scores else (0.0, 0.0)
            span = highest - lowest

            for hit in result.hits:
                normalised = (hit.score - lowest) / span if span > 0 else (1.0 if scores else 0.0)
                obj, score, signals = accumulated.get(hit.item.id, (hit.item, 0.0, []))
                signals.append(
                    ConfidenceSignal(
                        name=f"{component.role.value}_score",
                        value=round(normalised, 6),
                        weight=component.weight,
                        observation=(
                            f"{result.retrieved_by} scored {hit.score:.3f}, "
                            f"normalised to {normalised:.3f} within its own result set"
                        ),
                    )
                )
                accumulated[hit.item.id] = (
                    obj,
                    score + component.weight * normalised,
                    signals,
                )

        return [
            (obj, score / total_weight, signals) for obj, score, signals in accumulated.values()
        ]
