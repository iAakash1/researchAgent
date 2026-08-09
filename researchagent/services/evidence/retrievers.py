"""Deterministic retrieval implementations for the four layers.

Lexical and structural only — no embeddings, no vector store, no reranker. That is a
scope decision, not a shortcut: the point of this release is that the *contracts* are
right, so v0.7 can add hybrid retrieval as another implementation of these same ports
without a caller noticing.

Every score is assembled from named signals carrying their observation, so a ranking can
be explained, diffed against a future embedding ranker, and benchmarked against it.
"""

from __future__ import annotations

import time
from typing import ClassVar

from researchagent.config.schemas import RetrievalWeights
from researchagent.core.interfaces.bundle_repository import BundleRepository
from researchagent.core.interfaces.document_repository import DocumentRepository
from researchagent.core.interfaces.evidence_repository import EvidenceRepository
from researchagent.core.interfaces.knowledge_repository import KnowledgeRepository
from researchagent.core.interfaces.retrieval import (
    BundleRetriever,
    CrossPaperRetriever,
    DocumentRetriever,
    EvidenceRetriever,
    KnowledgeRetriever,
    RetrievalHit,
    RetrievalLayer,
    RetrievalResult,
)
from researchagent.core.logging import get_logger
from researchagent.core.validation import ConfidenceSignal
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.document import PaperDocument
from researchagent.models.evidence import EvidenceRecord
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery
from researchagent.utils.text import normalise

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our", "that",
        "the", "their", "there", "these", "this", "those", "to", "via", "we", "what",
        "which", "why", "with",
    }
)  # fmt: skip


def tokenise(text: str) -> set[str]:
    return {
        token for token in normalise(text).split() if len(token) > 2 and token not in _STOPWORDS
    }


def overlap(candidate: set[str], reference: set[str]) -> float:
    """Fraction of the query vocabulary present in the candidate.

    Asymmetric on purpose: a long description should not be penalised for containing
    words the query never mentioned.
    """
    if not candidate or not reference:
        return 0.0
    return len(candidate & reference) / len(reference)


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
            score = _weighted(signals)
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


class LinkedEvidenceRetriever(EvidenceRetriever):
    """Layer 2 — the quotes supporting a set of facts, or matching a query."""

    name: ClassVar[str] = "linked_evidence_retriever"

    def __init__(
        self, evidence: EvidenceRepository, weights: RetrievalWeights | None = None
    ) -> None:
        self._evidence = evidence
        self._weights = weights or RetrievalWeights()

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[EvidenceRecord]:
        started = time.perf_counter()
        terms = tokenise(" ".join(query.search_terms()))
        records = await self._evidence.search(
            tuple(terms), paper_ids=query.paper_ids, limit=query.limit * 4
        )

        hits = []
        for record in records:
            match = overlap(tokenise(record.quote), terms)
            precision = min(record.location.precision / 4, 1.0)
            signals = [
                ConfidenceSignal(
                    name="quote_match",
                    value=match,
                    weight=self._weights.text_match,
                    observation=f"{match:.2f} of query terms appear in the quoted sentence",
                ),
                ConfidenceSignal(
                    name="location_precision",
                    value=precision,
                    weight=self._weights.provenance_precision,
                    observation=f"evidence resolves to {record.location.describe()}",
                ),
            ]
            score = _weighted(signals)
            if score <= 0.0:
                continue
            hits.append(
                RetrievalHit[EvidenceRecord](
                    item=record, score=score, signals=tuple(signals), retrieved_by=self.name
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.item.id))
        return RetrievalResult[EvidenceRecord](
            layer=RetrievalLayer.EVIDENCE,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=len(records),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def for_objects(self, object_ids: tuple[str, ...]) -> tuple[EvidenceRecord, ...]:
        return await self._evidence.for_objects(object_ids)

    async def health(self) -> bool:
        return True


class RepositoryDocumentRetriever(DocumentRetriever):
    """Layer 3 — the canonical documents behind the quotes."""

    name: ClassVar[str] = "repository_document_retriever"

    def __init__(self, documents: DocumentRepository) -> None:
        self._documents = documents

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[PaperDocument]:
        started = time.perf_counter()
        terms = tokenise(" ".join(query.search_terms()))

        paper_ids = query.paper_ids or tuple(
            key.replace("-", ":", 1) for key in await self._documents.list_ids()
        )
        hits = []
        for paper_id in paper_ids:
            document = await self.by_paper_id(paper_id)
            if document is None:
                continue
            match = overlap(tokenise(document.metadata.title or ""), terms)
            signals = [
                ConfidenceSignal(
                    name="title_match",
                    value=match,
                    observation=f"{match:.2f} of query terms appear in the document title",
                )
            ]
            hits.append(
                RetrievalHit[PaperDocument](
                    item=document,
                    score=max(match, 0.01),  # a requested document is always returned
                    signals=tuple(signals),
                    retrieved_by=self.name,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.item.paper_id))
        return RetrievalResult[PaperDocument](
            layer=RetrievalLayer.DOCUMENT,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=len(paper_ids),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def by_paper_id(self, paper_id: str) -> PaperDocument | None:
        stored = await self._documents.get(paper_id)
        return stored.value if stored is not None and stored.is_trusted else None

    async def health(self) -> bool:
        return True


class AgreementCrossPaperRetriever(CrossPaperRetriever):
    """Layer 4 — the same entity as several papers describe it.

    Independent agreement is evidence in its own right: an entity three papers name is a
    stronger finding than one paper naming it three times, and a single-paper claim is
    visibly single-paper rather than silently blended into the rest.
    """

    name: ClassVar[str] = "agreement_cross_paper_retriever"

    def __init__(
        self, knowledge: KnowledgeRetriever, weights: RetrievalWeights | None = None
    ) -> None:
        self._knowledge = knowledge
        self._weights = weights or RetrievalWeights()

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        started = time.perf_counter()
        # Ask layer 1 broadly, then re-rank by how many distinct papers agree.
        underlying = await self._knowledge.retrieve(
            query.model_copy(update={"limit": min(query.limit * 5, 500)})
        )

        papers_by_name: dict[str, set[str]] = {}
        for hit in underlying.hits:
            papers_by_name.setdefault(normalise(hit.item.name), set()).add(hit.item.paper_id)

        hits = []
        for hit in underlying.hits:
            papers = papers_by_name[normalise(hit.item.name)]
            agreement = min(len(papers) / 3, 1.0)
            signals = (
                *hit.signals,
                ConfidenceSignal(
                    name="cross_paper_agreement",
                    value=agreement,
                    weight=self._weights.cross_paper_agreement,
                    observation=(
                        f"{len(papers)} distinct paper(s) describe {hit.item.name!r}: "
                        f"{sorted(papers)}"
                    ),
                ),
            )
            hits.append(
                RetrievalHit[KnowledgeObject](
                    item=hit.item,
                    score=_weighted(list(signals)),
                    signals=signals,
                    retrieved_by=self.name,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.item.id))
        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.CROSS_PAPER,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=underlying.considered,
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def papers_mentioning(self, name: str) -> tuple[str, ...]:
        result = await self._knowledge.retrieve(ResearchQuery(text=name, limit=200))
        needle = normalise(name)
        return tuple(
            sorted(
                {
                    hit.item.paper_id
                    for hit in result.hits
                    if needle in normalise(hit.item.name) or normalise(hit.item.name) in needle
                }
            )
        )

    async def health(self) -> bool:
        return await self._knowledge.health()


class StoredBundleRetriever(BundleRetriever):
    """Retrieves previously assembled bundles rather than rebuilding them."""

    name: ClassVar[str] = "stored_bundle_retriever"

    def __init__(self, bundles: BundleRepository) -> None:
        self._bundles = bundles

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[EvidenceBundle]:
        started = time.perf_counter()
        terms = tokenise(" ".join(query.search_terms()))

        candidates: list[EvidenceBundle] = []
        if query.question_id:
            candidates.extend(await self._bundles.for_question(query.question_id))
        else:
            for bundle_id in await self._bundles.list_ids():
                found = await self._bundles.get(bundle_id)
                if found is not None:
                    candidates.append(found)

        hits = []
        for bundle in candidates:
            match = overlap(tokenise(bundle.query.text), terms)
            signals = (
                ConfidenceSignal(
                    name="query_match",
                    value=match,
                    observation=f"{match:.2f} term overlap with the bundle's own query",
                ),
                ConfidenceSignal(
                    name="bundle_confidence",
                    value=bundle.confidence.score,
                    observation=f"bundle validated at confidence {bundle.confidence.score:.2f}",
                ),
            )
            hits.append(
                RetrievalHit[EvidenceBundle](
                    item=bundle,
                    score=_weighted(list(signals)),
                    signals=signals,
                    retrieved_by=self.name,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.item.id))
        return RetrievalResult[EvidenceBundle](
            layer=RetrievalLayer.BUNDLE,
            query=query,
            hits=tuple(hits[: query.limit]),
            considered=len(candidates),
            retrieved_by=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def health(self) -> bool:
        return True


def _weighted(signals: list[ConfidenceSignal]) -> float:
    """Weighted mean of signal values, clamped into [0, 1]."""
    total = sum(signal.weight for signal in signals)
    if total <= 0:
        return 0.0
    return round(min(sum(s.value * s.weight for s in signals) / total, 1.0), 6)
