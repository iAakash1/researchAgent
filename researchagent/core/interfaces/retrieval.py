"""Retrieval ports.

Four layers, one interface. A caller asks the same way at every layer and gets back the
same envelope; only the item type differs.

    Layer 1  knowledge   — structured facts
    Layer 2  evidence    — the quotes supporting them
    Layer 3  document    — the canonical documents behind those quotes
    Layer 4  cross-paper — the same entity as several papers describe it

Nothing here names a technology. v0.6 ships deterministic lexical implementations; v0.7
adds embeddings, BM25 and a reranker, and v0.8 adds a graph traversal retriever. Each
arrives as a new implementation of these ports, composed by a fusion retriever that is
itself one of these ports. No caller changes, because a caller never had a way to depend
on how retrieval worked.

Scores carry their signals for the same reason confidence does: a ranking that cannot
explain itself cannot be debugged, tuned, or defended.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.document import PaperDocument
from researchagent.models.evidence import EvidenceRecord
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery


class RetrievalLayer(StrEnum):
    KNOWLEDGE = "knowledge"
    EVIDENCE = "evidence"
    DOCUMENT = "document"
    CROSS_PAPER = "cross_paper"
    BUNDLE = "bundle"


class RetrievalHit[TItem: BaseModel](BaseModel):
    """One retrieved item, its score, and why it scored that way."""

    model_config = {"frozen": True}

    item: TItem
    score: float = Field(ge=0.0, le=1.0)
    signals: tuple[ConfidenceSignal, ...] = ()
    retrieved_by: str = Field(min_length=1)

    def explain(self) -> str:
        return Confidence(score=self.score, signals=self.signals).explain()


class RetrievalResult[TItem: BaseModel](BaseModel):
    """A page of hits, with what it cost and what it looked at."""

    model_config = {"frozen": True}

    layer: RetrievalLayer
    query: ResearchQuery
    hits: tuple[RetrievalHit[TItem], ...] = ()
    # How many candidates were examined. The denominator that makes a small result set
    # interpretable: two hits from three candidates is not two hits from four hundred.
    considered: int = Field(default=0, ge=0)
    retrieved_by: str = Field(min_length=1)
    latency_ms: float = Field(default=0.0, ge=0.0)

    # A retriever that could not run must be distinguishable from one that ran and found
    # nothing. Fusion depends on it: an outage in the semantic index should fall back to
    # lexical, whereas a genuinely empty semantic result is information.
    degraded: bool = False
    unavailable_reason: str | None = None

    @property
    def items(self) -> tuple[TItem, ...]:
        return tuple(hit.item for hit in self.hits)

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def is_usable(self) -> bool:
        """Whether this result reflects a retriever that actually ran."""
        return not self.degraded

    @classmethod
    def unavailable(
        cls,
        *,
        layer: RetrievalLayer,
        query: ResearchQuery,
        retrieved_by: str,
        reason: str,
    ) -> RetrievalResult[TItem]:
        """The honest empty result: nothing found because nothing could be asked."""
        return cls(
            layer=layer,
            query=query,
            retrieved_by=retrieved_by,
            degraded=True,
            unavailable_reason=reason,
        )


class Retriever[TItem: BaseModel](ABC):
    """Retrieves items of one type for a research query."""

    name: ClassVar[str]
    layer: ClassVar[RetrievalLayer]

    @abstractmethod
    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[TItem]:
        """Return ranked hits. An empty result is a valid answer, never an error."""

    @abstractmethod
    async def health(self) -> bool:
        """Whether this retriever can currently serve requests."""


# Concrete aliases so the parametrised models are built once, at import time, and appear
# by name in API schemas rather than as anonymous generics.
KnowledgeHits = RetrievalResult[KnowledgeObject]
EvidenceHits = RetrievalResult[EvidenceRecord]
DocumentHits = RetrievalResult[PaperDocument]
BundleHits = RetrievalResult[EvidenceBundle]


class KnowledgeRetriever(Retriever[KnowledgeObject], ABC):
    """Layer 1 — find structured facts."""

    layer: ClassVar[RetrievalLayer] = RetrievalLayer.KNOWLEDGE


class EvidenceRetriever(Retriever[EvidenceRecord], ABC):
    """Layer 2 — find the quotes that support them."""

    layer: ClassVar[RetrievalLayer] = RetrievalLayer.EVIDENCE

    @abstractmethod
    async def for_objects(self, object_ids: tuple[str, ...]) -> tuple[EvidenceRecord, ...]:
        """Evidence linked to specific knowledge objects.

        The provenance walk, distinct from search: the bundle builder already knows which
        facts it wants and needs their support, not a ranked guess at it.
        """


class DocumentRetriever(Retriever[PaperDocument], ABC):
    """Layer 3 — recover the canonical documents behind the quotes."""

    layer: ClassVar[RetrievalLayer] = RetrievalLayer.DOCUMENT

    @abstractmethod
    async def by_paper_id(self, paper_id: str) -> PaperDocument | None:
        """Direct lookup, for following a citation back to its source."""


class CrossPaperRetriever(Retriever[KnowledgeObject], ABC):
    """Layer 4 — the same entity as several papers describe it.

    Distinct from layer 1 because agreement across independent papers is itself evidence:
    a dataset three papers name is a different claim from one paper naming it once.
    """

    layer: ClassVar[RetrievalLayer] = RetrievalLayer.CROSS_PAPER

    @abstractmethod
    async def papers_mentioning(self, name: str) -> tuple[str, ...]:
        """Which papers describe an entity by this name."""


class BundleRetriever(Retriever[EvidenceBundle], ABC):
    """Retrieves previously assembled bundles.

    Bundles are expensive to build and stable once built; a later session asking the same
    question should find the existing bundle rather than rebuild it.
    """

    layer: ClassVar[RetrievalLayer] = RetrievalLayer.BUNDLE
