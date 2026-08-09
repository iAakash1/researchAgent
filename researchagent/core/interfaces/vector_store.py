"""Vector store port.

No business logic depends on Qdrant. This port is the whole surface: an in-memory
implementation serves tests and offline development, and the Qdrant adapter serves
production, with nothing above the port able to tell which is running.

The store is never the source of truth. Every vector carries enough metadata to recover
the knowledge object it represents, but the object itself lives in the knowledge
repository — a wiped vector store costs an index rebuild, never a fact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.interfaces.embeddings import ModelIdentity
from researchagent.models.knowledge import KnowledgeKind


class VectorMetadata(BaseModel):
    """Everything needed to recover a knowledge object and judge its vector.

    Denormalised deliberately: retrieval must be able to filter and explain without a
    second round trip, and the payload is what makes a vector hit interpretable rather
    than an opaque id and a number.
    """

    model_config = {"frozen": True}

    knowledge_object_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    kind: KnowledgeKind
    name: str = ""
    evidence_ids: tuple[str, ...] = ()
    section_title: str | None = None
    page: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Provenance of the vector itself.
    model_identity: ModelIdentity
    index_version: str = Field(min_length=1)
    embedded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VectorRecord(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    vector: tuple[float, ...]
    metadata: VectorMetadata


class VectorFilter(BaseModel):
    """Structural constraints applied before similarity, not after."""

    model_config = {"frozen": True}

    paper_ids: tuple[str, ...] = ()
    kinds: tuple[KnowledgeKind, ...] = ()
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VectorHit(BaseModel):
    model_config = {"frozen": True}

    id: str
    score: float = Field(description="Cosine similarity, higher is closer")
    metadata: VectorMetadata


class VectorStore(ABC):
    """Stores and searches embedded knowledge."""

    name: ClassVar[str]

    @abstractmethod
    async def ensure_collection(self, identity: ModelIdentity, index_version: str) -> None:
        """Create the collection if absent.

        Implementations must refuse to reuse a collection whose stored identity differs
        from ``identity`` — mixing vector spaces is the failure mode this port exists to
        prevent.
        """

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> int: ...

    @abstractmethod
    async def search(
        self, vector: tuple[float, ...], *, limit: int, filters: VectorFilter | None = None
    ) -> list[VectorHit]: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def delete_collection(self) -> bool: ...

    @abstractmethod
    async def health(self) -> bool:
        """Cheap probe; must not raise."""

    @abstractmethod
    async def aclose(self) -> None: ...
