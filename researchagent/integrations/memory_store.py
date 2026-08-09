"""In-memory vector store.

Not a stub. It implements the full port with exact cosine search, so the entire semantic
and hybrid retrieval stack is exercised in tests and offline development without Qdrant
running. Brute force is the right algorithm at this corpus size and makes the Qdrant
adapter's results checkable against an exact baseline.

Volatile by design: the vector store is never the source of truth, so losing it costs an
index rebuild and nothing else.
"""

from __future__ import annotations

import math
from typing import ClassVar

from researchagent.core.exceptions import IndexIncompatibleError
from researchagent.core.interfaces.embeddings import ModelIdentity
from researchagent.core.interfaces.vector_store import (
    VectorFilter,
    VectorHit,
    VectorRecord,
    VectorStore,
)
from researchagent.core.logging import get_logger

logger = get_logger(__name__)


class InMemoryVectorStore(VectorStore):
    name: ClassVar[str] = "memory"

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._identity: ModelIdentity | None = None
        self._index_version: str | None = None

    async def ensure_collection(self, identity: ModelIdentity, index_version: str) -> None:
        if self._identity is not None and not self._identity.is_compatible_with(identity):
            raise IndexIncompatibleError(
                "Stored vectors were produced by a different embedding model",
                stored=self._identity.fingerprint,
                requested=identity.fingerprint,
            )
        self._identity = identity
        self._index_version = index_version

    async def upsert(self, records: list[VectorRecord]) -> int:
        for record in records:
            if self._identity is not None and not record.metadata.model_identity.is_compatible_with(
                self._identity
            ):
                raise IndexIncompatibleError(
                    "Refusing to mix vector spaces in one collection",
                    stored=self._identity.fingerprint,
                    incoming=record.metadata.model_identity.fingerprint,
                )
            self._records[record.id] = record
        return len(records)

    async def search(
        self, vector: tuple[float, ...], *, limit: int, filters: VectorFilter | None = None
    ) -> list[VectorHit]:
        scored = []
        for record in self._records.values():
            if not _passes(record, filters):
                continue
            similarity = _cosine(vector, record.vector)
            scored.append(VectorHit(id=record.id, score=similarity, metadata=record.metadata))

        scored.sort(key=lambda hit: (-hit.score, hit.id))
        return scored[:limit]

    async def count(self) -> int:
        return len(self._records)

    async def delete_collection(self) -> bool:
        existed = bool(self._records) or self._identity is not None
        self._records.clear()
        self._identity = None
        return existed

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def _passes(record: VectorRecord, filters: VectorFilter | None) -> bool:
    if filters is None:
        return True
    metadata = record.metadata
    if filters.paper_ids and metadata.paper_id not in filters.paper_ids:
        return False
    if filters.kinds and metadata.kind not in filters.kinds:
        return False
    return not metadata.confidence < filters.min_confidence


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    # Clamp: floating point can push a unit-vector dot product a hair past 1.0.
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
