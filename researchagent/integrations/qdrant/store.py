"""Qdrant vector store adapter.

The only module that imports the Qdrant client. Everything above depends on
:class:`VectorStore`, so the in-memory store and this adapter are interchangeable.

Collection naming embeds the index version, and the stored identity is checked on every
``ensure_collection``: a changed embedding model produces a new collection rather than
contaminating the old one.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from researchagent.core.exceptions import IndexIncompatibleError, VectorStoreError
from researchagent.core.interfaces.embeddings import ModelIdentity
from researchagent.core.interfaces.vector_store import (
    VectorFilter,
    VectorHit,
    VectorMetadata,
    VectorRecord,
    VectorStore,
)
from researchagent.core.logging import get_logger

logger = get_logger(__name__)

_IDENTITY_KEY = "__index_identity__"


class QdrantVectorStore(VectorStore):
    name: ClassVar[str] = "qdrant"

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_prefix: str = "researchagent_knowledge",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._url = url
        self._prefix = collection_prefix
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._collection: str | None = None
        self._client: Any | None = None

    def _connect(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise VectorStoreError("qdrant-client is not installed") from exc
            self._client = AsyncQdrantClient(
                url=self._url, api_key=self._api_key, timeout=int(self._timeout)
            )
        return self._client

    async def ensure_collection(self, identity: ModelIdentity, index_version: str) -> None:
        from qdrant_client import models as qmodels

        client = self._connect()
        # The version is part of the name: incompatible vectors cannot share a collection
        # even by accident.
        self._collection = f"{self._prefix}_v{index_version}"

        try:
            exists = await client.collection_exists(self._collection)
            if not exists:
                await client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=identity.dimension, distance=qmodels.Distance.COSINE
                    ),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self._collection,
                    dimension=identity.dimension,
                    model=identity.fingerprint,
                )
                return

            info = await client.get_collection(self._collection)
            stored_size = info.config.params.vectors.size
            if stored_size != identity.dimension:
                raise IndexIncompatibleError(
                    "Existing collection has a different vector dimension",
                    collection=self._collection,
                    stored_dimension=stored_size,
                    requested_dimension=identity.dimension,
                )
        except IndexIncompatibleError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "Could not prepare the Qdrant collection",
                collection=self._collection,
                reason=str(exc),
            ) from exc

    async def upsert(self, records: list[VectorRecord]) -> int:
        from qdrant_client import models as qmodels

        if not records:
            return 0
        client = self._connect()
        points = [
            qmodels.PointStruct(
                # Qdrant requires UUID or integer ids; the domain id lives in the payload.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, record.id)),
                vector=list(record.vector),
                payload={**record.metadata.model_dump(mode="json"), "record_id": record.id},
            )
            for record in records
        ]
        try:
            await client.upsert(collection_name=self._require_collection(), points=points)
        except Exception as exc:
            raise VectorStoreError("Qdrant upsert failed", reason=str(exc)) from exc
        return len(points)

    async def search(
        self, vector: tuple[float, ...], *, limit: int, filters: VectorFilter | None = None
    ) -> list[VectorHit]:
        client = self._connect()
        try:
            response = await client.query_points(
                collection_name=self._require_collection(),
                query=list(vector),
                limit=limit,
                query_filter=_to_qdrant_filter(filters),
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant search failed", reason=str(exc)) from exc

        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            record_id = str(payload.pop("record_id", point.id))
            hits.append(
                VectorHit(
                    id=record_id,
                    score=float(point.score),
                    metadata=VectorMetadata.model_validate(payload),
                )
            )
        return hits

    async def count(self) -> int:
        client = self._connect()
        try:
            result = await client.count(self._require_collection())
        except Exception as exc:
            raise VectorStoreError("Qdrant count failed", reason=str(exc)) from exc
        return int(result.count)

    async def delete_collection(self) -> bool:
        client = self._connect()
        try:
            return bool(await client.delete_collection(self._require_collection()))
        except Exception as exc:
            raise VectorStoreError("Qdrant delete failed", reason=str(exc)) from exc

    async def health(self) -> bool:
        """Must not raise: an unreachable Qdrant degrades retrieval, it does not break it."""
        try:
            client = self._connect()
            await client.get_collections()
        except Exception as exc:  # noqa: BLE001 - probe
            logger.warning("qdrant_unhealthy", url=self._url, reason=str(exc))
            return False
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_collection(self) -> str:
        if self._collection is None:
            raise VectorStoreError("ensure_collection has not been called")
        return self._collection


def _to_qdrant_filter(filters: VectorFilter | None) -> Any:
    from qdrant_client import models as qmodels

    if filters is None:
        return None

    conditions: list[Any] = []
    if filters.paper_ids:
        conditions.append(
            qmodels.FieldCondition(
                key="paper_id", match=qmodels.MatchAny(any=list(filters.paper_ids))
            )
        )
    if filters.kinds:
        conditions.append(
            qmodels.FieldCondition(
                key="kind", match=qmodels.MatchAny(any=[kind.value for kind in filters.kinds])
            )
        )
    if filters.min_confidence > 0:
        conditions.append(
            qmodels.FieldCondition(
                key="confidence", range=qmodels.Range(gte=filters.min_confidence)
            )
        )
    return qmodels.Filter(must=conditions) if conditions else None
