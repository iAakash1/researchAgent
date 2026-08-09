"""Ollama embedding adapter.

Local-first: no cloud API, no key. The model is named in configuration and never in
code, so switching from ``nomic-embed-text`` to ``bge-m3`` is a YAML edit — and because
the model name is part of :class:`ModelIdentity`, that edit forces a new index version
rather than silently reusing incompatible vectors.

Dimension is discovered from the model on first use rather than declared, because a
declared dimension that disagrees with the model is a bug that surfaces only as bad
retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import httpx

from researchagent.core.exceptions import EmbeddingError
from researchagent.core.interfaces.embeddings import (
    EmbeddingHealth,
    EmbeddingModel,
    ModelIdentity,
)
from researchagent.core.logging import get_logger

logger = get_logger(__name__)

_EMBED_PATH = "/api/embed"


class OllamaEmbeddingModel(EmbeddingModel):
    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        batch_size: int = 32,
        preprocessing_version: str = "1",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._preprocessing_version = preprocessing_version
        self._dimension: int | None = None
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)

    async def embed_text(self, text: str) -> tuple[float, ...]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []

        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            vectors.extend(await self._embed(chunk))
        return vectors

    async def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        try:
            response = await self._client.post(_EMBED_PATH, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                "Ollama rejected the embedding request",
                model=self._model,
                status_code=exc.response.status_code,
                reason=exc.response.text[:200],
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                "Cannot reach Ollama for embeddings",
                model=self._model,
                base_url=self._base_url,
                reason=str(exc),
            ) from exc

        body = response.json()
        raw = body.get("embeddings")
        if not raw:
            raise EmbeddingError(
                "Ollama returned no embeddings", model=self._model, keys=sorted(body)
            )

        vectors = [tuple(float(value) for value in vector) for vector in raw]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
            logger.info(
                "embedding_dimension_discovered", model=self._model, dimension=self._dimension
            )
        return vectors

    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingError(
                "Embedding dimension is unknown until the model has been called once",
                model=self._model,
            )
        return self._dimension

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            provider=self.name,
            model_name=self._model,
            dimension=self.dimension(),
            preprocessing_version=self._preprocessing_version,
        )

    async def health(self) -> EmbeddingHealth:
        try:
            await self.embed_text("health check")
        except EmbeddingError as exc:
            return EmbeddingHealth(healthy=False, detail=exc.message)
        return EmbeddingHealth(healthy=True, identity=self.model_identity())

    async def warm_up(self) -> ModelIdentity:
        """Discover the dimension by embedding once. Cheap, and makes identity available."""
        await self.embed_text("warm up")
        return self.model_identity()

    async def aclose(self) -> None:
        await self._client.aclose()


class NullEmbeddingModel(EmbeddingModel):
    """Used when embeddings are disabled.

    Fails loudly on use but reports unhealthy rather than raising on probe, so the rest
    of the system degrades to lexical retrieval instead of refusing to start.
    """

    name: ClassVar[str] = "null"

    async def embed_text(self, text: str) -> tuple[float, ...]:
        raise EmbeddingError("Embeddings are disabled in configuration", model="null")

    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        raise EmbeddingError("Embeddings are disabled in configuration", model="null")

    def dimension(self) -> int:
        return 0

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider="null", model_name="disabled", dimension=1)

    async def health(self) -> EmbeddingHealth:
        return EmbeddingHealth(healthy=False, detail="embeddings disabled")

    async def aclose(self) -> None:
        await asyncio.sleep(0)
