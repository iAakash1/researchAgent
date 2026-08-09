"""Embedding model port.

The identity of the model that produced a vector is part of the vector's meaning.
Vectors from two models occupy different spaces, and comparing them produces confident
nonsense — so :class:`ModelIdentity` is persisted with every index and checked before a
query is served.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field


class ModelIdentity(BaseModel):
    """Everything that makes two vector spaces incompatible.

    Any change here must produce a new index version. Silently reusing vectors across a
    model change is the failure that is hardest to notice, because retrieval keeps
    returning results — just wrong ones.
    """

    model_config = {"frozen": True, "protected_namespaces": ()}

    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    # Bumped when preprocessing changes without the model changing.
    preprocessing_version: str = "1"

    @property
    def fingerprint(self) -> str:
        return f"{self.provider}/{self.model_name}/d{self.dimension}/p{self.preprocessing_version}"

    def is_compatible_with(self, other: ModelIdentity) -> bool:
        return self.fingerprint == other.fingerprint


class EmbeddingHealth(BaseModel):
    model_config = {"frozen": True}

    healthy: bool
    identity: ModelIdentity | None = None
    detail: str | None = None


class EmbeddingModel(ABC):
    """Turns text into vectors."""

    name: ClassVar[str]

    @abstractmethod
    async def embed_text(self, text: str) -> tuple[float, ...]:
        """Embed one string. Raises ``EmbeddingError`` when the backend is unreachable."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed many strings. Order of results matches order of input."""

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def model_identity(self) -> ModelIdentity: ...

    @abstractmethod
    async def health(self) -> EmbeddingHealth:
        """Cheap probe; must not raise."""

    @abstractmethod
    async def aclose(self) -> None: ...
