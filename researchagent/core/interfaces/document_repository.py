"""Canonical document persistence port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from researchagent.schemas.validated import ValidatedDocument


class DocumentRepository(ABC):
    """Stores validated canonical documents.

    The verdict is stored with the document, never separately: a document whose
    validation has been lost is a document nothing downstream can decide to trust.
    """

    @abstractmethod
    async def get(self, paper_id: str) -> ValidatedDocument | None: ...

    @abstractmethod
    async def save(self, document: ValidatedDocument) -> ValidatedDocument: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...
