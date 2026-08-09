"""Knowledge persistence port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from researchagent.schemas.knowledge import ValidatedKnowledge


class KnowledgeRepository(ABC):
    """Stores validated knowledge, always with the verdict that admitted it."""

    @abstractmethod
    async def get(self, paper_id: str) -> ValidatedKnowledge | None: ...

    @abstractmethod
    async def save(self, knowledge: ValidatedKnowledge) -> ValidatedKnowledge: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...
