"""Paper persistence port.

Services depend on this, never on a filesystem layout or a database driver. v0.3 ships a
JSON-sidecar implementation; moving to PostgreSQL later is a new adapter, not a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from researchagent.models.library import PaperRecord


class PaperRepository(ABC):
    @abstractmethod
    async def get(self, paper_id: str) -> PaperRecord | None: ...

    @abstractmethod
    async def save(self, record: PaperRecord) -> PaperRecord:
        """Insert or update. Implementations merge rather than clobber, so a second
        discovery run cannot erase pipeline flags set by a later stage."""

    @abstractmethod
    async def save_many(self, records: Sequence[PaperRecord]) -> list[PaperRecord]: ...

    @abstractmethod
    async def list_all(self) -> list[PaperRecord]: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...

    async def find_pending(self, flag: str) -> list[PaperRecord]:
        """Records whose ``ProcessingStatus.<flag>`` is still False.

        The hook every later version uses to find its outstanding work.
        """
        records = await self.list_all()
        return [r for r in records if getattr(r.processing, flag) is False]
