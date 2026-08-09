"""Bundle persistence port.

Bundles are expensive to assemble and stable once built: the same question asked in a
later session should find the existing bundle rather than rebuild it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from researchagent.models.bundle import EvidenceBundle


class BundleRepository(ABC):
    @abstractmethod
    async def get(self, bundle_id: str) -> EvidenceBundle | None: ...

    @abstractmethod
    async def save(self, bundle: EvidenceBundle) -> EvidenceBundle: ...

    @abstractmethod
    async def for_question(self, question_id: str) -> tuple[EvidenceBundle, ...]:
        """Bundles built to answer a specific research question."""

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, bundle_id: str) -> bool: ...
