"""Evidence persistence port.

Evidence is stored on its own terms, not as a sub-collection of knowledge. A caller that
wants a quote does not have to know which fact it supported, and the association is an
explicit link record queryable from either side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from researchagent.models.evidence import EvidenceLink, EvidenceRecord, PaperEvidence


class EvidenceRepository(ABC):
    @abstractmethod
    async def get_paper(self, paper_id: str) -> PaperEvidence | None: ...

    @abstractmethod
    async def save_paper(self, evidence: PaperEvidence) -> PaperEvidence:
        """Replace a paper's evidence index wholesale.

        Wholesale because evidence is re-derived per document: a re-parse invalidates
        every location at once, and merging stale records with fresh ones would leave
        provenance pointing at paragraphs that moved.
        """

    @abstractmethod
    async def get(self, evidence_id: str) -> EvidenceRecord | None:
        """Look up one item by its own id, without going through knowledge."""

    @abstractmethod
    async def for_objects(self, object_ids: tuple[str, ...]) -> tuple[EvidenceRecord, ...]:
        """Every evidence item linked to any of these knowledge objects."""

    @abstractmethod
    async def search(
        self, terms: tuple[str, ...], *, paper_ids: tuple[str, ...] = (), limit: int = 50
    ) -> tuple[EvidenceRecord, ...]:
        """Lexical search over quoted text.

        Deliberately part of the port: v0.7 replaces this implementation with hybrid
        retrieval, and nothing above this line changes.
        """

    @abstractmethod
    async def link(self, link: EvidenceLink) -> EvidenceLink:
        """Record an association. Adding a link never rewrites the evidence itself."""

    @abstractmethod
    async def list_paper_ids(self) -> list[str]: ...
