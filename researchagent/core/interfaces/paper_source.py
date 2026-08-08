"""Literature provider port.

The Planner produces questions and queries; it never learns which index answered them.
Adding PubMed Central or a private corpus later means one adapter under
``integrations/`` plus one line in ``config/sources.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from researchagent.models.paper import Paper, SourceName


class SearchQuery(BaseModel):
    """One search against one provider.

    ``terms`` carries the plan's keywords separately from the free-text query because
    several providers support field-restricted search and would otherwise lose them.
    """

    text: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=200)
    year_from: int | None = Field(default=None, ge=1500, le=2200)
    year_to: int | None = Field(default=None, ge=1500, le=2200)
    terms: list[str] = Field(default_factory=list)
    open_access_only: bool = False

    def with_limit(self, limit: int) -> SearchQuery:
        return self.model_copy(update={"limit": limit})


class SourceHealth(BaseModel):
    source: SourceName
    healthy: bool
    detail: str | None = None


class PaperSource(ABC):
    """A searchable literature index."""

    name: SourceName

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Paper]:
        """Return normalised papers. Raises ``PaperSourceError`` subclasses on failure.

        Implementations must never raise on "no results" — that is an empty list.
        """

    @abstractmethod
    async def get_paper(self, identifier: str) -> Paper | None:
        """Fetch one paper by this provider's identifier. ``None`` when absent."""

    @abstractmethod
    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        """Write the PDF to ``destination`` and return the path actually written.

        Raises ``PaperNotFoundError`` when the provider has no retrievable PDF.
        """

    @abstractmethod
    async def health(self) -> SourceHealth:
        """Cheap liveness probe; must not raise."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release sockets and background resources."""

    @property
    def supports_download(self) -> bool:
        """Whether this provider can generally serve PDFs.

        Metadata-only indexes (Crossref, PubMed abstracts) say False so the retrieval
        service does not waste requests on them.
        """
        return True
