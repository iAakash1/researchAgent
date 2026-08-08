"""Paper domain model — the single shape every provider normalises into.

No provider-specific model may escape ``integrations/``. Anything a provider knows that
this model does not is preserved verbatim in ``source_metadata``, so later versions
(parsing, extraction, knowledge graph) can mine it without another round of API calls.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")
_ARXIV_PREFIX = re.compile(r"^(https?://arxiv\.org/(abs|pdf)/|arxiv:)", re.IGNORECASE)
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


class SourceName(StrEnum):
    """Every provider that can produce papers."""

    ARXIV = "arxiv"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    MANUAL = "manual"


class PublicationType(StrEnum):
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    REPORT = "report"
    DATASET = "dataset"
    SPECIFICATION = "specification"
    UNKNOWN = "unknown"


class Author(BaseModel):
    name: str
    affiliation: str | None = None
    orcid: str | None = None

    def __str__(self) -> str:
        return self.name


class PaperIdentifiers(BaseModel):
    """External ids, normalised. Grouped rather than flat because deduplication and the
    v0.7 knowledge graph both need to reason over the whole set."""

    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    pubmed_id: str | None = None
    corpus_id: str | None = None

    @field_validator("doi")
    @classmethod
    def _normalise_doi(cls, value: str | None) -> str | None:
        if not value:
            return None
        return _DOI_PREFIX.sub("", value.strip()).lower() or None

    @field_validator("arxiv_id")
    @classmethod
    def _normalise_arxiv(cls, value: str | None) -> str | None:
        if not value:
            return None
        stripped = _ARXIV_PREFIX.sub("", value.strip())
        stripped = stripped.removesuffix(".pdf")
        # Versions are the same work: 2401.12345v2 must dedupe against 2401.12345.
        return _ARXIV_VERSION.sub("", stripped).lower() or None

    def merge(self, other: PaperIdentifiers) -> PaperIdentifiers:
        """Fill gaps from another record of the same paper; never overwrite."""
        merged = self.model_dump()
        for field, value in other.model_dump().items():
            if merged.get(field) is None and value is not None:
                merged[field] = value
        return PaperIdentifiers.model_validate(merged)

    def any_present(self) -> bool:
        return any(value is not None for value in self.model_dump().values())


class Paper(BaseModel):
    """A discovered paper, normalised across providers."""

    id: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = Field(default=None, ge=1500, le=2200)
    venue: str | None = None
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)

    url: str | None = None
    pdf_url: str | None = None
    # Set once the PDF is on disk (immediately true for manual papers).
    local_path: Path | None = None

    provider: SourceName
    # Providers that returned this same paper; populated by deduplication. Agreement
    # across independent indexes is itself evidence, and v0.6 verification will use it.
    also_seen_in: list[SourceName] = Field(default_factory=list)

    keywords: list[str] = Field(default_factory=list)
    citation_count: int | None = Field(default=None, ge=0)
    publication_type: PublicationType = PublicationType.UNKNOWN
    is_open_access: bool | None = None

    # Untouched provider payload, kept for downstream versions.
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def doi(self) -> str | None:
        return self.identifiers.doi

    @property
    def arxiv_id(self) -> str | None:
        return self.identifiers.arxiv_id

    @property
    def is_retrievable(self) -> bool:
        """Whether a PDF can be obtained without a paywall dance."""
        return self.local_path is not None or self.pdf_url is not None

    @property
    def normalised_title(self) -> str:
        return normalise_title(self.title)

    @property
    def author_names(self) -> list[str]:
        return [author.name for author in self.authors]

    def searchable_text(self) -> str:
        """Text used for relevance scoring. Keywords are included because several
        providers supply curated subject terms that the abstract never repeats."""
        parts = [self.title, self.abstract or "", " ".join(self.keywords), self.venue or ""]
        return " ".join(part for part in parts if part)


def normalise_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the dedup comparison key."""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", title.lower())).strip()


def make_paper_id(identifiers: PaperIdentifiers, provider: SourceName, title: str) -> str:
    """Deterministic, namespaced id.

    Two providers returning the same DOI produce the same id, which is what lets a run be
    re-executed without duplicating records on disk. Falls back to a title hash only when
    a paper carries no external identifier at all.
    """
    if identifiers.doi:
        return f"doi:{identifiers.doi}"
    if identifiers.arxiv_id:
        return f"arxiv:{identifiers.arxiv_id}"
    if identifiers.pubmed_id:
        return f"pmid:{identifiers.pubmed_id}"
    if identifiers.openalex_id:
        return f"openalex:{identifiers.openalex_id}"
    if identifiers.semantic_scholar_id:
        return f"s2:{identifiers.semantic_scholar_id}"

    digest = hashlib.sha1(normalise_title(title).encode("utf-8")).hexdigest()[:16]  # noqa: S324
    return f"{provider.value}:{digest}"
