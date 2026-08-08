"""Crossref adapter.

Crossref is the DOI registration authority: its metadata (venue, publisher, type) is the
most authoritative, which makes it the best source of *identifiers* even though it
almost never provides full text. Metadata-only by design — see ``supports_download``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from researchagent.core.exceptions import PaperNotFoundError
from researchagent.core.interfaces.paper_source import PaperSource, SearchQuery, SourceHealth
from researchagent.core.logging import get_logger
from researchagent.integrations.http import HttpClient
from researchagent.models.paper import (
    Author,
    Paper,
    PaperIdentifiers,
    PublicationType,
    SourceName,
    make_paper_id,
)

logger = get_logger(__name__)

API_URL = "https://api.crossref.org/works"

_TYPE_MAP = {
    "journal-article": PublicationType.JOURNAL_ARTICLE,
    "proceedings-article": PublicationType.CONFERENCE_PAPER,
    "posted-content": PublicationType.PREPRINT,
    "book-chapter": PublicationType.BOOK_CHAPTER,
    "dissertation": PublicationType.THESIS,
    "report": PublicationType.REPORT,
    "dataset": PublicationType.DATASET,
    "standard": PublicationType.SPECIFICATION,
}


class CrossrefSource(PaperSource):
    name = SourceName.CROSSREF

    def __init__(self, client: HttpClient) -> None:
        self._client = client

    @property
    def supports_download(self) -> bool:
        return False

    async def search(self, query: SearchQuery) -> list[Paper]:
        payload = await self._client.get_json(
            API_URL,
            params={
                "query.bibliographic": query.text,
                "rows": min(query.limit, 100),
                "select": (
                    "DOI,title,author,abstract,issued,container-title,type,"
                    "is-referenced-by-count,subject,URL,link,publisher"
                ),
                "filter": _build_filter(query) or None,
                "sort": "relevance",
            },
        )
        items = ((payload or {}).get("message") or {}).get("items", [])
        papers = [paper for item in items if (paper := self._to_paper(item))]
        logger.debug("crossref_search", query=query.text, returned=len(papers))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        payload = await self._client.get_json(f"{API_URL}/{identifier}")
        item = (payload or {}).get("message")
        return self._to_paper(item) if isinstance(item, dict) else None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        # Crossref `link` entries are usually publisher landing pages behind a paywall;
        # only follow one when it is explicitly typed as a PDF.
        if not paper.pdf_url:
            raise PaperNotFoundError(
                "Crossref exposes metadata only for this record",
                paper_id=paper.id,
                source=self.name.value,
            )
        return await self._client.download(paper.pdf_url, destination)

    async def health(self) -> SourceHealth:
        reachable = await self._client.is_reachable(API_URL, params={"rows": 1})
        return SourceHealth(source=self.name, healthy=reachable)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        title = next(iter(item.get("title") or []), None)
        if not title:
            return None

        identifiers = PaperIdentifiers(doi=item.get("DOI"))
        pdf_link = next(
            (
                link.get("URL")
                for link in item.get("link") or []
                if link.get("content-type") == "application/pdf"
            ),
            None,
        )

        return Paper(
            id=make_paper_id(identifiers, self.name, title),
            title=" ".join(title.split()),
            authors=[
                Author(
                    name=name,
                    affiliation=next((a.get("name") for a in entry.get("affiliation") or []), None),
                    orcid=entry.get("ORCID"),
                )
                for entry in item.get("author") or []
                if (name := _author_name(entry))
            ],
            abstract=_strip_jats(item.get("abstract")),
            year=_issued_year(item.get("issued")),
            venue=next(iter(item.get("container-title") or []), None),
            identifiers=identifiers,
            url=item.get("URL"),
            pdf_url=pdf_link,
            provider=self.name,
            keywords=item.get("subject") or [],
            citation_count=item.get("is-referenced-by-count"),
            publication_type=_TYPE_MAP.get(item.get("type", ""), PublicationType.UNKNOWN),
            source_metadata={"publisher": item.get("publisher"), "type": item.get("type")},
        )


def _build_filter(query: SearchQuery) -> str:
    filters = []
    if query.year_from is not None:
        filters.append(f"from-pub-date:{query.year_from}-01-01")
    if query.year_to is not None:
        filters.append(f"until-pub-date:{query.year_to}-12-31")
    return ",".join(filters)


def _author_name(entry: dict[str, Any]) -> str | None:
    given, family = entry.get("given"), entry.get("family")
    if given and family:
        return f"{given} {family}"
    return family or given or entry.get("name")


def _issued_year(issued: dict[str, Any] | None) -> int | None:
    parts = (issued or {}).get("date-parts") or []
    first = next(iter(parts), None)
    if not first or not first[0]:
        return None
    try:
        return int(first[0])
    except (TypeError, ValueError):
        return None


def _strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts are JATS XML fragments; strip the tags, keep the prose."""
    if not abstract:
        return None
    text = re.sub(r"<[^>]+>", " ", abstract)
    return " ".join(text.split()) or None
