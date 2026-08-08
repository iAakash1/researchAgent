"""OpenAlex adapter.

OpenAlex is the broadest open index and the only one that reliably reports open-access
PDF locations, so it is the main supplier of retrievable full text outside arXiv.

Its one oddity: abstracts ship as an inverted index (token -> positions) for licensing
reasons, and must be reconstructed.
"""

from __future__ import annotations

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

API_URL = "https://api.openalex.org/works"

_TYPE_MAP = {
    "article": PublicationType.JOURNAL_ARTICLE,
    "journal-article": PublicationType.JOURNAL_ARTICLE,
    "proceedings-article": PublicationType.CONFERENCE_PAPER,
    "preprint": PublicationType.PREPRINT,
    "book-chapter": PublicationType.BOOK_CHAPTER,
    "dissertation": PublicationType.THESIS,
    "report": PublicationType.REPORT,
    "dataset": PublicationType.DATASET,
}


class OpenAlexSource(PaperSource):
    name = SourceName.OPENALEX

    def __init__(self, client: HttpClient) -> None:
        self._client = client

    async def search(self, query: SearchQuery) -> list[Paper]:
        payload = await self._client.get_json(
            API_URL,
            params={
                "search": query.text,
                "per-page": min(query.limit, 200),
                "filter": _build_filter(query) or None,
            },
        )
        works = payload.get("results", []) if isinstance(payload, dict) else []
        papers = [paper for work in works if (paper := self._to_paper(work))]
        logger.debug("openalex_search", query=query.text, returned=len(papers))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        # OpenAlex accepts its own ids, DOIs and PMIDs on the same endpoint.
        payload = await self._client.get_json(f"{API_URL}/{identifier}")
        return self._to_paper(payload) if isinstance(payload, dict) else None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        if not paper.pdf_url:
            raise PaperNotFoundError(
                "No open-access PDF location", paper_id=paper.id, source=self.name.value
            )
        return await self._client.download(paper.pdf_url, destination)

    async def health(self) -> SourceHealth:
        reachable = await self._client.is_reachable(API_URL, params={"per-page": 1})
        return SourceHealth(source=self.name, healthy=reachable)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_paper(self, work: dict[str, Any]) -> Paper | None:
        title = work.get("display_name") or work.get("title")
        if not title:
            return None

        identifiers = PaperIdentifiers(
            doi=work.get("doi"),
            openalex_id=_short_id(work.get("id")),
            pubmed_id=_short_id((work.get("ids") or {}).get("pmid")),
        )
        best_location = work.get("best_oa_location") or work.get("primary_location") or {}
        open_access = work.get("open_access") or {}

        return Paper(
            id=make_paper_id(identifiers, self.name, title),
            title=title,
            authors=[
                Author(
                    name=name,
                    affiliation=next(
                        (i.get("display_name") for i in (entry.get("institutions") or [])), None
                    ),
                    orcid=_short_id((entry.get("author") or {}).get("orcid")),
                )
                for entry in work.get("authorships") or []
                if (name := (entry.get("author") or {}).get("display_name"))
            ],
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            year=work.get("publication_year"),
            venue=(best_location.get("source") or {}).get("display_name"),
            identifiers=identifiers,
            url=work.get("doi") or work.get("id"),
            pdf_url=best_location.get("pdf_url") or open_access.get("oa_url"),
            provider=self.name,
            keywords=[
                name
                for concept in (work.get("concepts") or [])[:8]
                if (name := concept.get("display_name"))
            ],
            citation_count=work.get("cited_by_count"),
            publication_type=_TYPE_MAP.get(work.get("type", ""), PublicationType.UNKNOWN),
            is_open_access=open_access.get("is_oa"),
            source_metadata={
                "referenced_works_count": len(work.get("referenced_works") or []),
                "type": work.get("type"),
                "language": work.get("language"),
                "publication_date": work.get("publication_date"),
            },
        )


def _build_filter(query: SearchQuery) -> str:
    filters = []
    if query.year_from is not None:
        filters.append(f"from_publication_date:{query.year_from}-01-01")
    if query.year_to is not None:
        filters.append(f"to_publication_date:{query.year_to}-12-31")
    if query.open_access_only:
        filters.append("is_oa:true")
    return ",".join(filters)


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild text from OpenAlex's ``{token: [positions]}`` representation."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = [
        (position, token) for token, spots in inverted_index.items() for position in spots
    ]
    if not positions:
        return None
    positions.sort()
    return " ".join(token for _, token in positions)


def _short_id(url: str | None) -> str | None:
    """OpenAlex returns ids as URLs (``https://openalex.org/W123``); keep the tail."""
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1] or None
