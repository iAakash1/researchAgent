"""Semantic Scholar adapter.

Uniquely useful for citation counts, influence and open-access PDF links, which is what
makes it the strongest signal for ranking. Also the strictest rate limiter of the five:
without an API key the shared pool is roughly one request per second, so the client is
configured conservatively and a 429 is a normal, retryable event.
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

API_URL = "https://api.semanticscholar.org/graph/v1/paper"

_FIELDS = (
    "paperId,corpusId,externalIds,title,abstract,year,venue,authors,"
    "citationCount,openAccessPdf,publicationTypes,fieldsOfStudy,url,isOpenAccess"
)

_TYPE_MAP = {
    "JournalArticle": PublicationType.JOURNAL_ARTICLE,
    "Conference": PublicationType.CONFERENCE_PAPER,
    "Book": PublicationType.BOOK_CHAPTER,
    "Dataset": PublicationType.DATASET,
    "Review": PublicationType.JOURNAL_ARTICLE,
}


class SemanticScholarSource(PaperSource):
    name = SourceName.SEMANTIC_SCHOLAR

    def __init__(self, client: HttpClient) -> None:
        self._client = client

    async def search(self, query: SearchQuery) -> list[Paper]:
        payload = await self._client.get_json(
            f"{API_URL}/search",
            params={
                "query": query.text,
                "limit": min(query.limit, 100),
                "fields": _FIELDS,
                "year": _year_range(query),
                "openAccessPdf": "" if query.open_access_only else None,
            },
        )
        items = payload.get("data", []) if isinstance(payload, dict) else []
        papers = [paper for item in items if (paper := self._to_paper(item))]
        logger.debug("semantic_scholar_search", query=query.text, returned=len(papers))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        payload = await self._client.get_json(f"{API_URL}/{identifier}", params={"fields": _FIELDS})
        return self._to_paper(payload) if isinstance(payload, dict) else None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        if not paper.pdf_url:
            raise PaperNotFoundError(
                "No open-access PDF for this paper", paper_id=paper.id, source=self.name.value
            )
        return await self._client.download(paper.pdf_url, destination)

    async def health(self) -> SourceHealth:
        reachable = await self._client.is_reachable(
            f"{API_URL}/search", params={"query": "test", "limit": 1, "fields": "title"}
        )
        return SourceHealth(source=self.name, healthy=reachable)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        title = item.get("title")
        if not title:
            return None

        external = item.get("externalIds") or {}
        identifiers = PaperIdentifiers(
            doi=external.get("DOI"),
            arxiv_id=external.get("ArXiv"),
            pubmed_id=external.get("PubMed"),
            semantic_scholar_id=item.get("paperId"),
            corpus_id=str(item["corpusId"]) if item.get("corpusId") is not None else None,
        )

        return Paper(
            id=make_paper_id(identifiers, self.name, title),
            title=title,
            authors=[
                Author(name=name)
                for entry in item.get("authors") or []
                if (name := entry.get("name"))
            ],
            abstract=item.get("abstract"),
            year=item.get("year"),
            venue=item.get("venue") or None,
            identifiers=identifiers,
            url=item.get("url"),
            pdf_url=(item.get("openAccessPdf") or {}).get("url"),
            provider=self.name,
            keywords=item.get("fieldsOfStudy") or [],
            citation_count=item.get("citationCount"),
            publication_type=_first_type(item.get("publicationTypes")),
            is_open_access=item.get("isOpenAccess"),
            source_metadata={"publication_types": item.get("publicationTypes")},
        )


def _first_type(types: list[str] | None) -> PublicationType:
    for entry in types or []:
        mapped = _TYPE_MAP.get(entry)
        if mapped is not None:
            return mapped
    return PublicationType.UNKNOWN


def _year_range(query: SearchQuery) -> str | None:
    """Semantic Scholar takes an open-ended range string such as ``2022-``."""
    if query.year_from is None and query.year_to is None:
        return None
    return f"{query.year_from or ''}-{query.year_to or ''}"
