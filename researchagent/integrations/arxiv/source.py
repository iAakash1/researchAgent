"""arXiv adapter.

arXiv serves Atom XML, not JSON, and its PDFs are always open access — it is the most
reliable source of actually-retrievable full text, so it is worth parsing XML for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from researchagent.core.exceptions import PaperNotFoundError, SourceResponseError
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

API_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSource(PaperSource):
    name = SourceName.ARXIV

    def __init__(self, client: HttpClient) -> None:
        self._client = client

    async def search(self, query: SearchQuery) -> list[Paper]:
        payload = await self._client.get_text(
            API_URL,
            params={
                "search_query": _build_search_expression(query),
                "start": 0,
                "max_results": query.limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        papers = [paper for entry in _entries(payload) if (paper := self._to_paper(entry))]
        # arXiv has no date filter in the query language that composes reliably with
        # boolean terms, so recency is applied after the fact.
        if query.year_from is not None:
            papers = [p for p in papers if p.year is None or p.year >= query.year_from]
        logger.debug("arxiv_search", query=query.text, returned=len(papers))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        payload = await self._client.get_text(
            API_URL, params={"id_list": identifier, "max_results": 1}
        )
        for entry in _entries(payload):
            return self._to_paper(entry)
        return None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        url = paper.pdf_url or (
            f"https://arxiv.org/pdf/{paper.arxiv_id}" if paper.arxiv_id else None
        )
        if url is None:
            raise PaperNotFoundError(
                "Paper has no arXiv PDF location", paper_id=paper.id, source=self.name.value
            )
        return await self._client.download(url, destination)

    async def health(self) -> SourceHealth:
        reachable = await self._client.is_reachable(
            API_URL, params={"search_query": "all:test", "max_results": 1}
        )
        return SourceHealth(source=self.name, healthy=reachable)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_paper(self, entry: ElementTree.Element) -> Paper | None:
        title = _text(entry, "atom:title")
        if not title:
            return None

        arxiv_id = _arxiv_id_from_url(_text(entry, "atom:id"))
        identifiers = PaperIdentifiers(arxiv_id=arxiv_id, doi=_text(entry, "arxiv:doi"))

        pdf_url = next(
            (
                link.get("href")
                for link in entry.findall("atom:link", _NS)
                if link.get("title") == "pdf" or link.get("type") == "application/pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
        )

        return Paper(
            id=make_paper_id(identifiers, self.name, title),
            title=title,
            authors=[
                Author(name=name)
                for author in entry.findall("atom:author", _NS)
                if (name := _text(author, "atom:name"))
            ],
            abstract=_text(entry, "atom:summary"),
            year=_year(_text(entry, "atom:published")),
            venue=_text(entry, "arxiv:journal_ref") or "arXiv",
            identifiers=identifiers,
            url=_text(entry, "atom:id"),
            pdf_url=pdf_url,
            provider=self.name,
            keywords=[
                term
                for category in entry.findall("atom:category", _NS)
                if (term := category.get("term"))
            ],
            publication_type=PublicationType.PREPRINT,
            is_open_access=True,
            source_metadata={
                "primary_category": _attr(entry, "arxiv:primary_category", "term"),
                "comment": _text(entry, "arxiv:comment"),
                "updated": _text(entry, "atom:updated"),
            },
        )


def _build_search_expression(query: SearchQuery) -> str:
    """arXiv's query language: field-prefixed terms joined by AND/OR.

    The free-text query goes against all fields; explicit plan keywords are OR-ed into
    the abstract field so a paper matching any of them still surfaces.
    """
    expression = f'all:"{query.text}"' if " " in query.text else f"all:{query.text}"
    if query.terms:
        keyword_clause = " OR ".join(f'abs:"{term}"' for term in query.terms[:6])
        expression = f"({expression}) OR ({keyword_clause})"
    return expression


def _entries(payload: str) -> list[ElementTree.Element]:
    try:
        root = ElementTree.fromstring(payload)  # noqa: S314 - arXiv is a trusted endpoint
    except ElementTree.ParseError as exc:
        raise SourceResponseError(
            "arXiv returned malformed XML", source=SourceName.ARXIV.value, reason=str(exc)
        ) from exc
    return root.findall("atom:entry", _NS)


def _text(element: ElementTree.Element, path: str) -> str | None:
    found = element.find(path, _NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split()) or None


def _attr(element: ElementTree.Element, path: str, attribute: str) -> Any:
    found = element.find(path, _NS)
    return None if found is None else found.get(attribute)


def _year(published: str | None) -> int | None:
    if not published or len(published) < 4:
        return None
    try:
        return int(published[:4])
    except ValueError:
        return None


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1] or None
