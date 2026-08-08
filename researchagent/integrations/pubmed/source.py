"""PubMed adapter (NCBI E-utilities).

The only two-step provider: ``esearch`` returns PMIDs, ``efetch`` returns the records.
Both calls go through the same rate limiter, which matters here because NCBI enforces
3 requests/second for anonymous clients and blocks abusers outright.

Biomedical coverage that none of the other four match — indispensable for any clinical
research goal.
"""

from __future__ import annotations

from pathlib import Path
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

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{BASE_URL}/esearch.fcgi"
EFETCH_URL = f"{BASE_URL}/efetch.fcgi"


class PubMedSource(PaperSource):
    name = SourceName.PUBMED

    def __init__(self, client: HttpClient) -> None:
        self._client = client

    @property
    def supports_download(self) -> bool:
        # PubMed indexes abstracts; full text lives in PubMed Central, which is a
        # different corpus and arrives with the retrieval work in a later version.
        return False

    async def search(self, query: SearchQuery) -> list[Paper]:
        pmids = await self._search_ids(query)
        if not pmids:
            return []
        papers = await self._fetch(pmids)
        logger.debug("pubmed_search", query=query.text, returned=len(papers))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        papers = await self._fetch([identifier])
        return papers[0] if papers else None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        raise PaperNotFoundError(
            "PubMed does not serve PDFs; use the DOI or an open-access mirror",
            paper_id=paper.id,
            source=self.name.value,
        )

    async def health(self) -> SourceHealth:
        reachable = await self._client.is_reachable(
            ESEARCH_URL, params={"db": "pubmed", "term": "test", "retmax": 1, "retmode": "json"}
        )
        return SourceHealth(source=self.name, healthy=reachable)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _search_ids(self, query: SearchQuery) -> list[str]:
        payload = await self._client.get_json(
            ESEARCH_URL,
            params={
                "db": "pubmed",
                "term": _build_term(query),
                "retmax": min(query.limit, 100),
                "retmode": "json",
                "sort": "relevance",
            },
        )
        result = (payload or {}).get("esearchresult") or {}
        return [str(pmid) for pmid in result.get("idlist", [])]

    async def _fetch(self, pmids: list[str]) -> list[Paper]:
        payload = await self._client.get_text(
            EFETCH_URL, params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        )
        try:
            root = ElementTree.fromstring(payload)  # noqa: S314 - NCBI is a trusted endpoint
        except ElementTree.ParseError as exc:
            raise SourceResponseError(
                "PubMed returned malformed XML", source=self.name.value, reason=str(exc)
            ) from exc

        return [
            paper
            for article in root.findall(".//PubmedArticle")
            if (paper := self._to_paper(article))
        ]

    def _to_paper(self, article: ElementTree.Element) -> Paper | None:
        title = _joined_text(article.find(".//ArticleTitle"))
        if not title:
            return None

        identifiers = PaperIdentifiers(
            pubmed_id=_id_of(article, "pubmed"),
            doi=_id_of(article, "doi"),
        )
        # Abstracts are split into labelled sections (BACKGROUND, METHODS, ...).
        sections = [
            f"{label}: {text}" if (label := part.get("Label")) else text
            for part in article.findall(".//Abstract/AbstractText")
            if (text := _joined_text(part))
        ]

        return Paper(
            id=make_paper_id(identifiers, self.name, title),
            title=title,
            authors=[
                Author(
                    name=name,
                    affiliation=_joined_text(entry.find(".//Affiliation")),
                )
                for entry in article.findall(".//Author")
                if (name := _author_name(entry))
            ],
            abstract=" ".join(sections) or None,
            year=_year(article),
            venue=_joined_text(article.find(".//Journal/Title")),
            identifiers=identifiers,
            url=(
                f"https://pubmed.ncbi.nlm.nih.gov/{identifiers.pubmed_id}/"
                if identifiers.pubmed_id
                else None
            ),
            provider=self.name,
            keywords=[
                term
                for heading in article.findall(".//MeshHeading/DescriptorName")
                if (term := _joined_text(heading))
            ][:10],
            publication_type=PublicationType.JOURNAL_ARTICLE,
            source_metadata={
                "journal_abbrev": _joined_text(article.find(".//Journal/ISOAbbreviation")),
                "publication_types": [
                    text
                    for entry in article.findall(".//PublicationType")
                    if (text := _joined_text(entry))
                ],
            },
        )


def _build_term(query: SearchQuery) -> str:
    term = query.text
    if query.year_from is not None:
        term = f"{term} AND {query.year_from}:{query.year_to or 3000}[dp]"
    return term


def _joined_text(element: ElementTree.Element | None) -> str | None:
    """PubMed mixes inline markup into text nodes; ``itertext`` flattens it."""
    if element is None:
        return None
    return " ".join("".join(element.itertext()).split()) or None


def _author_name(entry: ElementTree.Element) -> str | None:
    fore = _joined_text(entry.find("ForeName"))
    last = _joined_text(entry.find("LastName"))
    if fore and last:
        return f"{fore} {last}"
    return last or _joined_text(entry.find("CollectiveName"))


def _id_of(article: ElementTree.Element, id_type: str) -> str | None:
    for identifier in article.findall(".//ArticleId"):
        if identifier.get("IdType") == id_type:
            return _joined_text(identifier)
    return None


def _year(article: ElementTree.Element) -> int | None:
    for path in (".//PubDate/Year", ".//ArticleDate/Year", ".//DateCompleted/Year"):
        text = _joined_text(article.find(path))
        if text and text.isdigit():
            return int(text)
    medline = _joined_text(article.find(".//PubDate/MedlineDate"))
    if medline and medline[:4].isdigit():
        return int(medline[:4])
    return None
