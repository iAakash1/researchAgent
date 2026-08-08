"""Remote provider tests. Every response is mocked — no test touches a real API.

What is under test is our normalisation and error mapping, not the providers.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from researchagent.core.exceptions import (
    SourceRateLimitedError,
    SourceResponseError,
    SourceUnavailableError,
)
from researchagent.core.interfaces.paper_source import SearchQuery
from researchagent.integrations.arxiv import ArxivSource
from researchagent.integrations.crossref import CrossrefSource
from researchagent.integrations.http import HttpClient
from researchagent.integrations.openalex import OpenAlexSource
from researchagent.integrations.pubmed import PubMedSource
from researchagent.integrations.semantic_scholar import SemanticScholarSource
from researchagent.models.paper import PublicationType, SourceName

QUERY = SearchQuery(text="metastable failures", limit=5, year_from=2022, terms=["overload"])


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], source: str = "test"
) -> HttpClient:
    # No rate limiting in tests: the limiter is exercised separately.
    return HttpClient(source, requests_per_second=0, transport=httpx.MockTransport(handler))


def responder(
    payload: object, *, status: int = 200, text: str | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload)

    return handler


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <published>2024-01-22T10:00:00Z</published>
    <title>Metastable Failures
      in Distributed Systems</title>
    <summary>We study metastable failure modes.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <arxiv:doi>10.1145/3600006</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" type="application/pdf"/>
    <category term="cs.DC"/>
    <arxiv:primary_category term="cs.DC"/>
  </entry>
</feed>"""


async def test_arxiv_normalises_an_entry() -> None:
    source = ArxivSource(client_for(responder(None, text=ARXIV_FEED)))

    papers = await source.search(QUERY)

    assert len(papers) == 1
    paper = papers[0]
    # Whitespace inside the XML title must be collapsed, and the version stripped.
    assert paper.title == "Metastable Failures in Distributed Systems"
    assert paper.identifiers.arxiv_id == "2401.12345"
    assert paper.identifiers.doi == "10.1145/3600006"
    assert paper.id == "doi:10.1145/3600006"  # DOI wins over the arXiv id
    assert paper.author_names == ["Ada Lovelace", "Grace Hopper"]
    assert paper.year == 2024
    assert paper.pdf_url == "http://arxiv.org/pdf/2401.12345v2"
    assert paper.publication_type is PublicationType.PREPRINT
    assert paper.is_open_access is True
    assert paper.keywords == ["cs.DC"]
    await source.aclose()


async def test_arxiv_applies_the_year_filter_client_side() -> None:
    old_feed = ARXIV_FEED.replace("2024-01-22", "2015-01-22")
    source = ArxivSource(client_for(responder(None, text=old_feed)))

    assert await source.search(QUERY) == []
    await source.aclose()


async def test_arxiv_malformed_xml_is_a_response_error() -> None:
    source = ArxivSource(client_for(responder(None, text="<feed><entry>")))

    with pytest.raises(SourceResponseError):
        await source.search(QUERY)
    await source.aclose()


OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "display_name": "Metastable failures in the wild",
            "doi": "https://doi.org/10.1145/3600007",
            "publication_year": 2023,
            "cited_by_count": 57,
            "type": "article",
            "abstract_inverted_index": {"Metastable": [0], "failures": [1], "happen": [2]},
            "authorships": [
                {
                    "author": {"display_name": "Ada Lovelace", "orcid": "https://orcid.org/0001"},
                    "institutions": [{"display_name": "Cambridge"}],
                }
            ],
            "best_oa_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "source": {"display_name": "SOSP"},
            },
            "open_access": {"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
            "concepts": [{"display_name": "Distributed computing"}],
        }
    ]
}


async def test_openalex_reconstructs_the_inverted_abstract() -> None:
    source = OpenAlexSource(client_for(responder(OPENALEX_PAYLOAD)))

    paper = (await source.search(QUERY))[0]

    assert paper.abstract == "Metastable failures happen"
    assert paper.identifiers.doi == "10.1145/3600007"
    assert paper.identifiers.openalex_id == "W123"
    assert paper.citation_count == 57
    assert paper.venue == "SOSP"
    assert paper.authors[0].affiliation == "Cambridge"
    assert paper.authors[0].orcid == "0001"
    assert paper.publication_type is PublicationType.JOURNAL_ARTICLE
    assert paper.pdf_url == "https://example.org/paper.pdf"
    await source.aclose()


async def test_openalex_handles_a_missing_abstract() -> None:
    payload = {"results": [{"id": "https://openalex.org/W1", "display_name": "No abstract"}]}
    source = OpenAlexSource(client_for(responder(payload)))

    paper = (await source.search(QUERY))[0]

    assert paper.abstract is None
    await source.aclose()


CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "10.1145/3600008",
                "title": ["Congestion  Avoidance and Control"],
                "abstract": "<jats:p>We describe congestion control.</jats:p>",
                "author": [{"given": "Van", "family": "Jacobson"}],
                "issued": {"date-parts": [[1988, 8]]},
                "container-title": ["SIGCOMM"],
                "type": "proceedings-article",
                "is-referenced-by-count": 9001,
                "subject": ["Networks"],
                "URL": "https://doi.org/10.1145/3600008",
            }
        ]
    }
}


async def test_crossref_strips_jats_markup() -> None:
    source = CrossrefSource(client_for(responder(CROSSREF_PAYLOAD)))

    paper = (await source.search(QUERY))[0]

    assert paper.abstract == "We describe congestion control."
    assert paper.title == "Congestion Avoidance and Control"
    assert paper.year == 1988
    assert paper.authors[0].name == "Van Jacobson"
    assert paper.publication_type is PublicationType.CONFERENCE_PAPER
    assert paper.citation_count == 9001
    assert source.supports_download is False
    await source.aclose()


S2_PAYLOAD = {
    "data": [
        {
            "paperId": "abc123",
            "corpusId": 999,
            "externalIds": {"DOI": "10.1/x", "ArXiv": "2401.55555", "PubMed": "12345"},
            "title": "Why do multi-agent LLM systems fail",
            "abstract": "We taxonomise failure modes.",
            "year": 2025,
            "venue": "NeurIPS",
            "authors": [{"name": "Ada Lovelace"}],
            "citationCount": 12,
            "openAccessPdf": {"url": "https://example.org/mast.pdf"},
            "publicationTypes": ["JournalArticle"],
            "fieldsOfStudy": ["Computer Science"],
            "isOpenAccess": True,
        }
    ]
}


async def test_semantic_scholar_collects_every_identifier() -> None:
    source = SemanticScholarSource(client_for(responder(S2_PAYLOAD)))

    paper = (await source.search(QUERY))[0]

    assert paper.identifiers.doi == "10.1/x"
    assert paper.identifiers.arxiv_id == "2401.55555"
    assert paper.identifiers.pubmed_id == "12345"
    assert paper.identifiers.semantic_scholar_id == "abc123"
    assert paper.identifiers.corpus_id == "999"
    assert paper.citation_count == 12
    assert paper.pdf_url == "https://example.org/mast.pdf"
    await source.aclose()


PUBMED_SEARCH = {"esearchresult": {"idlist": ["40000001"]}}
PUBMED_FETCH = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <Article>
        <ArticleTitle>Agentic AI for <i>clinical</i> triage</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Triage is hard.</AbstractText>
          <AbstractText Label="RESULTS">Agents help.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName>
            <AffiliationInfo><Affiliation>Cambridge</Affiliation></AffiliationInfo>
          </Author>
        </AuthorList>
        <Journal><Title>Journal of Medical AI</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Triage</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">40000001</ArticleId>
      <ArticleId IdType="doi">10.1/med</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


async def test_pubmed_two_step_flow() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if "esearch" in request.url.path:
            return httpx.Response(200, json=PUBMED_SEARCH)
        return httpx.Response(200, text=PUBMED_FETCH)

    source = PubMedSource(client_for(handler))
    papers = await source.search(QUERY)

    assert len(calls) == 2  # esearch, then efetch
    paper = papers[0]
    # Inline markup inside the title must be flattened, not dropped.
    assert paper.title == "Agentic AI for clinical triage"
    assert paper.abstract == "BACKGROUND: Triage is hard. RESULTS: Agents help."
    assert paper.identifiers.pubmed_id == "40000001"
    assert paper.identifiers.doi == "10.1/med"
    assert paper.year == 2024
    assert paper.keywords == ["Triage"]
    assert paper.authors[0].affiliation == "Cambridge"
    assert source.supports_download is False
    await source.aclose()


async def test_pubmed_no_hits_skips_the_fetch() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    source = PubMedSource(client_for(handler))

    assert await source.search(QUERY) == []
    assert len(calls) == 1
    await source.aclose()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, SourceRateLimitedError),
        (500, SourceUnavailableError),
        (503, SourceUnavailableError),
        (400, SourceResponseError),
    ],
)
async def test_http_status_maps_to_a_domain_error(status: int, expected: type[Exception]) -> None:
    source = OpenAlexSource(client_for(responder({}, status=status)))

    with pytest.raises(expected):
        await source.search(QUERY)
    await source.aclose()


async def test_search_never_raises_on_zero_results() -> None:
    for source in (
        OpenAlexSource(client_for(responder({"results": []}))),
        CrossrefSource(client_for(responder({"message": {"items": []}}))),
        SemanticScholarSource(client_for(responder({"data": []}))),
    ):
        assert await source.search(QUERY) == []
        await source.aclose()


async def test_every_source_reports_its_name() -> None:
    assert ArxivSource(client_for(responder({}))).name is SourceName.ARXIV
    assert OpenAlexSource(client_for(responder({}))).name is SourceName.OPENALEX
    assert CrossrefSource(client_for(responder({}))).name is SourceName.CROSSREF
    assert SemanticScholarSource(client_for(responder({}))).name is SourceName.SEMANTIC_SCHOLAR
    assert PubMedSource(client_for(responder({}))).name is SourceName.PUBMED
