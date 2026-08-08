from __future__ import annotations

import asyncio
from pathlib import Path

from researchagent.config.schemas import DiscoverySettings
from researchagent.core.exceptions import SourceRateLimitedError, SourceUnavailableError
from researchagent.core.interfaces.paper_source import PaperSource, SearchQuery, SourceHealth
from researchagent.models.paper import Paper, PaperIdentifiers, SourceName
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.ranking import HeuristicScorer


def a_plan(queries: list[str] | None = None, year_from: int | None = None) -> ResearchPlan:
    return ResearchPlan(
        topic="Metastable failures in distributed systems",
        framing="A review of metastable failures, their triggers and their mitigations.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="What triggers metastable failures?",
                rationale="Triggers determine which mitigations apply at all.",
                priority=QuestionPriority.HIGH,
                keywords=["metastable", "overload"],
            )
        ],
        strategy=SearchStrategy(
            queries=queries or ["metastable failures", "overload collapse"], year_from=year_from
        ),
    )


class FakeSource(PaperSource):
    def __init__(
        self,
        name: SourceName,
        papers: list[Paper] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self._papers = papers or []
        self._error = error
        self._delay = delay
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> list[Paper]:
        self.queries.append(query)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(self._papers)

    async def get_paper(self, identifier: str) -> Paper | None:
        return None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        raise NotImplementedError

    async def health(self) -> SourceHealth:
        return SourceHealth(source=self.name, healthy=self._error is None)

    async def aclose(self) -> None:
        return None


def paper(title: str, provider: SourceName, **overrides: object) -> Paper:
    return Paper.model_validate(
        {"id": f"{provider.value}:{title[:10]}", "title": title, "provider": provider} | overrides
    )


def service(
    sources: list[PaperSource],
    *,
    settings: DiscoverySettings | None = None,
    repository: JsonPaperRepository | None = None,
) -> DiscoveryService:
    return DiscoveryService(
        sources,
        PaperDeduplicator(),
        HeuristicScorer(),
        settings or DiscoverySettings(),
        repository=repository,
    )


async def test_queries_every_enabled_source() -> None:
    arxiv = FakeSource(SourceName.ARXIV, [paper("Metastable failures", SourceName.ARXIV)])
    openalex = FakeSource(SourceName.OPENALEX, [paper("Overload collapse", SourceName.OPENALEX)])

    result = await service([arxiv, openalex]).discover(a_plan())

    assert len(arxiv.queries) == 2  # one per plan query
    assert len(openalex.queries) == 2
    assert result.total_returned == 4
    assert set(result.sources_succeeded) == {SourceName.ARXIV, SourceName.OPENALEX}


async def test_plan_drives_the_query_parameters() -> None:
    source = FakeSource(SourceName.ARXIV)

    await service([source]).discover(a_plan(queries=["metastable failures"], year_from=2022))

    query = source.queries[0]
    assert query.text == "metastable failures"
    assert query.year_from == 2022
    assert "metastable" in query.terms


async def test_one_failing_source_does_not_fail_the_run() -> None:
    """Five public APIs will not all be up; a degraded result still beats no result."""
    healthy = FakeSource(SourceName.ARXIV, [paper("Metastable failures", SourceName.ARXIV)])
    broken = FakeSource(
        SourceName.SEMANTIC_SCHOLAR, error=SourceRateLimitedError("429", source="s2")
    )

    result = await service([healthy, broken]).discover(a_plan())

    assert result.candidates
    assert result.sources_succeeded == [SourceName.ARXIV]
    assert result.sources_failed == [SourceName.SEMANTIC_SCHOLAR]
    report = next(r for r in result.reports if r.source is SourceName.SEMANTIC_SCHOLAR)
    assert report.error is not None and "source_rate_limited" in report.error


async def test_a_failing_source_stops_after_the_first_error() -> None:
    broken = FakeSource(SourceName.CROSSREF, error=SourceUnavailableError("down", source="cr"))

    await service([broken]).discover(a_plan(queries=["a", "b", "c"]))

    assert len(broken.queries) == 1  # not 3


async def test_all_sources_failing_returns_an_empty_but_valid_result() -> None:
    broken = FakeSource(SourceName.ARXIV, error=SourceUnavailableError("down", source="arxiv"))

    result = await service([broken]).discover(a_plan())

    assert result.candidates == []
    assert result.sources_failed == [SourceName.ARXIV]


async def test_duplicates_across_sources_are_collapsed() -> None:
    shared = PaperIdentifiers(doi="10.1145/123")
    arxiv = FakeSource(
        SourceName.ARXIV, [paper("Metastable failures", SourceName.ARXIV, identifiers=shared)]
    )
    crossref = FakeSource(
        SourceName.CROSSREF,
        [paper("Metastable failures", SourceName.CROSSREF, identifiers=shared)],
    )

    result = await service([arxiv, crossref]).discover(a_plan(queries=["metastable failures"]))

    assert result.total_returned == 2
    assert len(result.candidates) == 1
    assert result.duplicates_removed == 1


async def test_results_are_ranked() -> None:
    source = FakeSource(
        SourceName.ARXIV,
        [
            paper("Unrelated pottery glazing", SourceName.ARXIV),
            paper("Metastable failures and overload", SourceName.ARXIV),
        ],
    )

    result = await service([source]).discover(a_plan(queries=["metastable failures"]))

    assert result.candidates[0].paper.title == "Metastable failures and overload"
    assert result.candidates[0].score > result.candidates[-1].score


async def test_candidate_cap_is_applied() -> None:
    source = FakeSource(
        SourceName.ARXIV,
        [
            paper(
                f"Metastable failures under {word} conditions",
                SourceName.ARXIV,
                identifiers=PaperIdentifiers(doi=f"10.1/{word}"),
            )
            for word in [
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                "iota",
                "kappa",
            ]
        ],
    )

    result = await service([source], settings=DiscoverySettings(max_candidates=5)).discover(
        a_plan(queries=["metastable"])
    )

    assert len(result.candidates) == 5


async def test_query_cap_is_applied() -> None:
    source = FakeSource(SourceName.ARXIV)

    await service([source], settings=DiscoverySettings(max_queries=1)).discover(
        a_plan(queries=["a query", "b query", "c query"])
    )

    assert len(source.queries) == 1


async def test_require_retrievable_filters_metadata_only_papers() -> None:
    source = FakeSource(
        SourceName.ARXIV,
        [
            paper("Metastable with pdf", SourceName.ARXIV, pdf_url="https://x/1.pdf"),
            paper("Metastable without pdf", SourceName.ARXIV),
        ],
    )

    result = await service([source], settings=DiscoverySettings(require_retrievable=True)).discover(
        a_plan(queries=["metastable"])
    )

    assert [c.paper.title for c in result.candidates] == ["Metastable with pdf"]


async def test_sources_are_queried_concurrently() -> None:
    """Providers each own a rate limiter; serialising them would multiply wall time."""
    slow = [FakeSource(name, delay=0.05) for name in (SourceName.ARXIV, SourceName.OPENALEX)]

    started = asyncio.get_running_loop().time()
    await service(slow).discover(a_plan(queries=["one query"]))
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.09  # ~0.05 concurrent, not ~0.10 serial


async def test_no_enabled_sources_is_not_an_error() -> None:
    result = await service([]).discover(a_plan())

    assert result.candidates == []
    assert result.reports == []


async def test_discovered_papers_are_persisted(tmp_path: Path) -> None:
    repository = JsonPaperRepository(tmp_path / "metadata")
    source = FakeSource(
        SourceName.ARXIV,
        [
            paper(
                "Metastable failures", SourceName.ARXIV, identifiers=PaperIdentifiers(doi="10.1/a")
            )
        ],
    )

    await service([source], repository=repository).discover(a_plan(), run_id="run-1")

    stored = await repository.list_all()
    assert len(stored) == 1
    assert stored[0].paper.identifiers.doi == "10.1/a"
    assert stored[0].run_ids == ["run-1"]


async def test_local_papers_are_recorded_as_already_downloaded(tmp_path: Path) -> None:
    repository = JsonPaperRepository(tmp_path / "metadata")
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    source = FakeSource(
        SourceName.MANUAL, [paper("Metastable failures", SourceName.MANUAL, local_path=pdf)]
    )

    await service([source], repository=repository).discover(a_plan())

    stored = (await repository.list_all())[0]
    assert stored.processing.downloaded is True
    assert stored.pdf_path == pdf
