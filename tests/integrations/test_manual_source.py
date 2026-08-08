"""Manual provider tests run against the real committed PDF collection."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.core.exceptions import PaperNotFoundError
from researchagent.core.interfaces.paper_source import SearchQuery
from researchagent.integrations.manual import ManualPaperSource
from researchagent.models.paper import SourceName


def test_loads_the_real_collection(manual_source: ManualPaperSource) -> None:
    papers = manual_source.load_all()

    assert len(papers) >= 17
    assert all(paper.provider is SourceName.MANUAL for paper in papers)
    assert all(paper.local_path is not None and paper.local_path.is_file() for paper in papers)


def test_filename_convention_becomes_metadata(manual_source: ManualPaperSource) -> None:
    papers = {paper.id: paper for paper in manual_source.load_all()}

    first = papers["manual:01"]
    assert first.title == "Metastable Failures in Distributed Systems"
    assert first.source_metadata["priority"] == "P1"
    assert first.source_metadata["index"] == "01"
    assert first.source_metadata["size_bytes"] > 0


def test_letter_suffixed_indices_are_distinct_papers(manual_source: ManualPaperSource) -> None:
    ids = {paper.id for paper in manual_source.load_all()}

    assert {"manual:05a", "manual:05b"} <= ids


def test_a_bare_number_is_never_read_as_a_year(manual_source: ManualPaperSource) -> None:
    """`09_[P3]_A2A_Issue_1987_...` must not be dated 1987 — never invent metadata."""
    papers = {paper.id: paper for paper in manual_source.load_all()}

    assert papers["manual:09"].year is None
    assert "1987" in papers["manual:09"].title


def test_iso_dates_in_filenames_are_trusted(manual_source: ManualPaperSource) -> None:
    papers = {paper.id: paper for paper in manual_source.load_all()}

    assert papers["manual:05a"].year == 2026


def test_absent_metadata_stays_empty(manual_source: ManualPaperSource) -> None:
    paper = manual_source.load_all()[0]

    assert paper.abstract is None
    assert paper.venue is None
    assert paper.authors == []
    assert paper.identifiers.any_present() is False
    assert paper.citation_count is None


async def test_search_matches_on_title_tokens(manual_source: ManualPaperSource) -> None:
    results = await manual_source.search(SearchQuery(text="metastable failures", limit=10))

    titles = [paper.title for paper in results]
    assert any("Metastable" in title for title in titles)
    assert all("Metastable" in t or "Failure" in t or "failures" in t.lower() for t in titles)


async def test_search_ranks_stronger_overlap_first(manual_source: ManualPaperSource) -> None:
    results = await manual_source.search(SearchQuery(text="multi-agent LLM systems fail", limit=5))

    assert results
    assert "Multi-Agent" in results[0].title


async def test_search_uses_plan_keywords_too(manual_source: ManualPaperSource) -> None:
    results = await manual_source.search(
        SearchQuery(text="nothing matches this phrase", terms=["LangGraph"], limit=5)
    )

    assert any("LangGraph" in paper.title for paper in results)


async def test_search_respects_the_limit(manual_source: ManualPaperSource) -> None:
    results = await manual_source.search(SearchQuery(text="protocol specification", limit=2))

    assert len(results) <= 2


async def test_search_with_no_match_returns_empty(manual_source: ManualPaperSource) -> None:
    assert await manual_source.search(SearchQuery(text="zzzqqq unmatchable", limit=5)) == []


async def test_get_paper_by_index(manual_source: ManualPaperSource) -> None:
    paper = await manual_source.get_paper("manual:03")

    assert paper is not None
    assert paper.title == "Congestion Avoidance and Control"
    assert await manual_source.get_paper("manual:999") is None


async def test_download_returns_the_existing_file_without_copying(
    manual_source: ManualPaperSource, tmp_path: Path
) -> None:
    """The user's collection must never be duplicated or relocated."""
    paper = manual_source.load_all()[0]
    destination = tmp_path / "copy.pdf"

    result = await manual_source.download_pdf(paper, destination)

    assert result == paper.local_path
    assert not destination.exists()


async def test_download_of_a_missing_file_raises(
    manual_source: ManualPaperSource, tmp_path: Path
) -> None:
    paper = manual_source.load_all()[0].model_copy(update={"local_path": tmp_path / "gone.pdf"})

    with pytest.raises(PaperNotFoundError):
        await manual_source.download_pdf(paper, tmp_path / "out.pdf")


async def test_health_reports_the_collection_size(manual_source: ManualPaperSource) -> None:
    health = await manual_source.health()

    assert health.healthy is True
    assert health.detail is not None and "local papers" in health.detail


async def test_missing_directory_is_unhealthy_not_fatal(tmp_path: Path) -> None:
    source = ManualPaperSource(tmp_path / "does-not-exist")

    health = await source.health()

    assert health.healthy is False
    assert source.load_all() == []
    assert await source.search(SearchQuery(text="anything")) == []


async def test_unconventional_filename_is_still_indexed(tmp_path: Path) -> None:
    (tmp_path / "no-convention-here.pdf").write_bytes(b"%PDF-1.4")

    papers = ManualPaperSource(tmp_path).load_all()

    assert len(papers) == 1
    assert papers[0].title == "no-convention-here"
